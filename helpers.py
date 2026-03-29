"""Helper utilities wrapping pyinaturalist for small analysis tasks.

This module provides convenience functions for loading a local iNaturalist
API key, fetching a user's observations and nearby seasonal/common species,
and summarizing park-level species frequency. Changes here emphasize:
- runtime key loading (no import-time side effects)
- robust handling of missing fields from API responses
- non-blocking image display and safe normalization logic

Logging: the module uses a package-level `logger`. Do not call
``logging.basicConfig`` in libraries; configure logging in your application.
"""

import json
import requests
import pandas as pd
import numpy as np
import datetime as dt
from typing import Tuple, Union, Optional, Any, Iterable
from time import sleep
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from io import BytesIO
try:
    import pyinaturalist as inat
    HAS_PYINAT = True
except Exception:
    inat = None
    HAS_PYINAT = False
try:
    import dill
    HAS_DILL = True
except Exception:
    dill = None
    HAS_DILL = False
import pickle
import os
try:
    import ipyplot
    HAS_IPYPLOT = True
except Exception:
    ipyplot = None
    HAS_IPYPLOT = False
import logging
try:
    from IPython.display import HTML, display, Markdown
    HAS_IPYTHON_DISPLAY = True
except Exception:
    HTML = None
    display = None
    Markdown = None
    HAS_IPYTHON_DISPLAY = False

logger = logging.getLogger(__name__)

VERTEBRATES_TAXON_ID = 355675
PHENOTYPE_IMAGE_KINDS = {'flowers', 'fruits', 'butterflies', 'caterpillars'}
DEFAULT_NATIVITY_PLACE_ID = 1297  # Virginia
OBSERVATION_ROW_COLUMNS = [
    'obs_id',
    'observed_on',
    'observed_at',
    'created_at',
    'quality_grade',
    'query_kind',
    'query_project_id',
    'observer_id',
    'observer_login',
    'taxon_id',
    'taxon_name',
    'taxon_preferred_common_name',
    'taxon_rank',
    'iconic_taxon_name',
    'latitude',
    'longitude',
    'place_guess',
    'positional_accuracy',
    'identifications_count',
    'all_identification_count',
    'non_owner_identification_count',
    'num_identification_agreements',
    'num_identification_disagreements',
    'all_identification_timestamps',
    'non_owner_identification_timestamps',
    'first_identification_at',
    'first_non_owner_identification_at',
    'first_identification_delay_days',
    'first_non_owner_identification_delay_days',
    'photo_url',
]

# Minimal field set for the iNaturalist /observations endpoint.  Requesting
# only these fields reduces per-response payload by 50-80%, cutting bandwidth
# and server work.  Covers every path in _normalize_observation_record,
# _identification_time_fields, _parse_observation_location, and
# _extract_observation_photo_url.
OBSERVATION_FIELDS_MINIMAL = {
    'id': True,
    'observed_on': True,
    'time_observed_at': True,
    'created_at': True,
    'quality_grade': True,
    'place_guess': True,
    'positional_accuracy': True,
    'identifications_count': True,
    'num_identification_agreements': True,
    'num_identification_disagreements': True,
    'iconic_taxon_name': True,
    'taxon_id': True,
    'location': True,
    'user': {'id': True, 'login': True},
    'taxon': {
        'id': True,
        'name': True,
        'preferred_common_name': True,
        'rank': True,
        'iconic_taxon_name': True,
        'default_photo': {'medium_url': True},
    },
    'geojson': True,
    'identifications': {
        'created_at': True,
        'own_observation': True,
        'user': {'id': True},
    },
    'observation_photos': {
        'photo': {'url': True},
    },
}


def _derive_place_id_from_location(
    session: requests.Session,
    lat: float,
    lng: float,
    fallback_place_id: int = DEFAULT_NATIVITY_PLACE_ID,
) -> int:
    """Derive a regional place id from coordinates (prefer state/province)."""
    endpoint = 'https://api.inaturalist.org/v1/places/nearby'
    try:
        response = session.get(endpoint, params={'lat': lat, 'lng': lng, 'per_page': 30}, timeout=20)
        response.raise_for_status()
        nearby = response.json().get('results', [])
    except Exception as e:
        logger.debug('Failed to derive nativity place from location (%s,%s): %s', lat, lng, e)
        return int(fallback_place_id)

    normalized = []
    for item in nearby:
        if not isinstance(item, dict):
            continue
        place = item.get('place') if isinstance(item.get('place'), dict) else item
        if isinstance(place, dict) and place.get('id') is not None:
            normalized.append(place)

    for place in normalized:
        if place.get('admin_level') == 10:
            return int(place['id'])

    if normalized:
        return int(normalized[0]['id'])
    return int(fallback_place_id)


def _resolve_nativity_place_id(
    session: requests.Session,
    nativity_place_id: Union[int, str, None],
    places: Optional[list[int]],
    loc: Optional[tuple[float, float, float]],
) -> Optional[int]:
    """Resolve nativity place id from explicit value, auto mode, or global mode."""
    if nativity_place_id is None:
        return None

    if isinstance(nativity_place_id, str):
        mode = nativity_place_id.strip().lower()
        if mode != 'auto':
            raise ValueError("nativity_place_id must be an int, None, or 'auto'")

        if places and len(places) > 0:
            return int(places[0])
        if isinstance(loc, tuple) and len(loc) == 3:
            return _derive_place_id_from_location(session=session, lat=float(loc[0]), lng=float(loc[1]))
        return int(DEFAULT_NATIVITY_PLACE_ID)

    return int(nativity_place_id)


def _taxa_for_kind(kind: str) -> dict:
    """Return iNaturalist API taxa filters for a supported kind."""
    normalized_kind = (kind or '').lower().strip()
    if normalized_kind == 'fruit':
        normalized_kind = 'fruits'

    if normalized_kind == 'any':
        return {}
    if normalized_kind == 'plants':
        return {'taxon_id': 47126}
    if normalized_kind == 'flowers':
        return {'taxon_id': 47126, 'term_id': 12, 'term_value_id': 13}
    if normalized_kind == 'fruits':
        return {'taxon_id': 47126, 'term_id': 12, 'term_value_id': 14}
    if normalized_kind == 'mushrooms':
        return {'taxon_id': 47170}
    if normalized_kind == 'animals':
        return {'taxon_id': 1}
    if normalized_kind == 'fish':
        return {'taxon_id': 47178}
    if normalized_kind == 'mammals':
        return {'taxon_id': 40151}
    if normalized_kind == 'birds':
        return {'taxon_id': 3}
    if normalized_kind == 'herps':
        return {'taxon_id': [26036, 20978]}
    if normalized_kind == 'wugs':
        return {'taxon_id': 1, 'without_taxon_id': VERTEBRATES_TAXON_ID}
    if normalized_kind == 'butterflies':
        return {'taxon_id': 47157, 'term_id': 1, 'term_value_id': 2}
    if normalized_kind == 'caterpillars':
        return {'taxon_id': 47157, 'term_id': 1, 'term_value_id': 6}

    raise ValueError(f"kind '{kind}' not implemented")


def _extract_photo_url(observation: dict) -> Optional[str]:
    photos = observation.get('photos') if isinstance(observation, dict) else None
    if not isinstance(photos, list):
        return None
    for photo in photos:
        if isinstance(photo, dict):
            photo_url = photo.get('url')
            if photo_url:
                return photo_url.replace('square', 'medium')
    return None


def _get_filtered_photo_url(
    session: requests.Session,
    taxon_id: int,
    taxa_filters: dict,
    place_filters: dict,
    time_filters: list[dict],
    fallback_url: Optional[str],
) -> Optional[str]:
    """Pick a representative photo from observations matching current filters."""
    query_params = {
        'taxon_id': taxon_id,
        'verifiable': 'true',
        'photos': 'true',
        'per_page': 10,
        'order_by': 'votes',
        'order': 'desc',
        **place_filters,
    }
    for key in ('term_id', 'term_value_id'):
        if key in taxa_filters:
            query_params[key] = taxa_filters[key]

    endpoint = 'https://api.inaturalist.org/v1/observations'

    def _query_one(params: dict) -> Optional[str]:
        try:
            response = session.get(endpoint, params=params, timeout=30)
            response.raise_for_status()
            for observation in response.json().get('results', []):
                photo_url = _extract_photo_url(observation)
                if photo_url:
                    return photo_url
        except Exception as e:
            logger.debug('Could not fetch filtered photo for taxon %s: %s', taxon_id, e)
        return None

    for tf in time_filters:
        month = tf.get('month')
        days = tf.get('day')
        if month is None or not days:
            continue
        photo_url = _query_one({**query_params, 'month': int(month), 'day': [int(d) for d in days]})
        if photo_url:
            return photo_url

    photo_url = _query_one(query_params)
    if photo_url:
        return photo_url

    return fallback_url


def _infer_nativity_from_row(row: pd.Series) -> Optional[str]:
    """Infer nativity label from flattened taxon columns when present."""
    candidates = [
        row.get('taxon.establishment_means.establishment_means'),
        row.get('taxon.establishment_means.place.id'),
        row.get('taxon.native'),
        row.get('taxon.introduced'),
        row.get('taxon.endemic'),
    ]

    establishment = candidates[0]
    if isinstance(establishment, str) and establishment.strip():
        value = establishment.strip().lower()
        if value in {'native', 'introduced', 'endemic'}:
            return value.capitalize()

    if candidates[4] is True:
        return 'Endemic'
    if candidates[2] is True:
        return 'Native'
    if candidates[3] is True:
        return 'Introduced'
    return None


def _lookup_nativity_via_species_counts(
    session: requests.Session,
    taxon_id: int,
    nativity_place_id: Optional[int] = None,
) -> str:
    """Classify nativity with small iNaturalist count queries for one taxon.
    """
    base = {
        'taxon_id': taxon_id,
        'verifiable': 'true',
        'per_page': 0,
    }
    if nativity_place_id is not None:
        base['place_id'] = int(nativity_place_id)

    endpoint = 'https://api.inaturalist.org/v1/observations/species_counts'

    status_order = [('endemic', 'Endemic'), ('introduced', 'Introduced'), ('native', 'Native')]

    def _search_statuses(query_base: dict) -> Optional[str]:
        for key, label in status_order:
            params = {**query_base, key: 'true'}
            try:
                response = session.get(endpoint, params=params, timeout=20)
                response.raise_for_status()
                payload = response.json()
                if payload.get('total_results', 0) > 0:
                    return label
            except Exception as e:
                logger.debug('Nativity lookup failed for taxon %s (%s): %s', taxon_id, key, e)
                continue
        return None

    result = _search_statuses(base)
    if result:
        return result

    return 'Unknown'


def _batch_lookup_nativity(
    session: requests.Session,
    taxon_ids: list[int],
    nativity_place_id: Optional[int] = None,
    chunk_size: int = 100,
) -> dict[int, str]:
    """Classify nativity for many taxa in bulk using batched species_counts queries.

    Instead of 3 API calls per taxon, this issues 3 calls per *chunk* (one per
    establishment status: endemic, introduced, native).  For *N* unique taxa the
    call count drops from 3·N worst-case to 3·ceil(N/chunk_size).
    """
    if not taxon_ids:
        return {}

    endpoint = 'https://api.inaturalist.org/v1/observations/species_counts'
    status_order = [('endemic', 'Endemic'), ('introduced', 'Introduced'), ('native', 'Native')]
    result: dict[int, str] = {}

    for i in range(0, len(taxon_ids), chunk_size):
        chunk = taxon_ids[i:i + chunk_size]
        unresolved = set(chunk)
        base: dict[str, Any] = {
            'taxon_id': ','.join(str(t) for t in chunk),
            'verifiable': 'true',
            'per_page': 0,
        }
        if nativity_place_id is not None:
            base['place_id'] = int(nativity_place_id)

        for key, label in status_order:
            if not unresolved:
                break
            # Narrow to unresolved taxa only to reduce noise.
            params = {
                **base,
                'taxon_id': ','.join(str(t) for t in unresolved),
                key: 'true',
            }
            try:
                response = session.get(endpoint, params=params, timeout=20)
                response.raise_for_status()
                payload = response.json()
                # per_page=0 means results list is empty, but total_results > 0
                # tells us *at least one* taxon in the set has this status.
                if payload.get('total_results', 0) == 0:
                    continue
                # Re-query with per_page large enough to get taxon IDs back
                detail_params = {**params, 'per_page': len(unresolved)}
                detail_resp = session.get(endpoint, params=detail_params, timeout=20)
                detail_resp.raise_for_status()
                for rec in detail_resp.json().get('results', []):
                    tid = rec.get('taxon', {}).get('id') if isinstance(rec.get('taxon'), dict) else rec.get('taxon.id')
                    if tid is not None:
                        tid = int(tid)
                        if tid in unresolved:
                            result[tid] = label
                            unresolved.discard(tid)
            except Exception as e:
                logger.debug('Batch nativity lookup failed for chunk (%s): %s', key, e)
                continue

        # Anything left is unknown
        for tid in unresolved:
            result[tid] = 'Unknown'

    return result


def _format_photo_title(common_name: str, scientific_name: str, nativity: str) -> str:
    return f"{common_name}\n{scientific_name}\nNativity: {nativity}"

def load_api_key(fallback_path: str = 'pyinaturalistkey.pkd') -> Union[str, None]:
    """Load an iNaturalist API key from multiple sources.

    Order of attempts:
    1. Environment variables: ``INAT_TOKEN``, ``INAT_API_KEY``, ``PYINAT_API_KEY``, ``INAT_KEY``
    2. System keyring (optional ``keyring`` package)
    3. Fallback dill file lookup (legacy behavior) using the first readable path:
       - ``fallback_path`` as provided
       - ``cwd/<filename>`` when ``fallback_path`` is relative
       - ``<helpers.py dir>/<filename>`` when ``fallback_path`` is relative
       - ``<project root>/pynat/<filename>`` when ``fallback_path`` is relative

    Returns:
        The API key string when found, otherwise ``None``.

    Notes:
        This helper will not raise on missing keys; it returns ``None`` so
        callers can decide whether to proceed or surface an error.
    """
    # 1) environment
    for name in ('INAT_TOKEN', 'INAT_API_KEY', 'PYINAT_API_KEY', 'INAT_KEY'):
        val = os.environ.get(name)
        if val:
            return val

    # 2) keyring
    try:
        import keyring  # optional dependency
    except Exception:
        keyring = None

    if keyring is not None:
        try:
            val = keyring.get_password('pyinaturalist', 'api_key')
            if val:
                return val
        except Exception:
            pass

    # 3) fallback dill file
    candidates = [fallback_path]
    if not os.path.isabs(fallback_path):
        filename = os.path.basename(fallback_path)
        module_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.dirname(module_dir)
        candidates.extend([
            os.path.join(os.getcwd(), filename),
            os.path.join(module_dir, filename),
            os.path.join(project_root, 'pynat', filename),
        ])

    seen_paths = set()
    for path in candidates:
        abs_path = os.path.abspath(path)
        if abs_path in seen_paths:
            continue
        seen_paths.add(abs_path)
        if not HAS_DILL:
            continue
        try:
            with open(abs_path, 'rb') as f:
                return dill.load(f)
        except (OSError, EOFError, pickle.UnpicklingError) as e:
            logger.debug('Could not load API key from %s: %s', abs_path, e)

    return None

## (import-time API key loading removed; use `load_api_key()` at runtime)


def get_mine(uname: str,
             lookback_in_days: int = None,
             STRT: Optional[dt.date] = None,
             FNSH: Optional[dt.date] = None,
             api_key: str = None) -> None:
    """Print observations for ``uname`` and display their photos.

    Parameters
    ----------
    uname:
        iNaturalist username to query.
    lookback_in_days:
        If provided, subtract this number of days from ``STRT`` to form
        the query start date.
    STRT, FNSH:
        Optional start and finish dates; if omitted they default to
        ``date.today()`` and ``date.today()+1`` respectively.
    api_key:
        Optional API key; if ``None`` the function will call
        :func:`load_api_key` to attempt to locate a key.

    Returns
    -------
    None

    Side effects
    ------------
    - Prints a compact summary line per observation.
    - Attempts to display photos with ``ipyplot``; failures are logged
      and do not abort the loop.
    """
    logger.debug('get_mine called for user=%s lookback=%s', uname, lookback_in_days)
    
    # Define the base URL for the iNaturalist API
    base_url = "https://api.inaturalist.org/v1/observations"

    # Ensure we have an API key at call time (non-destructive)
    if api_key is None:
        api_key = load_api_key()

    # runtime defaults for dates
    if STRT is None:
        STRT = dt.date.today()
    if FNSH is None:
        FNSH = STRT + dt.timedelta(days=1)

    # Define scope
    if lookback_in_days:
        start_date = STRT - dt.timedelta(days=lookback_in_days)
    else:
        start_date = STRT
    end_date = FNSH

    response = inat.get_observations(user_id=[uname],
                                     d1=start_date,
                                     d2=end_date,
                                     page='all')

    df = pd.json_normalize(response.get('results', []))
    try:
        df.sort_values('observed_on', inplace=True)
    except Exception as e:
        logger.debug('Could not sort observations: %s', e)

    for index, row in df.iterrows():
        observed_on = row.get('observed_on')
        # parse observed_on robustly (may be string or datetime-like)
        try:
            obs_dt = pd.to_datetime(observed_on, errors='coerce')
            if pd.isna(obs_dt):
                date_str = str(observed_on)
            else:
                date_str = obs_dt.strftime('%Y%m%d')
        except Exception:
            date_str = str(observed_on)

        taxon_name = row.get('taxon.name', 'Unknown')
        species_guess = row.get('species_guess', 'Unknown')
        obs_id = row.get('id')
        print(f"\n\n{date_str} {taxon_name} ({species_guess}) [inat obs id: {obs_id}]   ref: www.inaturalist.org/observations/{obs_id}")

        photos = row.get('photos') if isinstance(row.get('photos'), list) else []
        if not photos:
            logger.debug('No photos for observation %s', obs_id)
            continue

        images = []
        for each_obs_photo in photos:
            if isinstance(each_obs_photo, dict):
                url = each_obs_photo.get('url')
                if url:
                    images.append(url.replace('square', 'small'))

        if not images:
            logger.debug('No valid photo URLs for observation %s', obs_id)
            continue

        if HAS_IPYPLOT:
            try:
                ipyplot.plot_images(images)
            except Exception as e:
                logger.warning('Failed to display images for %s: %s', obs_id, e)
        else:
            logger.warning('ipyplot not available; skipping image display for %s', obs_id)


def get_inat_session(token: Optional[str] = None,
                     use_cache: bool = True,
                     cache_name: str = 'inat_cache',
                     expire_seconds: int = 24 * 3600) -> requests.Session:
    """Create a requests session configured for iNaturalist.

    Uses explicit ``token`` first, then environment variables, then
    :func:`load_api_key` as a final fallback.
    Installs `requests_cache` if available and `use_cache` is True.
    Sets the appropriate Authorization header when a token is present.
    """
    token = token or os.environ.get('INAT_TOKEN') or os.environ.get('INAT_API_KEY') or os.environ.get('PYINAT_API_KEY') or os.environ.get('INAT_KEY')
    if not token:
        token = load_api_key()

    if use_cache:
        try:
            import requests_cache
            # install_cache is safe to call multiple times; will reuse existing cache
            requests_cache.install_cache(cache_name, expire_after=expire_seconds)
        except Exception:
            logger.debug('requests_cache not available; continuing without cache')

    session = requests.Session()
    if token:
        # iNaturalist expects Token token="..." in the Authorization header for some clients
        session.headers.update({'Authorization': f'Token token="{token}"'})
    return session


def _normalize_project_id_value(project_id: Union[int, str, Iterable[Union[int, str]], None]) -> Union[int, str, list[Union[int, str]], None]:
    if project_id is None:
        return None
    if isinstance(project_id, (list, tuple, set)):
        normalized = []
        for value in project_id:
            if isinstance(value, str) and value.strip().isdigit():
                normalized.append(int(value.strip()))
            else:
                normalized.append(value)
        return normalized
    if isinstance(project_id, str) and project_id.strip().isdigit():
        return int(project_id.strip())
    return project_id


def _serialize_query_value(value: Any) -> Any:
    if isinstance(value, bool):
        return 'true' if value else 'false'
    return value


def _rest_params_from_query(query: dict[str, Any]) -> dict[str, Any]:
    return {key: _serialize_query_value(value) for key, value in query.items() if value is not None}


def _build_observation_query(kind: str,
                             places: Optional[list[int]],
                             loc: Optional[tuple[float, float, float]],
                             project_id: Union[int, str, Iterable[Union[int, str]], None],
                             taxa_filters: Optional[dict[str, Any]],
                             d1: Optional[Union[str, dt.date, dt.datetime]],
                             d2: Optional[Union[str, dt.date, dt.datetime]],
                             quality_grade: Optional[str],
                             verifiable: bool,
                             order_by: str,
                             order: str) -> dict[str, Any]:
    assert not (places and loc), "only one of places and loc should be provided"

    query = {
        **_taxa_for_kind(kind),
        'project_id': _normalize_project_id_value(project_id),
        'quality_grade': quality_grade,
        'verifiable': verifiable,
        'order_by': order_by,
        'order': order,
    }
    if taxa_filters:
        query.update({key: value for key, value in taxa_filters.items() if value is not None})
    if d1 is not None:
        query['d1'] = d1
    if d2 is not None:
        query['d2'] = d2

    if places:
        query['place_id'] = [int(place_id) for place_id in places]
    elif isinstance(loc, tuple):
        if len(loc) != 3:
            raise ValueError('expected loc triple of lat,long,radius')
        query.update({'lat': float(loc[0]), 'lng': float(loc[1]), 'radius': float(loc[2])})

    return {key: value for key, value in query.items() if value is not None}


def _parse_observation_location(observation: dict[str, Any]) -> tuple[Optional[float], Optional[float]]:
    geojson = observation.get('geojson') if isinstance(observation.get('geojson'), dict) else None
    if geojson:
        coordinates = geojson.get('coordinates')
        if isinstance(coordinates, (list, tuple)) and len(coordinates) >= 2:
            try:
                return float(coordinates[1]), float(coordinates[0])
            except (TypeError, ValueError):
                pass

    location = observation.get('location')
    if isinstance(location, str) and ',' in location:
        parts = [part.strip() for part in location.split(',', 1)]
        try:
            return float(parts[0]), float(parts[1])
        except (TypeError, ValueError):
            return None, None
    return None, None


def _extract_observation_photo_url(observation: dict[str, Any], taxon: dict[str, Any]) -> Optional[str]:
    observation_photos = observation.get('observation_photos') if isinstance(observation.get('observation_photos'), list) else []
    for item in observation_photos:
        if not isinstance(item, dict):
            continue
        photo = item.get('photo') if isinstance(item.get('photo'), dict) else item
        photo_url = photo.get('url') if isinstance(photo, dict) else None
        if photo_url:
            return str(photo_url).replace('square', 'medium')

    fallback = taxon.get('default_photo') if isinstance(taxon.get('default_photo'), dict) else None
    fallback_url = fallback.get('medium_url') if isinstance(fallback, dict) else None
    if fallback_url:
        return str(fallback_url)
    return None


def _to_iso_timestamp(value: Any) -> Optional[str]:
    timestamp = pd.to_datetime(value, utc=True, errors='coerce')
    if pd.isna(timestamp):
        return None
    return timestamp.isoformat()


def _delay_in_days(observed_value: Any, identified_value: Any) -> Optional[float]:
    observed = pd.to_datetime(observed_value, utc=True, errors='coerce')
    identified = pd.to_datetime(identified_value, utc=True, errors='coerce')
    if pd.isna(observed) or pd.isna(identified):
        return None
    return float((identified - observed).total_seconds()) / 86400.0


def _identification_time_fields(observation: dict[str, Any]) -> dict[str, Any]:
    identifications = observation.get('identifications') if isinstance(observation.get('identifications'), list) else []
    observer = observation.get('user') if isinstance(observation.get('user'), dict) else {}
    observer_id = observer.get('id')
    earliest_identification = None
    earliest_non_owner = None
    all_identification_timestamps: list[str] = []
    non_owner_identification_timestamps: list[str] = []

    for identification in identifications:
        if not isinstance(identification, dict):
            continue
        created_at = pd.to_datetime(identification.get('created_at'), utc=True, errors='coerce')
        if pd.isna(created_at):
            continue
        created_iso = created_at.isoformat()
        all_identification_timestamps.append(created_iso)
        if earliest_identification is None or created_at < earliest_identification:
            earliest_identification = created_at

        ident_user = identification.get('user') if isinstance(identification.get('user'), dict) else {}
        ident_user_id = ident_user.get('id')
        if identification.get('own_observation') is True:
            continue
        if observer_id is not None and ident_user_id == observer_id:
            continue
        non_owner_identification_timestamps.append(created_iso)
        if earliest_non_owner is None or created_at < earliest_non_owner:
            earliest_non_owner = created_at

    first_identification_at = earliest_identification.isoformat() if earliest_identification is not None else None
    first_non_owner_identification_at = earliest_non_owner.isoformat() if earliest_non_owner is not None else None
    return {
        'all_identification_timestamps': all_identification_timestamps,
        'non_owner_identification_timestamps': non_owner_identification_timestamps,
        'all_identification_count': len(all_identification_timestamps),
        'non_owner_identification_count': len(non_owner_identification_timestamps),
        'first_identification_at': first_identification_at,
        'first_non_owner_identification_at': first_non_owner_identification_at,
        'first_identification_delay_days': _delay_in_days(observation.get('observed_on'), first_identification_at),
        'first_non_owner_identification_delay_days': _delay_in_days(observation.get('observed_on'), first_non_owner_identification_at),
    }


def _normalize_observation_record(observation: dict[str, Any],
                                  query_kind: str,
                                  query_project_id: Union[int, str, Iterable[Union[int, str]], None]) -> dict[str, Any]:
    taxon = observation.get('taxon') if isinstance(observation.get('taxon'), dict) else {}
    observer = observation.get('user') if isinstance(observation.get('user'), dict) else {}
    latitude, longitude = _parse_observation_location(observation)
    identification_fields = _identification_time_fields(observation)

    observed_on = observation.get('observed_on')
    if observed_on is None:
        observed_timestamp = _to_iso_timestamp(observation.get('time_observed_at'))
        if observed_timestamp:
            observed_on = observed_timestamp[:10]

    return {
        'obs_id': observation.get('id'),
        'observed_on': observed_on,
        'observed_at': _to_iso_timestamp(observation.get('time_observed_at') or observation.get('observed_on')),
        'created_at': _to_iso_timestamp(observation.get('created_at')),
        'quality_grade': observation.get('quality_grade'),
        'query_kind': (query_kind or 'any').lower().strip() or 'any',
        'query_project_id': None if query_project_id is None else str(query_project_id),
        'observer_id': observer.get('id'),
        'observer_login': observer.get('login'),
        'taxon_id': taxon.get('id') or observation.get('taxon_id'),
        'taxon_name': taxon.get('name'),
        'taxon_preferred_common_name': taxon.get('preferred_common_name'),
        'taxon_rank': taxon.get('rank'),
        'iconic_taxon_name': taxon.get('iconic_taxon_name') or observation.get('iconic_taxon_name'),
        'latitude': latitude,
        'longitude': longitude,
        'place_guess': observation.get('place_guess'),
        'positional_accuracy': observation.get('positional_accuracy'),
        'identifications_count': observation.get('identifications_count', len(observation.get('identifications', [])) if isinstance(observation.get('identifications'), list) else 0),
        'all_identification_count': identification_fields['all_identification_count'],
        'non_owner_identification_count': identification_fields['non_owner_identification_count'],
        'num_identification_agreements': observation.get('num_identification_agreements'),
        'num_identification_disagreements': observation.get('num_identification_disagreements'),
        'all_identification_timestamps': identification_fields['all_identification_timestamps'],
        'non_owner_identification_timestamps': identification_fields['non_owner_identification_timestamps'],
        'first_identification_at': identification_fields['first_identification_at'],
        'first_non_owner_identification_at': identification_fields['first_non_owner_identification_at'],
        'first_identification_delay_days': identification_fields['first_identification_delay_days'],
        'first_non_owner_identification_delay_days': identification_fields['first_non_owner_identification_delay_days'],
        'photo_url': _extract_observation_photo_url(observation, taxon),
    }


def get_observation_rows(kind: str = 'any',
                         places: Optional[list[int]] = None,
                         loc: Optional[tuple[float, float, float]] = None,
                         project_id: Union[int, str, Iterable[Union[int, str]], None] = None,
                         taxa_filters: Optional[dict[str, Any]] = None,
                         d1: Optional[Union[str, dt.date, dt.datetime]] = None,
                         d2: Optional[Union[str, dt.date, dt.datetime]] = None,
                         quality_grade: Optional[str] = None,
                         verifiable: bool = True,
                         per_page: int = 100,
                         max_pages: int = 10,
                         order_by: str = 'observed_on',
                         order: str = 'asc',
                         observation_fields: Optional[dict] = 'default',
                         token: Optional[str] = None,
                         session: Optional[requests.Session] = None,
                         use_cache: bool = True) -> pd.DataFrame:
    """Fetch observation-level rows for exploratory notebook analyses.

    Parameters mirror a subset of iNaturalist observation filters and return a
    stable dataframe focused on date, taxon, identification timing, and image
    fields that are useful for seasonal prevalence and identification-delay
    analyses.

    ``observation_fields`` controls which fields the API returns per
    observation.  The default (``'default'``) requests only the fields the
    normalizer uses, reducing payload 50-80%.  Pass ``None`` for the full
    payload or a custom dict matching the iNaturalist ``fields`` format.
    """
    per_page_value = min(200, max(1, int(per_page)))
    max_pages_value = max(1, int(max_pages))
    query = _build_observation_query(
        kind=kind,
        places=places,
        loc=loc,
        project_id=project_id,
        taxa_filters=taxa_filters,
        d1=d1,
        d2=d2,
        quality_grade=quality_grade,
        verifiable=verifiable,
        order_by=order_by,
        order=order,
    )
    if session is None:
        session = get_inat_session(token=token, use_cache=use_cache)

    resolved_fields = OBSERVATION_FIELDS_MINIMAL if observation_fields == 'default' else observation_fields
    fields_json = json.dumps(resolved_fields) if resolved_fields is not None else None

    obs_url = 'https://api.inaturalist.org/v1/observations'

    # Preflight: per_page=0 count check to size pagination and warn on truncation
    preflight_params = _rest_params_from_query({**query, 'per_page': 0})
    try:
        preflight_resp = session.get(obs_url, params=preflight_params, timeout=30)
        preflight_resp.raise_for_status()
        total_results = preflight_resp.json().get('total_results', None)
    except Exception:
        total_results = None

    if total_results is not None:
        import math
        pages_needed = math.ceil(total_results / per_page_value) if total_results > 0 else 0
        actual_pages = min(max_pages_value, pages_needed)
        if pages_needed > max_pages_value:
            fetchable = max_pages_value * per_page_value
            logger.warning(
                'Truncation: %d observations in scope but only %d fetchable '
                '(%d pages x %d/page). Increase max_pages to %d to capture all.',
                total_results, fetchable, max_pages_value, per_page_value, pages_needed,
            )
    else:
        actual_pages = max_pages_value

    if total_results == 0:
        return pd.DataFrame(columns=OBSERVATION_ROW_COLUMNS)

    rows = []
    for page in range(1, actual_pages + 1):
        # Always use REST API for paginated queries since pyinaturalist's get_observations
        # doesn't support manual pagination with page parameter.
        payload = {}
        params = _rest_params_from_query({**query, 'page': page, 'per_page': per_page_value})
        if fields_json is not None:
            params['fields'] = fields_json
        response = session.get(obs_url, params=params, timeout=30)
        response.raise_for_status()
        payload = response.json()

        results = payload.get('results', []) if isinstance(payload, dict) else []
        if not results:
            break
        rows.extend(_normalize_observation_record(observation, kind, project_id) for observation in results if isinstance(observation, dict))
        if len(results) < per_page_value:
            break

    frame = pd.DataFrame(rows, columns=OBSERVATION_ROW_COLUMNS)
    if frame.empty:
        return frame

    for date_col in ['observed_at', 'created_at', 'first_identification_at', 'first_non_owner_identification_at']:
        frame[date_col] = pd.to_datetime(frame[date_col], utc=True, errors='coerce')
    frame['observed_on'] = pd.to_datetime(frame['observed_on'], errors='coerce').dt.strftime('%Y-%m-%d')
    frame['taxon_id'] = pd.to_numeric(frame['taxon_id'], errors='coerce').astype('Int64')
    frame['observer_id'] = pd.to_numeric(frame['observer_id'], errors='coerce').astype('Int64')
    frame['identifications_count'] = pd.to_numeric(frame['identifications_count'], errors='coerce').fillna(0).astype(int)
    frame['all_identification_count'] = pd.to_numeric(frame['all_identification_count'], errors='coerce').fillna(0).astype(int)
    frame['non_owner_identification_count'] = pd.to_numeric(frame['non_owner_identification_count'], errors='coerce').fillna(0).astype(int)
    frame['num_identification_agreements'] = pd.to_numeric(frame['num_identification_agreements'], errors='coerce')
    frame['num_identification_disagreements'] = pd.to_numeric(frame['num_identification_disagreements'], errors='coerce')
    frame['latitude'] = pd.to_numeric(frame['latitude'], errors='coerce')
    frame['longitude'] = pd.to_numeric(frame['longitude'], errors='coerce')
    frame['positional_accuracy'] = pd.to_numeric(frame['positional_accuracy'], errors='coerce')
    frame['first_identification_delay_days'] = pd.to_numeric(frame['first_identification_delay_days'], errors='coerce')
    frame['first_non_owner_identification_delay_days'] = pd.to_numeric(frame['first_non_owner_identification_delay_days'], errors='coerce')
    return frame


def annotate_taxon_nativity(observations: pd.DataFrame,
                            taxon_id_col: str = 'taxon_id',
                            nativity_place_id: Optional[int] = DEFAULT_NATIVITY_PLACE_ID,
                            token: Optional[str] = None,
                            session: Optional[requests.Session] = None,
                            use_cache: bool = True) -> pd.DataFrame:
    """Add a nativity label for each taxon represented in ``observations``."""
    if taxon_id_col not in observations.columns:
        raise KeyError(f"Missing taxon id column: {taxon_id_col}")

    out = observations.copy()
    if out.empty:
        out['nativity'] = pd.Series(dtype='object')
        return out

    if session is None:
        session = get_inat_session(token=token, use_cache=use_cache)

    unique_ids = [
        int(v) for v in out[taxon_id_col].dropna().unique()
    ]
    nativity_cache = _batch_lookup_nativity(
        session=session,
        taxon_ids=unique_ids,
        nativity_place_id=nativity_place_id,
    )
    labels = []
    for value in out[taxon_id_col]:
        if pd.isna(value):
            labels.append('Unknown')
        else:
            labels.append(nativity_cache.get(int(value), 'Unknown'))
    out['nativity'] = labels
    return out


def summarize_time_series(frame: pd.DataFrame,
                          date_col: str,
                          freq: str = 'MS',
                          group_cols: Optional[list[str]] = None,
                          count_col: Optional[str] = 'obs_id',
                          count_name: str = 'count',
                          value_aggs: Optional[dict[str, tuple[str, str]]] = None) -> pd.DataFrame:
    """Aggregate observation-like rows into a period-based summary table."""
    if date_col not in frame.columns:
        raise KeyError(f"Missing date column: {date_col}")

    working = frame.copy()
    working[date_col] = pd.to_datetime(working[date_col], utc=True, errors='coerce')
    working = working[working[date_col].notna()].copy()

    group_cols = list(group_cols or [])
    value_aggs = value_aggs or {}
    output_columns = ['period_start', *group_cols, count_name, *value_aggs.keys()]
    if working.empty:
        return pd.DataFrame(columns=output_columns)

    working['period_start'] = working[date_col].dt.to_period(freq).dt.to_timestamp()
    named_aggs: dict[str, tuple[str, str]] = {}
    if count_col and count_col in working.columns:
        named_aggs[count_name] = (count_col, 'nunique')
    else:
        named_aggs[count_name] = (date_col, 'size')

    for output_name, (source_col, agg_func) in value_aggs.items():
        if source_col not in working.columns:
            raise KeyError(f"Missing value column: {source_col}")
        named_aggs[output_name] = (source_col, agg_func)

    summary = working.groupby(['period_start', *group_cols], dropna=False).agg(**named_aggs).reset_index()
    summary.sort_values(['period_start', *group_cols], inplace=True)
    return summary[output_columns]


def coming_soon(kind: str = 'any',
                places: list[int] = None,
                loc: tuple[float, float, float] = None,
                norm: str = None,
                limit: int = 10,
                token: Optional[str] = None,
                session: Optional[requests.Session] = None,
                per_page: Optional[int] = None,
                max_pages: Optional[int] = None,
                fetch_images: bool = False,
                use_cache: bool = True,
                lineage_filter: str = 'any',
                nativity_place_id: Union[int, str, None] = 'auto',
                ) -> pd.DataFrame:
    """Return nearby seasonal/common taxa, optionally normalized.

    Parameters
    ----------
    kind:
        One of the supported kinds; e.g. ``'any'``, ``'plants'``, ``'birds'``.
        Defaults to ``'any'`` when omitted.
    places:
        Optional list of iNaturalist place IDs to scope the query.
    loc:
        Optional (lat, lng, radius) tuple to scope the query.
    norm:
        If provided, normalize counts by ``'time'``, ``'place'`` or
        ``'overall'`` frequency to produce a relative ranking.
    limit:
        Maximum number of species rows to return. Values above 25 are capped.
    per_page, max_pages:
        Optional pagination overrides for API calls. Leave as ``None`` to use
        API defaults.
    lineage_filter:
        Optional nativity filter. Supported values are ``'any'`` (default),
        ``'native_endemic'``, and ``'introduced'``.
    nativity_place_id:
        Optional place scope used when inferring nativity. Use ``'auto'``
        (default) to derive region from query context, an integer place ID
        (e.g. ``1297`` for Virginia), or ``None`` for global nativity checks.

    Returns
    -------
    pandas.DataFrame
        A DataFrame with columns including ``count``, ``taxon.name``, and
        ``taxon.preferred_common_name``. If no results are found an empty
        DataFrame with the expected columns is returned.
    """
    logger.debug('coming_soon called kind=%s places=%s loc=%s norm=%s', kind, places, loc, norm)
    assert not (places and loc), "only one of places and loc should be provided"
    
    if not places and not loc:
        logger.info('no place or location specified, assuming loc=(37.6669, -77.8883, 25)')
        loc = (37.6669, -77.8883, 25)

    assert norm in [None, 'time', 'place', 'overall'], "norm must be one of None, 'time', 'place', or 'overall'"
    assert lineage_filter in ['any', 'native_endemic', 'introduced'], "lineage_filter must be one of 'any', 'native_endemic', or 'introduced'"

    taxa = _taxa_for_kind(kind)
    normalized_kind = kind.lower().strip()
    if normalized_kind == 'fruit':
        normalized_kind = 'fruits'

    if places:
        place = {'place_id':places}
    elif isinstance(loc, tuple) and (len(loc) == 3):
        place = {'lat':loc[0], 
                 'lng':loc[1], 
                 'radius':loc[2]}
    else:
        raise ValueError(f"expected loc triple of lat,long,radius")

    requested_limit = max(1, int(limit))
    capped_limit = min(requested_limit, 25)
    if requested_limit > 25:
        logger.info('Requested limit=%s exceeds cap; using 25', requested_limit)

    per_page_value = max(1, int(per_page)) if per_page is not None else None
    max_pages_value = max(1, int(max_pages)) if max_pages is not None else None

    time = []
    strt = dt.date.today()+dt.timedelta(days=-6)
    fnsh = dt.date.today()+dt.timedelta(days=7)
    dates = pd.date_range(start=strt, end=fnsh, freq='D')
    for month in dates.month.unique():
        time.append({'month':month, 'day':list(dates[dates.month==month].day)})

    COLS = ['taxon.id', 'taxon.name', 'taxon.preferred_common_name', 'taxon.wikipedia_url', 'taxon.default_photo.medium_url', 'count']

    # Prepare session if needed (fallback path)
    if session is None:
        session = get_inat_session(token=token, use_cache=use_cache)

    resolved_nativity_place_id = _resolve_nativity_place_id(
        session=session,
        nativity_place_id=nativity_place_id,
        places=places,
        loc=loc,
    )

    results = []
    for t in time:
        if HAS_PYINAT:
            species_count_kwargs = {'verifiable': True, **taxa, **t, **place}
            if per_page_value is not None:
                species_count_kwargs['per_page'] = per_page_value
            resp = inat.get_observation_species_counts(**species_count_kwargs)
            frames = pd.json_normalize(resp.get('results', []))
        else:
            # fallback to direct REST API; use API defaults unless paging overrides are provided
            url = "https://api.inaturalist.org/v1/observations/species_counts"
            if max_pages_value is None:
                params = {**taxa, **t, **place, 'verifiable': 'true'}
                if per_page_value is not None:
                    params['per_page'] = per_page_value
                try:
                    r = session.get(url, params=params, timeout=30)
                    r.raise_for_status()
                    frames = pd.json_normalize(r.json().get('results', []))
                except Exception as e:
                    logger.warning('Failed REST call for observation_species_counts: %s', e)
                    frames = pd.DataFrame()
            else:
                chunk_frames = []
                for page in range(1, max_pages_value + 1):
                    params = {**taxa, **t, **place, 'verifiable': 'true', 'page': page}
                    if per_page_value is not None:
                        params['per_page'] = per_page_value
                    try:
                        r = session.get(url, params=params, timeout=30)
                        r.raise_for_status()
                        page_df = pd.json_normalize(r.json().get('results', []))
                        if page_df.empty:
                            break
                        chunk_frames.append(page_df)
                        if per_page_value is not None and len(page_df) < per_page_value:
                            break
                    except Exception as e:
                        logger.warning('Failed REST call for observation_species_counts: %s', e)
                        break
                frames = pd.concat(chunk_frames, ignore_index=True) if chunk_frames else pd.DataFrame()
        results.append(frames)
    results = pd.concat(results)[COLS]
    results = results.groupby(['taxon.id', 'taxon.name', 'taxon.preferred_common_name', 'taxon.wikipedia_url', 'taxon.default_photo.medium_url']).sum().reset_index()

    # ensure numeric counts and handle missing taxon ids
    results['count'] = pd.to_numeric(results.get('count', pd.Series(dtype=float)), errors='coerce').fillna(0)

    if norm:
        # prepare taxon id list and helper for chunked API calls
        taxon_ids = results['taxon.id'].dropna().astype(int).tolist()

        def _chunked_get_counts(taxon_ids_list, chunk_size=100, extra_kwargs=None):
            frames = []
            extra_kwargs = extra_kwargs or {}
            for i in range(0, len(taxon_ids_list), chunk_size):
                chunk = taxon_ids_list[i:i + chunk_size]
                if HAS_PYINAT:
                    count_kwargs = {'taxon_id': chunk, 'verifiable': True, **extra_kwargs}
                    if per_page_value is not None:
                        count_kwargs['per_page'] = per_page_value
                    resp = inat.get_observation_species_counts(**count_kwargs)
                    df = pd.json_normalize(resp.get('results', []))
                else:
                    url = "https://api.inaturalist.org/v1/observations/species_counts"
                    if max_pages_value is None:
                        params = {**extra_kwargs, 'taxon_id': chunk, 'verifiable': 'true'}
                        if per_page_value is not None:
                            params['per_page'] = per_page_value
                        try:
                            r = session.get(url, params=params, timeout=30)
                            r.raise_for_status()
                            df = pd.json_normalize(r.json().get('results', []))
                        except Exception as e:
                            logger.warning('Failed REST call for chunked observation_species_counts: %s', e)
                            df = pd.DataFrame()
                    else:
                        page_frames = []
                        for page in range(1, max_pages_value + 1):
                            params = {**extra_kwargs, 'taxon_id': chunk, 'verifiable': 'true', 'page': page}
                            if per_page_value is not None:
                                params['per_page'] = per_page_value
                            try:
                                r = session.get(url, params=params, timeout=30)
                                r.raise_for_status()
                                page_df = pd.json_normalize(r.json().get('results', []))
                                if page_df.empty:
                                    break
                                page_frames.append(page_df)
                                if per_page_value is not None and len(page_df) < per_page_value:
                                    break
                            except Exception as e:
                                logger.warning('Failed REST call for chunked observation_species_counts: %s', e)
                                break
                        df = pd.concat(page_frames, ignore_index=True) if page_frames else pd.DataFrame()
                if not df.empty:
                    frames.append(df)
            return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()

        if not taxon_ids:
            results['normalizer'] = np.nan
            results['sorter'] = 0
        else:
            extra = {key: value for key, value in taxa.items() if key in {'term_id', 'term_value_id'}}
            if norm == 'place':
                frames = []
                for t in time:
                    df = _chunked_get_counts(taxon_ids, extra_kwargs={**extra, **t})
                    if not df.empty:
                        frames.append(df)
                normer_df = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
            else:
                extra_kwargs = place if norm == 'time' else {}
                normer_df = _chunked_get_counts(taxon_ids, extra_kwargs=extra_kwargs)

            if not normer_df.empty and 'taxon.id' in normer_df.columns:
                normer_series = normer_df.groupby('taxon.id', as_index=True)['count'].sum()
            else:
                normer_series = pd.Series(dtype=float)

            results['normalizer'] = results['taxon.id'].map(normer_series).astype(float)
            results['normalizer'] = results['normalizer'].replace(0, np.nan)
            results['sorter'] = results['count'].astype(float).div(results['normalizer']).fillna(0)
            results.sort_values('sorter', ascending=False, inplace=True)

    if normalized_kind in PHENOTYPE_IMAGE_KINDS and not results.empty:
        for idx in results.head(capped_limit).index:
            taxon_id = results.at[idx, 'taxon.id']
            if pd.isna(taxon_id):
                continue
            fallback_url = results.at[idx, 'taxon.default_photo.medium_url']
            results.at[idx, 'taxon.default_photo.medium_url'] = _get_filtered_photo_url(
                session=session,
                taxon_id=int(taxon_id),
                taxa_filters=taxa,
                place_filters=place,
                time_filters=time,
                fallback_url=fallback_url,
            )

    nativity_cache = {}
    if lineage_filter != 'any' and not results.empty:
        for idx in results.index:
            taxon_id = results.at[idx, 'taxon.id']
            if pd.isna(taxon_id):
                results.at[idx, 'nativity'] = 'Unknown'
                continue
            cache_key = int(taxon_id)
            if cache_key not in nativity_cache:
                nativity_cache[cache_key] = _lookup_nativity_via_species_counts(
                    session=session,
                    taxon_id=cache_key,
                    nativity_place_id=resolved_nativity_place_id,
                )
            results.at[idx, 'nativity'] = nativity_cache[cache_key]

        if lineage_filter == 'introduced':
            results = results[results['nativity'] == 'Introduced']
        else:
            # Keep native, endemic, AND unknown (taxa without establishment
            # records shouldn't be silently dropped — only confirmed
            # introduced taxa are excluded).
            results = results[results['nativity'] != 'Introduced']

    # Display species names and their main images
    for index, row in results.head(capped_limit).iterrows():
        taxon_name = row['taxon.name']
        common_name = row.get('taxon.preferred_common_name', 'N/A')
        wiki_url = row.get('taxon.wikipedia_url')
        if pd.isna(common_name) or not str(common_name).strip():
            common_name = 'N/A'
        if pd.isna(taxon_name) or not str(taxon_name).strip():
            taxon_name = 'Unknown'
        image_url = row['taxon.default_photo.medium_url']
        nativity = _infer_nativity_from_row(row)
        taxon_id = row.get('taxon.id')
        if not nativity:
            cache_key = int(taxon_id) if pd.notna(taxon_id) else None
            if cache_key is None:
                nativity = 'Unknown'
            elif 'nativity' in results.columns and pd.notna(row.get('nativity')):
                nativity = row.get('nativity')
            elif cache_key in nativity_cache:
                nativity = nativity_cache[cache_key]
            else:
                nativity = _lookup_nativity_via_species_counts(
                    session=session,
                    taxon_id=cache_key,
                    nativity_place_id=resolved_nativity_place_id,
                )
                nativity_cache[cache_key] = nativity

        logger.info("%s (%s) - %s", taxon_name, common_name, row.get('taxon.wikipedia_url'))

        if fetch_images:
            try:
                if not image_url:
                    raise ValueError('No image URL')
                response = requests.get(image_url, timeout=10)
                response.raise_for_status()
                img = mpimg.imread(BytesIO(response.content), format='jpg')
                if HAS_IPYTHON_DISPLAY and isinstance(wiki_url, str) and wiki_url.strip():
                    label = f"{common_name} — {taxon_name}"
                    display(HTML(f'<div style="margin: 0.2em 0 0.4em 0;"><a href="{wiki_url}" target="_blank" rel="noopener noreferrer">{label}</a></div>'))
                fig, ax = plt.subplots()
                ax.imshow(img)
                ax.set_title(_format_photo_title(common_name=str(common_name), scientific_name=str(taxon_name), nativity=nativity))
                ax.set_xticks([])
                ax.set_yticks([])
                plt.show()
            except requests.exceptions.RequestException as e:
                logger.warning('Failed to load image %s: %s', image_url, e)
            except Exception as e:
                logger.debug('Skipping image display for %s: %s', image_url, e)
        ### It'd be nice to specifically select images w/ appropriate phenotype
            
    return results.head(capped_limit).copy()


def get_park_data(geocenter:tuple, kind:str, limit:int, token: Optional[str] = None, session: Optional[requests.Session] = None, use_cache: bool = True, per_page: int = 50, max_pages: int = 5) -> pd.DataFrame:
    """Return the most common species in a park, sorted by relative frequency.

    Parameters
    ----------
    geocenter:
        A (lat, lng, radius) triple defining the park center and search radius.
    kind:
        One of the supported kind strings (see :func:`coming_soon`).
    limit:
        Maximum number of rows to return.

    Returns
    -------
    pandas.DataFrame
        DataFrame of top taxa with columns ``count``, ``taxon.name``, and
        ``taxon.preferred_common_name``. An empty DataFrame is returned when
        no species are found.
    """
    logger.debug('get_park_data called geocenter=%s kind=%s limit=%s', geocenter, kind, limit)

    taxa = _taxa_for_kind(kind)
    
    if session is None:
        session = get_inat_session(token=token, use_cache=use_cache)

    per_page = max(1, int(per_page))
    max_pages = max(1, int(max_pages))

    if HAS_PYINAT:
        res = pd.json_normalize(inat.get_observation_species_counts(lat=geocenter[0], 
                                                                    lng=geocenter[1], 
                                                                    radius=geocenter[2], 
                                                                    **taxa,
                                                                    verifiable=True,
                                                                    per_page=per_page,)['results'])
    else:
        url = "https://api.inaturalist.org/v1/observations/species_counts"
        frames = []
        for page in range(1, max_pages + 1):
            params = {**taxa, 'lat': geocenter[0], 'lng': geocenter[1], 'radius': geocenter[2], 'verifiable': 'true', 'per_page': per_page, 'page': page}
            try:
                r = session.get(url, params=params, timeout=30)
                r.raise_for_status()
                page_df = pd.json_normalize(r.json().get('results', []))
                if page_df.empty:
                    break
                frames.append(page_df)
                if len(page_df) < per_page:
                    break
            except Exception as e:
                logger.warning('Failed REST call for park observation_species_counts: %s', e)
                break
        res = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    if res.empty:
        return pd.DataFrame(columns=['count', 'taxon.name', 'taxon.preferred_common_name', 'taxon.wikipedia_url']).head(limit)
    res['count'] = pd.to_numeric(res.get('count', pd.Series(dtype=float)), errors='coerce').fillna(0)
    taxon_ids = res['taxon.id'].dropna().astype(int).tolist()

    def _chunked_get_counts(taxon_ids_list, chunk_size=100):
        frames = []
        for i in range(0, len(taxon_ids_list), chunk_size):
            chunk = taxon_ids_list[i:i + chunk_size]
            if HAS_PYINAT:
                r = inat.get_observation_species_counts(taxon_id=chunk, verifiable=True, per_page=per_page)
                df = pd.json_normalize(r.get('results', []))
            else:
                url = "https://api.inaturalist.org/v1/observations/species_counts"
                page_frames = []
                for page in range(1, max_pages + 1):
                    params = {'taxon_id': chunk, 'verifiable': 'true', 'per_page': per_page, 'page': page}
                    try:
                        rr = session.get(url, params=params, timeout=30)
                        rr.raise_for_status()
                        page_df = pd.json_normalize(rr.json().get('results', []))
                        if page_df.empty:
                            break
                        page_frames.append(page_df)
                        if len(page_df) < per_page:
                            break
                    except Exception as e:
                        logger.warning('Failed REST call for chunked park counts: %s', e)
                        break
                df = pd.concat(page_frames, ignore_index=True) if page_frames else pd.DataFrame()
            if not df.empty:
                frames.append(df)
        return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()

    if not taxon_ids:
        res['normalizer'] = np.nan
        res['sorter'] = 0
    else:
        normer_df = _chunked_get_counts(taxon_ids)
        if not normer_df.empty and 'taxon.id' in normer_df.columns:
            normer = normer_df.groupby('taxon.id', as_index=True)['count'].sum()
        else:
            normer = pd.Series(dtype=float)

        res['normalizer'] = res['taxon.id'].map(normer).astype(float)
        res['normalizer'] = res['normalizer'].replace(0, np.nan)
        res['sorter'] = res['count'].astype(float).div(res['normalizer']).fillna(0)
        res.sort_values('sorter',ascending=False,inplace=True)

    logger.info('%s:', kind)
    if len(res) > 499:
        logger.warning('Too many results; normalization may be incomplete')
    return res[['count', 'taxon.name', 'taxon.preferred_common_name', 'taxon.wikipedia_url']].head(limit)


def _compute_location_from_values(mode: str,
                                  place_id_value: Optional[int],
                                  lat_value: Optional[float],
                                  lon_value: Optional[float],
                                  radius_value: Optional[float]) -> dict[str, Any]:
    if mode == 'place':
        if not place_id_value:
            raise ValueError('Please provide a valid Place ID.')
        return {'places': [int(place_id_value)]}

    coords = (lat_value, lon_value, radius_value)
    if any(v is None for v in coords):
        raise ValueError('Latitude, longitude, and radius are required for coordinate mode.')
    return {'loc': (float(lat_value), float(lon_value), float(radius_value))}


def _nativity_value(mode: str, nativity_id_value: Optional[int]) -> Union[str, int, None]:
    if mode == 'auto':
        return 'auto'
    if mode == 'none':
        return None
    return int(nativity_id_value) if nativity_id_value is not None else 'auto'


def _run_coming_soon_query(kind_value: str,
                           location_kwargs: dict[str, Any],
                           norm_value: Optional[str],
                           limit_value: int,
                           fetch_images_value: bool,
                           use_cache_value: bool,
                           lineage_filter_value: str,
                           nativity_value: Union[str, int, None],
                           token: Optional[str] = None,
                           session: Optional[requests.Session] = None) -> pd.DataFrame:
    norm_arg = None if norm_value in [None, 'none'] else norm_value
    return coming_soon(
        kind=kind_value,
        **location_kwargs,
        norm=norm_arg,
        limit=int(limit_value),
        fetch_images=bool(fetch_images_value),
        use_cache=bool(use_cache_value),
        lineage_filter=lineage_filter_value,
        nativity_place_id=nativity_value,
        token=token,
        session=session,
    )


def _lookup_place_name(session: Optional[requests.Session], place_id: Optional[int]) -> Optional[str]:
    if place_id is None:
        return None
    try:
        pid = int(place_id)
    except (TypeError, ValueError):
        return None

    endpoint = f'https://api.inaturalist.org/v1/places/{pid}'
    session_to_use = session or requests.Session()
    try:
        response = session_to_use.get(endpoint, timeout=12)
        response.raise_for_status()
        results = response.json().get('results', [])
        if not results:
            return None
        place = results[0] if isinstance(results[0], dict) else {}
        place_name = place.get('display_name') or place.get('name')
        if isinstance(place_name, str) and place_name.strip():
            return place_name.strip()
    except Exception as e:
        logger.debug('Could not resolve place name for place_id=%s: %s', place_id, e)
    return None


def _search_places(
    session: Optional[requests.Session],
    query: str,
    per_page: int = 10,
) -> list[dict[str, Any]]:
    """Search iNaturalist places by name using the autocomplete endpoint.

    Returns a list of dicts with ``id`` and ``display_name`` keys, or an
    empty list if the query is blank or the request fails.
    """
    if not query or not query.strip():
        return []
    endpoint = 'https://api.inaturalist.org/v1/places/autocomplete'
    session_to_use = session or requests.Session()
    try:
        response = session_to_use.get(
            endpoint,
            params={'q': query.strip(), 'per_page': per_page},
            timeout=12,
        )
        response.raise_for_status()
        results = response.json().get('results', [])
        places = []
        for place in results:
            pid = place.get('id')
            name = place.get('display_name') or place.get('name') or ''
            if pid is not None and name:
                places.append({'id': int(pid), 'display_name': name.strip()})
        return places
    except Exception as e:
        logger.debug('Place autocomplete failed for query=%r: %s', query, e)
        return []


def coming_soon_notebook(defaults: Optional[dict[str, Any]] = None,
                         use_widgets: Optional[bool] = None,
                         token: Optional[str] = None,
                         session: Optional[requests.Session] = None) -> dict[str, Any]:
    """Render a compact notebook UI and run :func:`coming_soon` with sane defaults.

    This helper centralizes notebook orchestration so users can run one cell
    to launch controls and one click to execute the query.

    Parameters
    ----------
    defaults:
        Optional overrides for control defaults. Supported keys include:
        ``kind``, ``place_mode``, ``place_id``, ``lat``, ``lon``, ``radius_km``,
        ``norm``, ``lineage_filter``, ``nativity_place_mode``,
        ``nativity_place_id``, ``limit``, ``fetch_images``, ``use_cache``.
    use_widgets:
        Force widget mode on/off. If ``None``, auto-detects availability.
    token, session:
        Optional iNaturalist auth/session overrides passed through to query calls.

    Returns
    -------
    dict
        State dictionary with at least ``mode`` and ``res`` entries.
    """

    if not HAS_IPYTHON_DISPLAY:
        raise RuntimeError('IPython display is not available in this environment.')

    cfg = {
        'kind': 'flowers',
        'place_mode': 'place',
        'place_id': 160915,
        'lat': 37.66933,
        'lon': -77.81001,
        'radius_km': 42.0,
        'norm': 'time',
        'lineage_filter': 'native_endemic',
        'nativity_place_mode': 'auto',
        'nativity_place_id': 1297,
        'limit': 7,
        'fetch_images': True,
        'use_cache': True,
    }
    if defaults:
        cfg.update(defaults)

    env_token_present = bool(os.getenv('INAT_TOKEN') or os.getenv('INAT_API_KEY') or os.getenv('PYINAT_API_KEY') or os.getenv('INAT_KEY'))
    loaded_key = load_api_key()
    print(f'Env token set: {env_token_present}')
    print(f'Credential available (env/keyring/file): {bool(loaded_key)}')

    kind_options = [
        'any', 'plants', 'flowers', 'fruits', 'mushrooms',
        'animals', 'wugs', 'fish', 'herps', 'birds', 'mammals',
        'butterflies', 'caterpillars',
    ]
    norm_options = ['none', 'time', 'place', 'overall']
    lineage_options = ['any', 'native_endemic', 'introduced']

    state: dict[str, Any] = {'mode': None, 'res': None}

    if use_widgets is None:
        try:
            import ipywidgets as widgets
            widgets_available = True
        except Exception:
            widgets = None
            widgets_available = False
    elif use_widgets:
        import ipywidgets as widgets
        widgets_available = True
    else:
        widgets = None
        widgets_available = False

    if widgets_available:
        place_name_cache: dict[int, Optional[str]] = {}

        def _cached_place_name(place_id: Optional[int]) -> Optional[str]:
            if place_id is None:
                return None
            try:
                pid = int(place_id)
            except (TypeError, ValueError):
                return None
            if pid not in place_name_cache:
                place_name_cache[pid] = _lookup_place_name(session=session, place_id=pid)
            return place_name_cache[pid]

        kind_w = widgets.Dropdown(options=kind_options, value=cfg['kind'], description='Kind:')
        place_mode_w = widgets.Dropdown(
            options=[('Use Place ID', 'place'), ('Use Coordinates', 'coords')],
            value=cfg['place_mode'],
            description='Location:',
        )
        place_id_w = widgets.IntText(value=int(cfg['place_id']), description='Place ID:')
        place_id_name_w = widgets.HTML(value='')
        lat_w = widgets.FloatText(value=float(cfg['lat']), description='Lat:')
        lon_w = widgets.FloatText(value=float(cfg['lon']), description='Lon:')
        radius_w = widgets.FloatText(value=float(cfg['radius_km']), description='Radius km:')
        limit_default = max(1, min(int(cfg['limit']), 25))
        limit_w = widgets.BoundedIntText(value=limit_default, min=1, max=25, description='Limit:')
        fetch_images_w = widgets.Checkbox(value=bool(cfg['fetch_images']), description='Fetch images')

        norm_w = widgets.Dropdown(options=norm_options, value=cfg['norm'], description='Norm:')
        lineage_filter_w = widgets.Dropdown(options=lineage_options, value=cfg['lineage_filter'], description='Lineage:')
        nativity_place_mode_w = widgets.Dropdown(
            options=[('Auto', 'auto'), ('Use Place ID', 'id'), ('None', 'none')],
            value=cfg['nativity_place_mode'],
            description='Reference Loc:',
        )
        nativity_place_id_w = widgets.IntText(value=int(cfg['nativity_place_id']), description='Nativity ID:')
        nativity_place_name_w = widgets.HTML(value='')
        use_cache_w = widgets.Checkbox(value=bool(cfg['use_cache']), description='Use cache')

        place_id_box = widgets.VBox([place_id_w, place_id_name_w])
        coords_box = widgets.VBox([lat_w, lon_w, radius_w])
        nativity_id_box = widgets.VBox([nativity_place_id_w, nativity_place_name_w])

        def _refresh_place_labels(*_):
            place_name = _cached_place_name(place_id_w.value)
            if place_name:
                place_id_name_w.value = f'<span style="opacity:0.8;">(Place: {place_name})</span>'
            else:
                place_id_name_w.value = ''

            nativity_name = _cached_place_name(nativity_place_id_w.value)
            if nativity_name:
                nativity_place_name_w.value = f'<span style="opacity:0.8;">(Reference: {nativity_name})</span>'
            else:
                nativity_place_name_w.value = ''

        def _update_location_visibility(*_):
            using_place = place_mode_w.value == 'place'
            place_id_box.layout.display = '' if using_place else 'none'
            coords_box.layout.display = 'none' if using_place else ''

        def _update_nativity_visibility(*_):
            needs_nativity_id = nativity_place_mode_w.value == 'id'
            nativity_id_box.layout.display = '' if needs_nativity_id else 'none'

        place_id_w.observe(_refresh_place_labels, names='value')
        nativity_place_id_w.observe(_refresh_place_labels, names='value')
        place_mode_w.observe(_update_location_visibility, names='value')
        nativity_place_mode_w.observe(_update_nativity_visibility, names='value')
        _update_location_visibility()
        _update_nativity_visibility()
        _refresh_place_labels()

        options_row = widgets.VBox([fetch_images_w, use_cache_w])
        controls_box = widgets.VBox([
            kind_w,
            place_mode_w,
            place_id_box,
            coords_box,
            norm_w,
            lineage_filter_w,
            nativity_place_mode_w,
            nativity_id_box,
            limit_w,
            options_row,
        ])

        run_button = widgets.Button(description='Run Query', button_style='success')
        out = widgets.Output()

        def _on_run(_):
            out.clear_output()
            with out:
                try:
                    location_kwargs = _compute_location_from_values(
                        place_mode_w.value,
                        place_id_w.value,
                        lat_w.value,
                        lon_w.value,
                        radius_w.value,
                    )
                    nativity_value = _nativity_value(nativity_place_mode_w.value, nativity_place_id_w.value)
                    print('Running query...')
                    res = _run_coming_soon_query(
                        kind_w.value,
                        location_kwargs,
                        norm_w.value,
                        limit_w.value,
                        fetch_images_w.value,
                        use_cache_w.value,
                        lineage_filter_w.value,
                        nativity_value,
                        token=token,
                        session=session,
                    )
                    state['res'] = res
                    if getattr(res, 'empty', True):
                        if Markdown is not None:
                            display(Markdown('No results found. Try a larger radius or different group.'))
                        else:
                            print('No results found. Try a larger radius or different group.')
                        return
                    if Markdown is not None:
                        display(Markdown(f'### Found {len(res)} taxa'))
                    show_cols = ['count', 'taxon.name', 'taxon.preferred_common_name', 'taxon.wikipedia_url']
                    safe_cols = [c for c in show_cols if c in res.columns]
                    display(res[safe_cols].head(int(limit_w.value)))
                except Exception as exc:
                    if Markdown is not None:
                        display(Markdown(f'**Could not run query:** {exc}'))
                        display(Markdown('Tips: verify Place ID or coordinates and try a different group.'))
                    else:
                        print(f'Could not run query: {exc}')

        run_button.on_click(_on_run)
        state['mode'] = 'widgets'
        state['run_button'] = run_button
        state['output'] = out
        display(controls_box)
        if Markdown is not None:
            display(Markdown('Click **Run Query** to execute.'))
        display(run_button, out)
        return state

    if Markdown is not None:
        display(Markdown('`ipywidgets` is unavailable; running in fallback text mode.'))

    try:
        location_kwargs = _compute_location_from_values(
            str(cfg['place_mode']),
            int(cfg['place_id']) if cfg.get('place_id') is not None else None,
            float(cfg['lat']) if cfg.get('lat') is not None else None,
            float(cfg['lon']) if cfg.get('lon') is not None else None,
            float(cfg['radius_km']) if cfg.get('radius_km') is not None else None,
        )
        nativity_value = _nativity_value(str(cfg['nativity_place_mode']), cfg.get('nativity_place_id'))
        res = _run_coming_soon_query(
            str(cfg['kind']),
            location_kwargs,
            str(cfg['norm']),
            int(cfg['limit']),
            bool(cfg['fetch_images']),
            bool(cfg['use_cache']),
            str(cfg['lineage_filter']),
            nativity_value,
            token=token,
            session=session,
        )
        state['mode'] = 'fallback'
        state['res'] = res
        if getattr(res, 'empty', True):
            if Markdown is not None:
                display(Markdown('No results found. Try a larger radius or different group.'))
            else:
                print('No results found. Try a larger radius or different group.')
        else:
            show_cols = ['count', 'taxon.name', 'taxon.preferred_common_name', 'taxon.wikipedia_url']
            safe_cols = [c for c in show_cols if c in res.columns]
            display(res[safe_cols].head(int(cfg['limit'])))
    except Exception as exc:
        state['mode'] = 'fallback'
        state['res'] = None
        if Markdown is not None:
            display(Markdown(f'**Could not run query:** {exc}'))
        else:
            print(f'Could not run query: {exc}')

    return state


