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

import requests
import pandas as pd
import numpy as np
import datetime as dt
from typing import Tuple, Union, Optional, Any
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
import dill
import pickle
import os
import ipyplot
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
        return {'taxon_name': 'plants'}
    if normalized_kind == 'flowers':
        return {'term_id': 12, 'term_value_id': 13}
    if normalized_kind == 'fruits':
        return {'term_id': 12, 'term_value_id': 14}
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
        'per_page': 1,
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
                if payload.get('results'):
                    return label
            except Exception as e:
                logger.debug('Nativity lookup failed for taxon %s (%s): %s', taxon_id, key, e)
                continue
        return None

    result = _search_statuses(base)
    if result:
        return result

    return 'Unknown'


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

        try:
            ipyplot.plot_images(images)
        except Exception as e:
            logger.warning('Failed to display images for %s: %s', obs_id, e)


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


def coming_soon(kind: str = 'any',
                places: list[int] = None,
                loc: tuple[float, float, float] = None,
                norm: str = None,
                limit: int = 10,
                token: Optional[str] = None,
                session: Optional[requests.Session] = None,
                per_page: int = 50,
                max_pages: int = 5,
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
        Maximum number of species rows to return.
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

    per_page = max(1, int(per_page))
    max_pages = max(1, int(max_pages))

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
            resp = inat.get_observation_species_counts(verifiable=True, per_page=per_page, **taxa, **t, **place)
            frames = pd.json_normalize(resp.get('results', []))
        else:
            # fallback to direct REST API with bounded pagination
            url = "https://api.inaturalist.org/v1/observations/species_counts"
            chunk_frames = []
            for page in range(1, max_pages + 1):
                params = {**taxa, **t, **place, 'verifiable': 'true', 'per_page': per_page, 'page': page}
                try:
                    r = session.get(url, params=params, timeout=30)
                    r.raise_for_status()
                    page_df = pd.json_normalize(r.json().get('results', []))
                    if page_df.empty:
                        break
                    chunk_frames.append(page_df)
                    if len(page_df) < per_page:
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
                    resp = inat.get_observation_species_counts(taxon_id=chunk, verifiable=True, per_page=per_page, **extra_kwargs)
                    df = pd.json_normalize(resp.get('results', []))
                else:
                    url = "https://api.inaturalist.org/v1/observations/species_counts"
                    page_frames = []
                    for page in range(1, max_pages + 1):
                        params = {**extra_kwargs, 'taxon_id': chunk, 'verifiable': 'true', 'per_page': per_page, 'page': page}
                        try:
                            r = session.get(url, params=params, timeout=30)
                            r.raise_for_status()
                            page_df = pd.json_normalize(r.json().get('results', []))
                            if page_df.empty:
                                break
                            page_frames.append(page_df)
                            if len(page_df) < per_page:
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
        for idx in results.head(limit).index:
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
            results = results[results['nativity'].isin(['Native', 'Endemic'])]

    # Display species names and their main images
    for index, row in results.head(limit).iterrows():
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
            
    return results


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
                           per_page_value: int,
                           max_pages_value: int,
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
        per_page=int(per_page_value),
        max_pages=int(max_pages_value),
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
        ``nativity_place_id``, ``limit``, ``fetch_images``, ``per_page``,
        ``max_pages``, ``use_cache``.
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
        'per_page': 25,
        'max_pages': 2,
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
        limit_w = widgets.BoundedIntText(value=int(cfg['limit']), min=1, max=200, description='Limit:')
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
        per_page_w = widgets.BoundedIntText(value=int(cfg['per_page']), min=10, max=200, description='Per page:')
        max_pages_w = widgets.BoundedIntText(value=int(cfg['max_pages']), min=1, max=8, description='Pages:')
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

        basic_box = widgets.VBox([
            widgets.HTML('<h4>Start here (basic)</h4>'),
            kind_w,
            place_mode_w,
            place_id_box,
            coords_box,
            norm_w,
            lineage_filter_w,
            nativity_place_mode_w,
            fetch_images_w,
        ])
        advanced_box = widgets.VBox([
            widgets.HTML('<h4>Advanced (optional)</h4>'),
            limit_w,
            nativity_id_box,
            per_page_w,
            max_pages_w,
            use_cache_w,
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
                        per_page_w.value,
                        max_pages_w.value,
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
                        display(Markdown('Tips: verify Place ID or coordinates, then try fewer pages (1-2).'))
                    else:
                        print(f'Could not run query: {exc}')

        run_button.on_click(_on_run)
        state['mode'] = 'widgets'
        state['run_button'] = run_button
        state['output'] = out
        display(basic_box, advanced_box)
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
            int(cfg['per_page']),
            int(cfg['max_pages']),
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
