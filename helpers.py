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

## TODO: CoPilot suggests these:
# Secrets: Move completely off dill and use env/keyring; add a .env.example and README notes.
# Packaging: Add pyproject.toml (or setup.cfg) and support an editable install (pip install -e .) to make local dev and CI simpler.
# CI: Add GitHub Actions / GitLab CI to run pytest, lint, and type checks on pushes/PRs.
# Tests: Expand tests and coverage (edge cases, pagination, API error responses, rate limits). See test_helpers.py.
# API robustness: Add pagination handling, request session with retries/backoff, and optional caching for repeated queries.
# Logging & config: Provide a short example in README showing how to configure logging and where to place API keys.
# Type checks & linting: Add mypy and flake8/ruff configs and run them in CI.
# Image selection: Improve photo choice (prefer taxon.default_photo or filter by photo metadata like is_primary/phenotype).
# Docs & examples: Add usage examples and a tiny CLI/runner script demonstrating get_mine and coming_soon.
# Security audit: Consider replacing pickle/dill storage entirely (encrypted file or a secrets manager) and add a short security note in the changelog.


import requests
import pandas as pd
import numpy as np
import datetime as dt
from typing import Tuple, Union, Optional
from time import sleep
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from io import BytesIO
import pyinaturalist as inat
import dill
import pickle
import os
import ipyplot
import logging

logger = logging.getLogger(__name__)

def load_api_key(fallback_path: str = 'pyinaturalistkey.pkd') -> Union[str, None]:
    """Load an iNaturalist API key from multiple sources.

    Order of attempts:
    1. Environment variables: ``INAT_API_KEY``, ``PYINAT_API_KEY``, ``INAT_KEY``
    2. System keyring (optional ``keyring`` package)
    3. Fallback dill file at ``fallback_path`` (legacy behavior)

    Returns:
        The API key string when found, otherwise ``None``.

    Notes:
        This helper will not raise on missing keys; it returns ``None`` so
        callers can decide whether to proceed or surface an error.
    """
    # 1) environment
    for name in ('INAT_API_KEY', 'PYINAT_API_KEY', 'INAT_KEY'):
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
    try:
        with open(fallback_path, 'rb') as f:
            return dill.load(f)
    except (OSError, EOFError, pickle.UnpicklingError) as e:
        logger.debug('Could not load API key from %s: %s', fallback_path, e)
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


def coming_soon(kind: str,
                places: list[int] = None,
                loc: tuple[float, float, float] = None,
                norm: str = None,
                limit: int = 10,
                ) -> pd.DataFrame:
    """Return nearby seasonal/common taxa, optionally normalized.

    Parameters
    ----------
    kind:
        One of the supported kinds; e.g. ``'any'``, ``'plants'``, ``'birds'``.
    places:
        Optional list of iNaturalist place IDs to scope the query.
    loc:
        Optional (lat, lng, radius) tuple to scope the query.
    norm:
        If provided, normalize counts by ``'time'``, ``'place'`` or
        ``'overall'`` frequency to produce a relative ranking.
    limit:
        Maximum number of species rows to return.

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

    if kind == 'any':
        taxa = {}
    elif kind == 'plants':
        taxa = {'taxon_name':'plants'}
    elif kind == 'flowers':
        taxa = {'term_id':12, 'term_value_id':13}
    elif kind == 'fruits':
        taxa = {'term_id':12, 'term_value_id':14}
    elif kind == 'mushrooms':
        taxa = {'taxon_id':47170}
    elif kind == 'animals':
        taxa = {'taxon_id':1}
    elif kind == 'fish':
        taxa = {'taxon_id':47178}
    elif kind == 'mammals':
        taxa = {'taxon_id':40151}
    elif kind == 'birds':
        taxa = {'taxon_id':3}
    elif kind == 'herps':
        taxa = {'taxon_id':[26036, 20978]}
    elif kind == 'wugs':
        taxa = {'taxon_id':1, 'not_id': 355675}
    elif kind == 'butterflies':
        taxa = {'taxon_id': 47157, 'term_id':1, 'term_value_id':2}
    elif kind == 'caterpillars':
        taxa = {'taxon_id': 47157, 'term_id':1, 'term_value_id':6}
    else:
        raise ValueError(f"kind '{kind}' not implemented")

    if places:
        place = {'place_id':places}
    elif isinstance(loc, tuple) and (len(loc) == 3):
        place = {'lat':loc[0], 
                 'lng':loc[1], 
                 'radius':loc[2]}
    else:
        raise ValueError(f"expected loc triple of lat,long,radius")

    time = []
    strt = dt.date.today()+dt.timedelta(days=-6)
    fnsh = dt.date.today()+dt.timedelta(days=7)
    dates = pd.date_range(start=strt, end=fnsh, freq='D')
    for month in dates.month.unique():
        time.append({'month':month, 'day':list(dates[dates.month==month].day)})

    COLS = ['taxon.id', 'taxon.name', 'taxon.preferred_common_name', 'taxon.wikipedia_url', 'taxon.default_photo.medium_url', 'count']

    results = []
    for t in time:
        results.append(pd.json_normalize(inat.get_observation_species_counts(verifiable=True,
                                                                            **taxa,
                                                                            **t,
                                                                            **place,)['results']))
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
                resp = inat.get_observation_species_counts(taxon_id=chunk, verifiable=True, **extra_kwargs)
                df = pd.json_normalize(resp.get('results', []))
                if not df.empty:
                    frames.append(df)
            return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()

        if not taxon_ids:
            results['normalizer'] = np.nan
            results['sorter'] = 0
        else:
            extra = {key: value for key, value in taxa.items() if key == 'term_value_id'}
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
            results['normalizer'].replace(0, np.nan, inplace=True)
            results['sorter'] = results['count'].astype(float).div(results['normalizer']).fillna(0)
            results.sort_values('sorter', ascending=False, inplace=True)

    # Display species names and their main images
    for index, row in results.head(limit).iterrows():
        taxon_name = row['taxon.name']
        common_name = row.get('taxon.preferred_common_name', 'N/A')
        image_url = row['taxon.default_photo.medium_url']

        logger.info("%s (%s) - %s", taxon_name, common_name, row.get('taxon.wikipedia_url'))

        try:
            if not image_url:
                raise ValueError('No image URL')
            response = requests.get(image_url, timeout=10)
            response.raise_for_status()
            img = mpimg.imread(BytesIO(response.content), format='jpg')
            plt.imshow(img)
            plt.xticks([])  # Hide x tick labels
            plt.yticks([])  # Hide y tick labels
            plt.show()
        except requests.exceptions.RequestException as e:
            logger.warning('Failed to load image %s: %s', image_url, e)
        except Exception as e:
            logger.debug('Skipping image display for %s: %s', image_url, e)
        ### It'd be nice to specifically select images w/ appropriate phenotype
            
    return results


def get_park_data(geocenter:tuple, kind:str, limit:int) -> pd.DataFrame:
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

    if kind == 'any':
        taxa = {}
    elif kind == 'plants':
        taxa = {'taxon_name':'plants'}
    elif kind == 'mushrooms':
        taxa = {'taxon_id':47170}
    elif kind == 'animals':
        taxa = {'taxon_id':1}
    elif kind == 'wugs':
        taxa = {'taxon_id':1, 'not_id': 355675}
    elif kind == 'fish':
        taxa = {'taxon_id':47178}
    elif kind == 'herps':
        taxa = {'taxon_id':[26036, 20978]}
    elif kind == 'birds':
        taxa = {'taxon_id':3}
    elif kind == 'mammals':
        taxa = {'taxon_id':40151}
    else:
        raise ValueError(f"Unknown kind: {kind}")
    
    res = pd.json_normalize(inat.get_observation_species_counts(lat=geocenter[0], 
                                                                lng=geocenter[1], 
                                                                radius=geocenter[2], 
                                                                **taxa,
                                                                verifiable=True,)['results'])
    if res.empty:
        return pd.DataFrame(columns=['count', 'taxon.name', 'taxon.preferred_common_name', 'taxon.wikipedia_url']).head(limit)
    res['count'] = pd.to_numeric(res.get('count', pd.Series(dtype=float)), errors='coerce').fillna(0)
    taxon_ids = res['taxon.id'].dropna().astype(int).tolist()

    def _chunked_get_counts(taxon_ids_list, chunk_size=100):
        frames = []
        for i in range(0, len(taxon_ids_list), chunk_size):
            chunk = taxon_ids_list[i:i + chunk_size]
            r = inat.get_observation_species_counts(taxon_id=chunk, verifiable=True)
            df = pd.json_normalize(r.get('results', []))
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
        res['normalizer'].replace(0, np.nan, inplace=True)
        res['sorter'] = res['count'].astype(float).div(res['normalizer']).fillna(0)
        res.sort_values('sorter',ascending=False,inplace=True)

    logger.info('%s:', kind)
    if len(res) > 499:
        logger.warning('Too many results; normalization may be incomplete')
    return res[['count', 'taxon.name', 'taxon.preferred_common_name', 'taxon.wikipedia_url']].head(limit)
