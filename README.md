# pynat

Small iNaturalist helper workflows for exploration and reliability analysis.

This repository currently has two main tracks:
- Observation discovery utilities in `helpers.py` (for example, coming-soon seasonal species queries).
- Identification reliability analysis in `reliability.py` (proposal-level outcomes and taxon-level summaries).

## Quickstart (local development)

Run these from the repository root:

```bash
uv venv
uv sync --extra dev
uv run --extra dev pytest -q
```

Notes:
- The project uses `uv` for environment and dependency management.
- `pytest` lives in the `dev` extra, so include `--extra dev` for test commands.

## Public notebook entrypoint

Try out `https://coming-soon-near-you.streamlit.app/`. I'm very happy with the interface here.

Exploratory notebooks are under `notebooks/exploratory/` and are not treated as stable public entrypoints.

## Authentication and secrets

Never commit API tokens or key files.

Runtime key lookup in `helpers.load_api_key()` uses this order:
1. Environment variables (`INAT_TOKEN`, `INAT_API_KEY`, `PYINAT_API_KEY`, `INAT_KEY`)
2. System keyring (if installed)
3. Legacy dill fallback file (`pyinaturalistkey.pkd`)

Session-only token example:

```python
import os, getpass
token = getpass.getpass("Paste iNaturalist token (optional): ").strip()
if token:
    os.environ["INAT_TOKEN"] = token
```

## Core helper workflows

Implemented in `helpers.py`:
- `coming_soon(...)`: seasonal/common taxa with optional normalization and nativity filtering.
- `get_park_data(...)`: park-centered species ranking by relative frequency.
- `get_observation_rows(...)`: project/place/date-scoped observation rows for exploratory analyses.
- `annotate_taxon_nativity(...)`: add Virginia-or-place nativity labels to observation/taxon tables.
- `summarize_time_series(...)`: period-based counts and summary metrics for observation dataframes.
- `coming_soon_notebook(...)`: notebook UI wrapper used by the public notebook.
- `get_mine(...)`: fetch and print recent observations for a user.

Quick example:

```python
from helpers import coming_soon, get_park_data

df = coming_soon(
    "birds",
    loc=(37.6669, -77.8883, 25),
    per_page=25,
    max_pages=2,
    fetch_images=False,
)

parks = get_park_data(
    (37.6669, -77.8883, 5),
    "plants",
    limit=20,
    per_page=25,
    max_pages=2,
)

obs = get_observation_rows(
    project_id="virginia-physiographic-regions-piedmont",
    d1="2024-01-01",
    per_page=100,
    max_pages=5,
)

monthly_obs = summarize_time_series(obs, date_col="observed_on", freq="M")
```

Fallback behavior:
- When `pyinaturalist` is unavailable, helper queries fall back to direct REST calls.
- Fallback calls are bounded by `per_page` and `max_pages` to avoid unbounded network fetches.

## API Version Trials (v1/v2)

The default API version is `v1`. You can trial `v2` in two ways:

1. Environment default for the process:

```bash
set INAT_API_VERSION=v2
uv run --extra dev pytest -q
```

2. Per-call override in helper functions:

```python
from helpers import coming_soon, get_park_data, get_observation_rows

df = coming_soon("birds", loc=(37.66, -77.88, 25), api_version="v2")
parks = get_park_data((37.66, -77.88, 5), "plants", limit=20, api_version="v2")
obs = get_observation_rows(project_id="some-project", per_page=50, max_pages=2, api_version="v2")
```

Notes:
- Keep tests mocked for API behavior; avoid relying on live API responses.
- Roll out v2 incrementally and compare output schema where notebook/dataframe stability matters.

Exploratory notebook prototypes:
- `notebooks/exploratory/va_piedmont_native_phenology.ipynb`: native Lepidoptera prevalence and life-stage exploration for the Virginia Piedmont project.
- `notebooks/exploratory/va_piedmont_identification_timing.ipynb`: observation volume, first non-owner identification volume, and delay summaries over time.

## Reliability analysis (`reliability.py`)

`reliability.py` contains an `Analyzer` class for taxon-scoped reliability analysis.

Primary workflow:
- Use `Analyzer.assess_taxon(...)` (or CLI `assess-taxon`) for one user and one target taxon.
- The method builds proposal outcomes and taxonomic overlap summaries in one pass.

Deprecated workflow:
- Legacy cache-first `ingest`/`summarize` commands are deprecated and kept only for compatibility.
- New usage should prefer taxon-scoped assessment.

Minimal programmatic example:

```python
from reliability import Analyzer

an = Analyzer(cache_dir="data")
out = an.assess_taxon(
    user_login="your_login",
    taxon_id=130953,
    start="2024-01-01",
    end="2025-12-31",
    print_report=False,
)

out["taxon_reliability"].head()
```

## Testing

Run the test suite with:

```bash
uv run --extra dev pytest -q
```

Current tests cover helper utilities and core reliability behavior paths.
