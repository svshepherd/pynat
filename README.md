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

## Public notebook entrypoint (Binder)

Public Binder scope: `coming_soon_near_you.ipynb`.

Exploratory notebooks are under `notebooks/exploratory/` and are not treated as stable public entrypoints.

- Binder (JupyterLab):
	https://mybinder.org/v2/gh/svshepherd/pynat/main?urlpath=lab/tree/coming_soon_near_you.ipynb
- Binder (classic fallback):
	https://mybinder.org/v2/gh/svshepherd/pynat/main?filepath=coming_soon_near_you.ipynb
- Voila render (code hidden):
	https://mybinder.org/v2/gh/svshepherd/pynat/main?urlpath=voila/render/coming_soon_near_you.ipynb

Binder tips:
- Cold starts can take a few minutes.
- For reproducible demos, replace `main` in URLs with a release tag.
- Start with low-traffic settings (`per_page=25`, `max_pages=2`, `fetch_images=False`).

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
```

Fallback behavior:
- When `pyinaturalist` is unavailable, helper queries fall back to direct REST calls.
- Fallback calls are bounded by `per_page` and `max_pages` to avoid unbounded network fetches.

## Reliability analysis (`reliability.py`)

`reliability.py` contains an `Analyzer` class for proposal-level reliability workflows.

High-level stages:
1. Ingest (online): cache identification timelines and observation shells.
2. Build proposals (offline): derive proposal events and outcomes.
3. Summarize (offline): species/rank reliability metrics.

Ingest policy:
- Default mode is incremental.
- Incremental mode refreshes user IDs and only refetches changed/new observation timelines.
- Missing cache files trigger automatic fallback to full ingest.

Minimal programmatic example:

```python
from reliability import Analyzer

an = Analyzer()
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
