# work in progress

## environment setup

Run these inside this repository folder.

- `uv venv`
- `uv sync`
- `uv run pytest tests`

## binder/public usage

Public binder notebook scope (current): only coming_soon_near_you.ipynb is
treated as the public entrypoint.

Exploratory notebooks are kept under notebooks/exploratory/:
- notebooks/exploratory/inaturalismus.ipynb
- notebooks/exploratory/parks.ipynb

- Binder (JupyterLab):
	https://mybinder.org/v2/gh/svshepherd/pynat/main?urlpath=lab/tree/coming_soon_near_you.ipynb
- Binder (fallback, classic open):
	https://mybinder.org/v2/gh/svshepherd/pynat/main?filepath=coming_soon_near_you.ipynb
- Voila app view (code hidden):
	https://mybinder.org/v2/gh/svshepherd/pynat/main?urlpath=voila/render/coming_soon_near_you.ipynb

Tips:
- For repeatable public demos, replace `main` in the URLs with a release tag (example: `v0.1.0`).
- Binder cold starts can take a few minutes.
- `coming_soon()` and `get_park_data()` support a REST fallback when `pyinaturalist` is unavailable.

### first run in binder

1. Open the Binder JupyterLab link above.
2. Run all cells in `coming_soon_near_you.ipynb`.
3. Use the basic controls first (kind, location, norm, lineage).
4. Keep advanced settings small for faster public sessions (`per_page=25`, `max_pages=1-2`, `fetch_images=False`).

Optional session-only token (never commit secrets):

```python
import os, getpass
token = getpass.getpass('Paste iNaturalist token (optional): ').strip()
if token:
    os.environ['INAT_TOKEN'] = token
```

Quick low-traffic examples:

```python
from pynat.helpers import coming_soon, get_park_data

coming_soon('birds', loc=(37.6669, -77.8883, 25), per_page=25, max_pages=2, fetch_images=False)
get_park_data((37.6669, -77.8883, 5), 'plants', limit=20, per_page=25, max_pages=2)
```

## tools for checking local observations

"coming soon" by iconic taxon

## tools for reviewing IDs

standardizes file names for use with other services e.g. flickr

## tools for comparing obs/ident trends

tbd

## confidence_manimal
tool to score ID reliability per user | observation

### architecture notes (scoped to pynat/reliability.py only)

These notes describe only the proposal-level reliability analysis implemented
in pynat/reliability.py. They do not describe architecture for the rest of this
repository.

Pipeline stages for pynat/reliability.py:

1. Ingest (online): download the minimum raw entities needed to reconstruct
	 identification timelines for one user.
2. Build proposals (offline): derive user proposal events from timeline data,
	 then compute confirmation and correctness outcomes with rank-aware rules.
3. Summarize (offline): aggregate per-species and per-rank reliability metrics.

Core concepts in pynat/reliability.py:

- A proposal is each time a user changes taxon on an observation.
- Confirmations are later IDs by other users that meet a boundary defined by
	proposal rank and disagreement flag.
- Correctness depth compares proposal taxon versus final community taxon using
	the ladder: species, genus, family, higher, wrong, or no_ct.

Status ladder in pynat/reliability.py:

- vindicated
- undecided_support
- overruled
- withdrawn
- unknown

Status assignment uses ordered precedence in the implementation and is
intentional for this file-specific reliability logic.

Ingest policy in pynat/reliability.py:

- Default ingest mode is incremental.
- Incremental mode still refreshes the user's full identification list, then
	only refetches observation timelines for observations that are new or whose
	observation `updated_at` changed since the last cache snapshot.
- First-run or incomplete-cache scenarios automatically fall back to full ingest.
- Full ingest remains available for forced cache rebuilds.

API etiquette in pynat/reliability.py:

- Requests are paced and retried with exponential backoff for transient errors.
- `Retry-After` headers are honored when provided by the API.
- This keeps steady refresh workflows resilient while reducing pressure on the API.
