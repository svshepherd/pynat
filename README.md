# work in progress

## binder/public usage

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
