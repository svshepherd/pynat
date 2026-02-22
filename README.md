# work in progress

## binder/public usage

- View in Binder here: https://mybinder.org/v2/gh/svshepherd/pynat/HEAD
- Select "coming soon near you" notebook
- In Binder, either run anonymously or set a session-only token:

```python
import os, getpass
token = getpass.getpass('Paste iNaturalist token (optional): ').strip()
if token:
	os.environ['INAT_TOKEN'] = token
```

- `coming_soon()` and `get_park_data()` now support optional REST fallback when `pyinaturalist` is unavailable.
- For low traffic / faster runs in Binder, use smaller pages and fast mode:

```python
from pynat.helpers import coming_soon, get_park_data

coming_soon('birds', loc=(37.6669, -77.8883, 25), fast=True, per_page=25, max_pages=2, fetch_images=False)
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
