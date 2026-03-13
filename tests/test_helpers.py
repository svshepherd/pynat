import os
import datetime as dt

import pandas as pd
import pytest


def test_get_park_data_empty(monkeypatch):
    import helpers as h

    def fake_counts(**kwargs):
        return {"results": []}

    if not h.HAS_PYINAT:
        pytest.skip("pyinaturalist not installed; this test covers pyinaturalist path")
    monkeypatch.setattr(h.inat, "get_observation_species_counts", fake_counts)
    res = h.get_park_data((0, 0, 1), "any", limit=5)
    assert isinstance(res, pd.DataFrame)
    assert res.empty


def test_coming_soon_normalizer_safe(monkeypatch):
    import helpers as h

    def fake_counts(**kwargs):
        # When called for the monthly results include a single taxon
        if "month" in kwargs or "day" in kwargs:
            return {"results": [{
                "taxon.id": 1,
                "taxon.name": "Specimen",
                "taxon.preferred_common_name": "Spec",
                "taxon.wikipedia_url": None,
                "taxon.default_photo.medium_url": None,
                "count": 10,
            }]}
        # For normalizer calls return empty
        return {"results": []}

    if not h.HAS_PYINAT:
        pytest.skip("pyinaturalist not installed; this test covers pyinaturalist path")
    monkeypatch.setattr(h.inat, "get_observation_species_counts", fake_counts)
    res = h.coming_soon("any", loc=(0, 0, 1), norm="overall", limit=5)
    assert isinstance(res, pd.DataFrame)
    # ensure function ran and produced columns
    assert "count" in res.columns
    assert "sorter" in res.columns


def test_get_mine_monkeypatched(monkeypatch):
    import helpers as h

    def fake_obs(**kwargs):
        return {"results": [{
            "observed_on": "2026-02-09",
            "taxon.name": "T",
            "species_guess": "G",
            "id": 123,
            "photos": [{"url": "http://example.com/square.jpg"}],
        }]}

    if not h.HAS_PYINAT:
        pytest.skip("pyinaturalist not installed; this test covers pyinaturalist path")
    monkeypatch.setattr(h.inat, "get_observations", fake_obs)
    # Avoid plotting side-effects
    monkeypatch.setattr(h.ipyplot, "plot_images", lambda imgs: None)

    # Should run without raising; returns None
    assert h.get_mine("me", STRT=None, FNSH=None) is None


def test_get_park_data_rest_fallback(monkeypatch):
    import helpers as h

    class FakeResp:
        def __init__(self, payload):
            self._payload = payload

        def raise_for_status(self):
            return None

        def json(self):
            return self._payload

    class FakeSession:
        def __init__(self):
            self.calls = 0

        def get(self, url, params=None, timeout=30):
            self.calls += 1
            if params and params.get("taxon_id"):
                return FakeResp({"results": [{"taxon.id": 1, "count": 100}]})
            return FakeResp({"results": [{
                "taxon.id": 1,
                "taxon.name": "Specimen",
                "taxon.preferred_common_name": "Spec",
                "taxon.wikipedia_url": None,
                "count": 10,
            }]})

    monkeypatch.setattr(h, "HAS_PYINAT", False)
    session = FakeSession()
    res = h.get_park_data((0, 0, 1), "any", limit=5, session=session, per_page=10, max_pages=1)
    assert isinstance(res, pd.DataFrame)
    assert not res.empty
    assert "taxon.name" in res.columns


def test_wugs_uses_without_taxon_id():
    import helpers as h

    taxa = h._taxa_for_kind('wugs')
    assert taxa['taxon_id'] == 1
    assert taxa['without_taxon_id'] == h.VERTEBRATES_TAXON_ID
    assert 'not_id' not in taxa


def test_get_filtered_photo_url_applies_phenology_filters():
    import helpers as h

    class FakeResp:
        def raise_for_status(self):
            return None

        def json(self):
            return {
                'results': [
                    {'photos': [{'url': 'https://example.org/photos/1/square.jpg'}]}
                ]
            }

    class FakeSession:
        def __init__(self):
            self.last_params = None

        def get(self, url, params=None, timeout=30):
            self.last_params = params
            return FakeResp()

    session = FakeSession()
    photo_url = h._get_filtered_photo_url(
        session=session,
        taxon_id=123,
        taxa_filters={'term_id': 12, 'term_value_id': 13},
        place_filters={'lat': 1.0, 'lng': 2.0, 'radius': 3.0},
        time_filters=[{'month': 2, 'day': [1, 2, 3]}],
        fallback_url='https://fallback/image.jpg',
    )

    assert photo_url == 'https://example.org/photos/1/medium.jpg'
    assert session.last_params['term_id'] == 12
    assert session.last_params['term_value_id'] == 13
    assert session.last_params['taxon_id'] == 123


def test_coming_soon_defaults_to_any(monkeypatch):
    import helpers as h

    def fake_counts(**kwargs):
        if "month" in kwargs or "day" in kwargs:
            return {"results": [{
                "taxon.id": 1,
                "taxon.name": "Specimen",
                "taxon.preferred_common_name": "Spec",
                "taxon.wikipedia_url": "https://example.org/wiki/specimen",
                "taxon.default_photo.medium_url": "https://example.org/specimen.jpg",
                "count": 3,
            }]}
        return {"results": []}

    if not h.HAS_PYINAT:
        pytest.skip("pyinaturalist not installed; this test covers pyinaturalist path")
    monkeypatch.setattr(h.inat, "get_observation_species_counts", fake_counts)

    res = h.coming_soon(loc=(0, 0, 1), limit=3)
    assert isinstance(res, pd.DataFrame)
    assert not res.empty
    assert "taxon.name" in res.columns


def test_infer_nativity_from_row_establishment_means():
    import helpers as h

    row = pd.Series({'taxon.establishment_means.establishment_means': 'introduced'})
    assert h._infer_nativity_from_row(row) == 'Introduced'


def test_format_photo_title_contains_all_fields():
    import helpers as h

    title = h._format_photo_title('Common Name', 'Scientificus exampleus', 'Native')
    assert 'Common Name' in title
    assert 'Scientificus exampleus' in title
    assert 'Nativity: Native' in title


def test_lookup_nativity_returns_native_when_present():
    import helpers as h

    class FakeResp:
        def __init__(self, payload):
            self._payload = payload

        def raise_for_status(self):
            return None

        def json(self):
            return self._payload

    class FakeSession:
        def __init__(self):
            self.last_params = None

        def get(self, url, params=None, timeout=20):
            self.last_params = params
            if params.get('native') == 'true':
                return FakeResp({'results': [{'taxon.id': 1, 'count': 1}]})
            return FakeResp({'results': []})

    session = FakeSession()
    nativity = h._lookup_nativity_via_species_counts(session=session, taxon_id=1, nativity_place_id=1297)
    assert nativity == 'Native'
    assert session.last_params.get('place_id') == 1297


def test_lookup_nativity_returns_unknown_when_no_status_found():
    import helpers as h

    class FakeResp:
        def __init__(self, payload):
            self._payload = payload

        def raise_for_status(self):
            return None

        def json(self):
            return self._payload

    class FakeSession:
        def get(self, url, params=None, timeout=20):
            return FakeResp({'results': []})

    session = FakeSession()
    nativity = h._lookup_nativity_via_species_counts(session=session, taxon_id=1)
    assert nativity == 'Unknown'


def test_coming_soon_lineage_filter_introduced(monkeypatch):
    import helpers as h

    def fake_counts(**kwargs):
        if "month" in kwargs or "day" in kwargs:
            return {"results": [
                {
                    "taxon.id": 1,
                    "taxon.name": "Native specimen",
                    "taxon.preferred_common_name": "Native",
                    "taxon.wikipedia_url": "https://example.org/wiki/native",
                    "taxon.default_photo.medium_url": "https://example.org/native.jpg",
                    "count": 5,
                },
                {
                    "taxon.id": 2,
                    "taxon.name": "Introduced specimen",
                    "taxon.preferred_common_name": "Introduced",
                    "taxon.wikipedia_url": "https://example.org/wiki/introduced",
                    "taxon.default_photo.medium_url": "https://example.org/introduced.jpg",
                    "count": 5,
                },
            ]}
        return {"results": []}

    def fake_nativity(session, taxon_id, nativity_place_id=None):
        return "Introduced" if int(taxon_id) == 2 else "Native"

    if not h.HAS_PYINAT:
        pytest.skip("pyinaturalist not installed; this test covers pyinaturalist path")

    monkeypatch.setattr(h.inat, "get_observation_species_counts", fake_counts)
    monkeypatch.setattr(h, "_lookup_nativity_via_species_counts", fake_nativity)

    res = h.coming_soon(loc=(0, 0, 1), limit=10, lineage_filter='introduced', nativity_place_id=1297)
    assert not res.empty
    assert set(res['taxon.id'].astype(int).tolist()) == {2}


def test_coming_soon_lineage_filter_native_endemic(monkeypatch):
    import helpers as h

    def fake_counts(**kwargs):
        if "month" in kwargs or "day" in kwargs:
            return {"results": [
                {
                    "taxon.id": 1,
                    "taxon.name": "Native specimen",
                    "taxon.preferred_common_name": "Native",
                    "taxon.wikipedia_url": "https://example.org/wiki/native",
                    "taxon.default_photo.medium_url": "https://example.org/native.jpg",
                    "count": 5,
                },
                {
                    "taxon.id": 2,
                    "taxon.name": "Introduced specimen",
                    "taxon.preferred_common_name": "Introduced",
                    "taxon.wikipedia_url": "https://example.org/wiki/introduced",
                    "taxon.default_photo.medium_url": "https://example.org/introduced.jpg",
                    "count": 5,
                },
            ]}
        return {"results": []}

    def fake_nativity(session, taxon_id, nativity_place_id=None):
        return "Introduced" if int(taxon_id) == 2 else "Native"

    if not h.HAS_PYINAT:
        pytest.skip("pyinaturalist not installed; this test covers pyinaturalist path")

    monkeypatch.setattr(h.inat, "get_observation_species_counts", fake_counts)
    monkeypatch.setattr(h, "_lookup_nativity_via_species_counts", fake_nativity)

    res = h.coming_soon(loc=(0, 0, 1), limit=10, lineage_filter='native_endemic', nativity_place_id=1297)
    assert not res.empty
    assert set(res['taxon.id'].astype(int).tolist()) == {1}


def test_resolve_nativity_place_id_auto_from_places():
    import helpers as h

    session = object()
    resolved = h._resolve_nativity_place_id(session=session, nativity_place_id='auto', places=[1297, 97394], loc=None)
    assert resolved == 1297


def test_resolve_nativity_place_id_auto_from_loc(monkeypatch):
    import helpers as h

    monkeypatch.setattr(h, '_derive_place_id_from_location', lambda session, lat, lng: 54321)
    session = object()
    resolved = h._resolve_nativity_place_id(session=session, nativity_place_id='auto', places=None, loc=(37.6, -77.8, 25))
    assert resolved == 54321


def test_resolve_nativity_place_id_none_means_global():
    import helpers as h

    session = object()
    resolved = h._resolve_nativity_place_id(session=session, nativity_place_id=None, places=[1297], loc=(0, 0, 1))
    assert resolved is None


def test_lookup_nativity_prefers_introduced_over_native_when_both_present():
    import helpers as h

    class FakeResp:
        def __init__(self, payload):
            self._payload = payload

        def raise_for_status(self):
            return None

        def json(self):
            return self._payload

    class FakeSession:
        def get(self, url, params=None, timeout=20):
            if params.get('introduced') == 'true':
                return FakeResp({'results': [{'taxon.id': 1, 'count': 1}]})
            if params.get('native') == 'true':
                return FakeResp({'results': [{'taxon.id': 1, 'count': 1}]})
            return FakeResp({'results': []})

    session = FakeSession()
    nativity = h._lookup_nativity_via_species_counts(session=session, taxon_id=1, nativity_place_id=1297)
    assert nativity == 'Introduced'


def test_coming_soon_accepts_places_list(monkeypatch):
    import helpers as h

    captured = {'place_id': None}

    def fake_counts(**kwargs):
        if 'month' in kwargs or 'day' in kwargs:
            captured['place_id'] = kwargs.get('place_id')
            return {'results': [{
                'taxon.id': 1,
                'taxon.name': 'Specimen',
                'taxon.preferred_common_name': 'Spec',
                'taxon.wikipedia_url': 'https://example.org/wiki/specimen',
                'taxon.default_photo.medium_url': 'https://example.org/specimen.jpg',
                'count': 1,
            }]}
        return {'results': []}

    if not h.HAS_PYINAT:
        pytest.skip("pyinaturalist not installed; this test covers pyinaturalist path")
    monkeypatch.setattr(h.inat, 'get_observation_species_counts', fake_counts)

    res = h.coming_soon(places=[1297], limit=1)
    assert not res.empty
    assert captured['place_id'] == [1297]


def test_coming_soon_places_conflicts_with_loc():
    import helpers as h

    with pytest.raises(AssertionError, match='only one of places and loc should be provided'):
        h.coming_soon(places=[1], loc=(0, 0, 1), limit=1)

