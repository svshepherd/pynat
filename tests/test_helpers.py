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
                return FakeResp({'total_results': 1, 'results': []})
            return FakeResp({'total_results': 0, 'results': []})

    session = FakeSession()
    label, resolved_pid = h._lookup_nativity_via_species_counts(session=session, taxon_id=1, nativity_place_id=1297)
    assert label == 'Native'
    assert resolved_pid == 1297
    assert session.last_params.get('per_page') == 0
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
            return FakeResp({'total_results': 0, 'results': []})

    session = FakeSession()
    label, resolved_pid = h._lookup_nativity_via_species_counts(session=session, taxon_id=1)
    assert label == 'Unknown'
    assert resolved_pid is None


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
        label = "Introduced" if int(taxon_id) == 2 else "Native"
        return (label, nativity_place_id)

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
        label = "Introduced" if int(taxon_id) == 2 else "Native"
        return (label, nativity_place_id)

    if not h.HAS_PYINAT:
        pytest.skip("pyinaturalist not installed; this test covers pyinaturalist path")

    monkeypatch.setattr(h.inat, "get_observation_species_counts", fake_counts)
    monkeypatch.setattr(h, "_lookup_nativity_via_species_counts", fake_nativity)

    res = h.coming_soon(loc=(0, 0, 1), limit=10, lineage_filter='native_endemic', nativity_place_id=1297)
    assert not res.empty
    assert set(res['taxon.id'].astype(int).tolist()) == {1}


def test_coming_soon_lineage_filter_native_keeps_unknown(monkeypatch):
    """Taxa with Unknown nativity should be kept under native_endemic filter."""
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
                    "taxon.name": "Unknown specimen",
                    "taxon.preferred_common_name": "Mystery",
                    "taxon.wikipedia_url": "https://example.org/wiki/mystery",
                    "taxon.default_photo.medium_url": "https://example.org/mystery.jpg",
                    "count": 3,
                },
                {
                    "taxon.id": 3,
                    "taxon.name": "Introduced specimen",
                    "taxon.preferred_common_name": "Introduced",
                    "taxon.wikipedia_url": "https://example.org/wiki/introduced",
                    "taxon.default_photo.medium_url": "https://example.org/introduced.jpg",
                    "count": 5,
                },
            ]}
        return {"results": []}

    def fake_nativity(session, taxon_id, nativity_place_id=None):
        mapping = {1: "Native", 2: "Unknown", 3: "Introduced"}
        return (mapping.get(int(taxon_id), "Unknown"), nativity_place_id)

    if not h.HAS_PYINAT:
        pytest.skip("pyinaturalist not installed; this test covers pyinaturalist path")

    monkeypatch.setattr(h.inat, "get_observation_species_counts", fake_counts)
    monkeypatch.setattr(h, "_lookup_nativity_via_species_counts", fake_nativity)

    res = h.coming_soon(loc=(0, 0, 1), limit=10, lineage_filter='native_endemic', nativity_place_id=1297)
    assert not res.empty
    # Native + Unknown kept; Introduced dropped
    assert set(res['taxon.id'].astype(int).tolist()) == {1, 2}


def test_taxa_for_kind_plants_uses_taxon_id():
    import helpers as h

    taxa = h._taxa_for_kind('plants')
    assert taxa == {'taxon_id': 47126}


def test_taxa_for_kind_flowers_includes_taxon_id():
    import helpers as h

    taxa = h._taxa_for_kind('flowers')
    assert taxa['taxon_id'] == 47126
    assert 'term_id' in taxa
    assert 'term_value_id' in taxa


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
                return FakeResp({'total_results': 1, 'results': []})
            if params.get('native') == 'true':
                return FakeResp({'total_results': 1, 'results': []})
            return FakeResp({'total_results': 0, 'results': []})

    session = FakeSession()
    label, resolved_pid = h._lookup_nativity_via_species_counts(session=session, taxon_id=1, nativity_place_id=1297)
    assert label == 'Introduced'
    assert resolved_pid == 1297


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


def test_get_observation_rows_pyinat(monkeypatch):
    import helpers as h

    test_observation = {
        'id': 101,
        'observed_on': '2026-03-01',
        'time_observed_at': '2026-03-01T14:00:00Z',
        'created_at': '2026-03-01T15:00:00Z',
        'quality_grade': 'research',
        'place_guess': 'Virginia Piedmont',
        'user': {'id': 11, 'login': 'observer'},
        'taxon': {
            'id': 47157,
            'name': 'Danaus plexippus',
            'preferred_common_name': 'Monarch',
            'rank': 'species',
            'iconic_taxon_name': 'Insecta',
            'default_photo': {'medium_url': 'https://example.org/default.jpg'},
        },
        'geojson': {'coordinates': [-77.5, 37.6]},
        'identifications_count': 2,
        'num_identification_agreements': 1,
        'num_identification_disagreements': 0,
        'observation_photos': [{'photo': {'url': 'https://example.org/photos/1/square.jpg'}}],
        'identifications': [
            {
                'created_at': '2026-03-01T15:00:00Z',
                'own_observation': True,
                'user': {'id': 11},
            },
            {
                'created_at': '2026-03-03T12:00:00Z',
                'own_observation': False,
                'user': {'id': 12},
            },
        ],
    }

    class FakeSession:
        def __init__(self):
            self.call_count = 0

        def get(self, url, params=None, timeout=None):
            # Verify we're calling the observations endpoint
            assert 'observations' in url
            self.call_count += 1
            
            class FakeResp:
                def raise_for_status(self):
                    return None

                def json(self_inner):
                    # Call 1 is preflight (per_page=0), call 2 is data page 1
                    if self.call_count == 1:
                        return {'total_results': 1, 'results': []}
                    if self.call_count == 2:
                        return {'results': [test_observation], 'total_results': 1}
                    return {'results': []}
            
            resp = FakeResp()
            resp.call_count = self.call_count
            return resp

    fake_session = FakeSession()
    frame = h.get_observation_rows(
        kind='butterflies',
        project_id='virginia-physiographic-regions-piedmont',
        d1='2026-03-01',
        d2='2026-03-31',
        per_page=10,
        max_pages=1,
        session=fake_session,
    )

    assert list(frame['obs_id']) == [101]
    assert frame.loc[0, 'query_kind'] == 'butterflies'
    assert frame.loc[0, 'taxon_name'] == 'Danaus plexippus'
    assert frame.loc[0, 'iconic_taxon_name'] == 'Insecta'
    assert frame.loc[0, 'photo_url'] == 'https://example.org/photos/1/medium.jpg'
    assert frame.loc[0, 'all_identification_count'] == 2
    assert frame.loc[0, 'non_owner_identification_count'] == 1
    assert frame.loc[0, 'all_identification_timestamps'] == [
        '2026-03-01T15:00:00+00:00',
        '2026-03-03T12:00:00+00:00',
    ]
    assert frame.loc[0, 'non_owner_identification_timestamps'] == ['2026-03-03T12:00:00+00:00']
    assert frame.loc[0, 'first_non_owner_identification_at'].isoformat() == '2026-03-03T12:00:00+00:00'
    assert frame.loc[0, 'first_non_owner_identification_delay_days'] == pytest.approx(2.5)


def test_get_observation_rows_rest_fallback(monkeypatch):
    import helpers as h

    class FakeResp:
        def raise_for_status(self):
            return None

        def json(self):
            return {
                'results': [
                    {
                        'id': 202,
                        'observed_on': '2026-02-14',
                        'created_at': '2026-02-14T10:00:00Z',
                        'quality_grade': 'needs_id',
                        'location': '37.55,-78.10',
                        'user': {'id': 20, 'login': 'rest-observer'},
                        'taxon': {
                            'id': 123,
                            'name': 'Actias luna',
                            'preferred_common_name': 'Luna Moth',
                            'rank': 'species',
                            'iconic_taxon_name': 'Insecta',
                        },
                        'identifications': [],
                    }
                ]
            }

    class FakeSession:
        def __init__(self):
            self.calls = []

        def get(self, url, params=None, timeout=30):
            self.calls.append((url, params))
            if params.get('per_page') == 0:
                return FakeResp({'total_results': 1, 'results': []})
            return FakeResp()

    monkeypatch.setattr(h, 'HAS_PYINAT', False)
    session = FakeSession()
    frame = h.get_observation_rows(
        kind='any',
        project_id='virginia-physiographic-regions-piedmont',
        per_page=10,
        max_pages=1,
        session=session,
    )

    assert list(frame['obs_id']) == [202]
    assert frame.loc[0, 'observer_login'] == 'rest-observer'
    assert frame.loc[0, 'all_identification_count'] == 0
    assert frame.loc[0, 'non_owner_identification_count'] == 0
    assert frame.loc[0, 'all_identification_timestamps'] == []
    assert frame.loc[0, 'latitude'] == pytest.approx(37.55)
    assert frame.loc[0, 'longitude'] == pytest.approx(-78.10)
    # calls[0] is preflight (per_page=0), calls[1] is data page
    assert session.calls[1][1]['project_id'] == 'virginia-physiographic-regions-piedmont'
    # Default observation_fields should be passed as JSON 'fields' param
    assert 'fields' in session.calls[1][1]


def test_get_observation_rows_no_fields_when_none(monkeypatch):
    """Passing observation_fields=None omits the fields param entirely."""
    import helpers as h

    class FakeResp:
        def raise_for_status(self):
            return None

        def json(self):
            return {'results': []}

    class FakeSession:
        def __init__(self):
            self.calls = []

        def get(self, url, params=None, timeout=30):
            self.calls.append((url, params))
            return FakeResp()

    monkeypatch.setattr(h, 'HAS_PYINAT', False)
    session = FakeSession()
    h.get_observation_rows(
        kind='any',
        per_page=10,
        max_pages=1,
        observation_fields=None,
        session=session,
    )

    # calls[0] is preflight (per_page=0), calls[1] would be data page if any results
    # With 0 total_results from preflight, there's only the preflight call
    # Preflight never includes 'fields', data calls should also omit it
    for _, params in session.calls:
        if params.get('per_page', 0) != 0:
            assert 'fields' not in params


def test_annotate_taxon_nativity_reuses_cache(monkeypatch):
    import helpers as h

    calls = []

    def fake_batch_lookup(session, taxon_ids, nativity_place_id=None, chunk_size=100):
        calls.append((sorted(taxon_ids), nativity_place_id))
        return {tid: 'Native' for tid in taxon_ids}

    monkeypatch.setattr(h, '_batch_lookup_nativity', fake_batch_lookup)

    frame = pd.DataFrame({'taxon_id': [1, 1, 2, None]})
    out = h.annotate_taxon_nativity(frame, nativity_place_id=1297, session=object())

    assert out['nativity'].tolist() == ['Native', 'Native', 'Native', 'Unknown']
    # Should be called once with both unique taxon IDs
    assert len(calls) == 1
    assert calls[0] == ([1, 2], 1297)


def test_summarize_time_series_counts_and_means():
    import helpers as h

    frame = pd.DataFrame(
        {
            'obs_id': [1, 2, 3],
            'iconic_taxon_name': ['Insecta', 'Insecta', 'Plantae'],
            'observed_on': ['2026-03-01', '2026-03-15', '2026-04-01'],
            'first_non_owner_identification_delay_days': [2.0, 4.0, 1.0],
        }
    )

    summary = h.summarize_time_series(
        frame,
        date_col='observed_on',
        freq='M',
        group_cols=['iconic_taxon_name'],
        value_aggs={'mean_delay_days': ('first_non_owner_identification_delay_days', 'mean')},
    )

    insect_row = summary[summary['iconic_taxon_name'] == 'Insecta'].iloc[0]
    assert insect_row['count'] == 2
    assert insect_row['mean_delay_days'] == pytest.approx(3.0)


def test_batch_lookup_nativity_classifies_in_bulk():
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
            self.call_count = 0

        def get(self, url, params=None, timeout=20):
            self.call_count += 1
            status_key = None
            for k in ('endemic', 'introduced', 'native'):
                if params.get(k) == 'true':
                    status_key = k
                    break
            per_page = params.get('per_page', 0)

            if status_key == 'endemic':
                if per_page == 0:
                    return FakeResp({'total_results': 0, 'results': []})
                return FakeResp({'total_results': 0, 'results': []})

            if status_key == 'introduced':
                if per_page == 0:
                    return FakeResp({'total_results': 1, 'results': []})
                return FakeResp({'total_results': 1, 'results': [
                    {'taxon': {'id': 2}, 'count': 5},
                ]})

            if status_key == 'native':
                if per_page == 0:
                    return FakeResp({'total_results': 1, 'results': []})
                return FakeResp({'total_results': 1, 'results': [
                    {'taxon': {'id': 1}, 'count': 10},
                ]})

            return FakeResp({'total_results': 0, 'results': []})

    session = FakeSession()
    result = h._batch_lookup_nativity(session, [1, 2, 3], nativity_place_id=1297)

    assert result[1] == 'Native'
    assert result[2] == 'Introduced'
    assert result[3] == 'Unknown'
    # 3 per_page=0 probes + 2 detail fetches (introduced hit, native hit),
    # endemic hit had total_results=0 so no detail fetch
    assert session.call_count == 5


def test_batch_lookup_nativity_empty_list():
    import helpers as h
    result = h._batch_lookup_nativity(object(), [])
    assert result == {}


def test_get_observation_rows_preflight_limits_pages(monkeypatch):
    """Preflight per_page=0 check should limit actual pages fetched."""
    import helpers as h

    class FakeResp:
        def __init__(self, payload):
            self._payload = payload

        def raise_for_status(self):
            return None

        def json(self):
            return self._payload

    observation = {
        'id': 1,
        'observed_on': '2026-03-01',
        'created_at': '2026-03-01T10:00:00Z',
        'quality_grade': 'research',
        'user': {'id': 1, 'login': 'u'},
        'taxon': {'id': 1, 'name': 'T', 'preferred_common_name': 'T', 'rank': 'species'},
        'identifications': [],
    }

    class FakeSession:
        def __init__(self):
            self.calls = []

        def get(self, url, params=None, timeout=30):
            self.calls.append(params)
            per_page = params.get('per_page', 0)
            if per_page == 0:
                # Preflight: 15 total results
                return FakeResp({'total_results': 15, 'results': []})
            # Data pages: return full page of results each time
            return FakeResp({'results': [observation] * per_page})

    monkeypatch.setattr(h, 'HAS_PYINAT', False)
    session = FakeSession()
    frame = h.get_observation_rows(
        kind='any',
        per_page=10,
        max_pages=5,
        session=session,
    )

    # 15 results / 10 per_page = 2 pages needed; max_pages=5 should be clamped to 2
    data_calls = [c for c in session.calls if c.get('per_page', 0) != 0]
    assert len(data_calls) == 2
    assert len(frame) == 20  # 10 per page * 2 pages


def test_get_observation_rows_preflight_zero_results(monkeypatch):
    """When preflight says 0 results, return empty immediately without data calls."""
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
            self.calls = []

        def get(self, url, params=None, timeout=30):
            self.calls.append(params)
            return FakeResp({'total_results': 0, 'results': []})

    monkeypatch.setattr(h, 'HAS_PYINAT', False)
    session = FakeSession()
    frame = h.get_observation_rows(
        kind='any',
        per_page=10,
        max_pages=5,
        session=session,
    )

    assert frame.empty
    # Only the preflight call, no data calls
    assert len(session.calls) == 1
    assert session.calls[0].get('per_page') == 0


def test_search_places_returns_results():
    import helpers as h

    class FakeResp:
        def raise_for_status(self):
            return None

        def json(self):
            return {
                'results': [
                    {'id': 42, 'display_name': 'Shenandoah National Park'},
                    {'id': 99, 'display_name': 'Shenandoah Valley'},
                ]
            }

    class FakeSession:
        def __init__(self):
            self.last_params = None

        def get(self, url, params=None, timeout=12):
            assert 'autocomplete' in url
            self.last_params = params
            return FakeResp()

    session = FakeSession()
    results = h._search_places(session, 'Shenandoah')
    assert len(results) == 2
    assert results[0] == {'id': 42, 'display_name': 'Shenandoah National Park'}
    assert session.last_params['q'] == 'Shenandoah'


def test_search_places_empty_query_returns_empty():
    import helpers as h

    results = h._search_places(None, '')
    assert results == []

    results = h._search_places(None, '   ')
    assert results == []


def test_search_places_api_error_returns_empty():
    import helpers as h

    class BrokenSession:
        def get(self, url, params=None, timeout=12):
            raise ConnectionError('network down')

    results = h._search_places(BrokenSession(), 'Virginia')
    assert results == []


def test_lookup_nativity_crawls_to_parent_when_unknown_at_primary():
    """When primary place returns no status, should resolve via state ancestor."""
    import helpers as h

    PARK_ID = 9999
    COUNTY_ID = 100
    STATE_ID = 1297
    COUNTRY_ID = 1

    class FakeResp:
        def __init__(self, payload):
            self._payload = payload

        def raise_for_status(self):
            return None

        def json(self):
            return self._payload

    class FakeSession:
        def get(self, url, params=None, timeout=20):
            # Single-place lookup → returns ancestor list (broadest first)
            if f'/places/{PARK_ID}' in url:
                return FakeResp({'results': [{
                    'id': PARK_ID,
                    'ancestor_place_ids': [COUNTRY_ID, STATE_ID, COUNTY_ID],
                }]})
            # Bulk place lookup for admin_level filtering
            if '/places/' in url and params is None:
                return FakeResp({'results': [
                    {'id': COUNTY_ID, 'admin_level': 20},
                    {'id': STATE_ID, 'admin_level': 10},
                    {'id': COUNTRY_ID, 'admin_level': 0},
                ]})
            # species_counts: only return native at state level
            if params and params.get('place_id') == STATE_ID and params.get('native') == 'true':
                return FakeResp({'total_results': 1, 'results': []})
            return FakeResp({'total_results': 0, 'results': []})

    label, resolved_pid = h._lookup_nativity_via_species_counts(
        session=FakeSession(), taxon_id=42, nativity_place_id=PARK_ID
    )
    assert label == 'Native'
    assert resolved_pid == STATE_ID


def test_lookup_nativity_does_not_crawl_past_state_to_country():
    """A species native somewhere in the US should not resolve as native in California."""
    import helpers as h

    COUNTY_ID = 200   # e.g. Santa Clara County
    STATE_ID = 14     # California
    COUNTRY_ID = 1    # United States

    class FakeResp:
        def __init__(self, payload):
            self._payload = payload

        def raise_for_status(self):
            return None

        def json(self):
            return self._payload

    class FakeSession:
        def get(self, url, params=None, timeout=20):
            if f'/places/{COUNTY_ID}' in url:
                return FakeResp({'results': [{
                    'id': COUNTY_ID,
                    'ancestor_place_ids': [COUNTRY_ID, STATE_ID],
                }]})
            if '/places/' in url and params is None:
                return FakeResp({'results': [
                    {'id': STATE_ID, 'admin_level': 10},
                    {'id': COUNTRY_ID, 'admin_level': 0},
                ]})
            # native=true only hits at country level (SE-US native, not CA native)
            if params and params.get('place_id') == COUNTRY_ID and params.get('native') == 'true':
                return FakeResp({'total_results': 5, 'results': []})
            return FakeResp({'total_results': 0, 'results': []})

    label, resolved_pid = h._lookup_nativity_via_species_counts(
        session=FakeSession(), taxon_id=42, nativity_place_id=COUNTY_ID
    )
    # Country-level match should be excluded; result must be Unknown
    assert label == 'Unknown'
    assert resolved_pid is None


def test_lookup_nativity_ancestor_crawl_exhausted_returns_unknown():
    """Should return Unknown when no ancestor within max_admin_level resolves the status."""
    import helpers as h

    class FakeResp:
        def __init__(self, payload=None):
            self._payload = payload or {'total_results': 0, 'results': []}

        def raise_for_status(self):
            return None

        def json(self):
            return self._payload

    STATE_ID = 10
    COUNTRY_ID = 20

    class FakeSession:
        def get(self, url, params=None, timeout=20):
            if '/places/999' in url:
                return FakeResp({'results': [{'id': 999, 'ancestor_place_ids': [COUNTRY_ID, STATE_ID]}]})
            if '/places/' in url and params is None:
                return FakeResp({'results': [
                    {'id': STATE_ID, 'admin_level': 10},
                    {'id': COUNTRY_ID, 'admin_level': 0},
                ]})
            return FakeResp()

    label, resolved_pid = h._lookup_nativity_via_species_counts(
        session=FakeSession(), taxon_id=42, nativity_place_id=999
    )
    assert label == 'Unknown'
    assert resolved_pid is None

