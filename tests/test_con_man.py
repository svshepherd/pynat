import os
import sys

# Ensure package root is importable when tests run from pynat/tests.
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import pandas as pd

import con_man as cm


def test_summarize_from_proposals_basic_rates():
    analyzer = cm.Analyzer(cache_dir=os.path.join(os.path.dirname(__file__), '..', 'data'))

    df_props = pd.DataFrame([
        {
            'obs_id': 1,
            'taxon_id': 101,
            'taxon_name': 'Spec A',
            'taxon_rank': 'species',
            'status': 'vindicated',
            'correctness_depth': 'species',
        },
        {
            'obs_id': 2,
            'taxon_id': 101,
            'taxon_name': 'Spec A',
            'taxon_rank': 'species',
            'status': 'overruled',
            'correctness_depth': 'wrong',
        },
        {
            'obs_id': 3,
            'taxon_id': 202,
            'taxon_name': 'Genus B',
            'taxon_rank': 'genus',
            'status': 'undecided_support',
            'correctness_depth': 'genus',
        },
    ])

    sp, rk = analyzer.summarize_from_proposals(
        df_props,
        csv=False,
        save_outputs=False,
        print_report=False,
    )

    assert not sp.empty
    assert not rk.empty

    row = sp[sp['taxon_id'] == 101].iloc[0]
    assert row['n_props'] == 2
    assert row['n_vindicated'] == 1
    assert row['n_overruled'] == 1
    assert row['n_vindicated_rate'] == 0.5

    genus_row = rk[rk['taxon_rank'] == 'genus'].iloc[0]
    assert genus_row['undecided_rate'] == 1.0


def test_run_returns_expected_tables(monkeypatch):
    analyzer = cm.Analyzer(cache_dir=os.path.join(os.path.dirname(__file__), '..', 'data'))

    fake_props = pd.DataFrame([
        {
            'obs_id': 1,
            'taxon_id': 101,
            'taxon_name': 'Spec A',
            'taxon_rank': 'species',
            'status': 'vindicated',
            'correctness_depth': 'species',
        }
    ])

    monkeypatch.setattr(analyzer, 'build_proposals', lambda *args, **kwargs: fake_props)

    out = analyzer.run('demo-user', ingest=False, save_outputs=False, print_report=False)

    assert set(out.keys()) == {'proposals', 'summary_by_species', 'summary_by_rank'}
    assert len(out['proposals']) == 1
    assert len(out['summary_by_species']) == 1


def test_run_forwards_ingest_mode(monkeypatch):
    analyzer = cm.Analyzer(cache_dir=os.path.join(os.path.dirname(__file__), '..', 'data'))

    fake_props = pd.DataFrame([
        {
            'obs_id': 1,
            'taxon_id': 101,
            'taxon_name': 'Spec A',
            'taxon_rank': 'species',
            'status': 'vindicated',
            'correctness_depth': 'species',
        }
    ])

    seen = {'mode': None}

    def fake_ingest(user_login, mode='incremental'):
        seen['mode'] = mode

    monkeypatch.setattr(analyzer, 'ingest', fake_ingest)
    monkeypatch.setattr(analyzer, 'build_proposals', lambda *args, **kwargs: fake_props)

    analyzer.run('demo-user', ingest=True, ingest_mode='full', save_outputs=False, print_report=False)

    assert seen['mode'] == 'full'


def test_incremental_ingest_falls_back_to_full_on_first_run(monkeypatch, tmp_path):
    analyzer = cm.Analyzer(cache_dir=str(tmp_path))

    my_ids = [
        {
            'id': 1,
            'created_at': '2024-01-01T00:00:00Z',
            'current': True,
            'observation': {'id': 1001},
            'user': {'id': 11, 'login': 'demo-user'},
            'taxon': {'id': 5001, 'name': 'Spec A', 'rank': 'species'},
        }
    ]
    all_ids = [
        {
            'id': 1,
            'created_at': '2024-01-01T00:00:00Z',
            'current': True,
            'observation': {'id': 1001},
            'user': {'id': 11, 'login': 'demo-user'},
            'taxon': {'id': 5001, 'name': 'Spec A', 'rank': 'species'},
        }
    ]
    obs = [
        {
            'id': 1001,
            'observed_on': '2024-01-01',
            'updated_at': '2024-01-01T01:00:00Z',
            'geojson': {'coordinates': [0.0, 0.0]},
            'place_ids': [],
            'community_taxon': {'id': 5001, 'name': 'Spec A', 'rank': 'species'},
        }
    ]
    taxa = [
        {
            'id': 5001,
            'name': 'Spec A',
            'rank': 'species',
            'ancestor_ids': [1, 2, 3],
        }
    ]

    seen = {'all_ids_calls': 0, 'load_parquet_calls': 0}

    monkeypatch.setattr(analyzer.client, 'user_identifications', lambda user, taxon_id=None: my_ids)

    def fake_identifications_for_observations(obs_ids):
        seen['all_ids_calls'] += 1
        return all_ids

    monkeypatch.setattr(analyzer.client, 'identifications_for_observations', fake_identifications_for_observations)
    monkeypatch.setattr(analyzer.client, 'observations_by_ids', lambda obs_ids: obs)
    monkeypatch.setattr(analyzer.client, 'taxa_by_ids', lambda taxon_ids: taxa)

    def fail_if_called(name):
        seen['load_parquet_calls'] += 1
        raise AssertionError(f'load_parquet should not be called on first-run fallback: {name}')

    monkeypatch.setattr(analyzer, 'load_parquet', fail_if_called)
    monkeypatch.setattr(analyzer, 'save_parquet', lambda df, name: None)

    analyzer.ingest('demo-user', mode='incremental')

    assert seen['all_ids_calls'] == 1
    assert seen['load_parquet_calls'] == 0


def test_assess_taxon_returns_scored_output(monkeypatch, tmp_path):
    analyzer = cm.Analyzer(cache_dir=str(tmp_path))

    my_ids = [
        {
            'id': 1,
            'created_at': '2024-01-01T00:00:00Z',
            'current': True,
            'observation': {'id': 1001},
            'user': {'id': 11, 'login': 'demo-user'},
            'taxon': {'id': 5001, 'name': 'Spec A', 'rank': 'species'},
        },
        {
            'id': 2,
            'created_at': '2024-01-01T00:30:00Z',
            'current': True,
            'observation': {'id': 1002},
            'user': {'id': 11, 'login': 'demo-user'},
            'taxon': {'id': 9999, 'name': 'Other', 'rank': 'species'},
        },
    ]

    all_ids = [
        {
            'id': 1,
            'created_at': '2024-01-01T00:00:00Z',
            'current': True,
            'disagreement': False,
            'observation': {'id': 1001},
            'user': {'id': 11, 'login': 'demo-user'},
            'taxon': {'id': 5001, 'name': 'Spec A', 'rank': 'species'},
        },
        {
            'id': 3,
            'created_at': '2024-01-01T01:00:00Z',
            'current': True,
            'disagreement': False,
            'observation': {'id': 1001},
            'user': {'id': 22, 'login': 'other-user'},
            'taxon': {'id': 5001, 'name': 'Spec A', 'rank': 'species'},
        },
    ]

    obs = [
        {
            'id': 1001,
            'observed_on': '2024-01-01',
            'created_at': '2024-01-01T00:00:00Z',
            'updated_at': '2024-01-01T01:00:00Z',
            'quality_grade': 'research',
            'geojson': {'coordinates': [0.0, 0.0]},
            'place_ids': [],
            'community_taxon': {'id': 5001, 'name': 'Spec A', 'rank': 'species'},
        }
    ]

    taxa = [
        {'id': 5001, 'name': 'Spec A', 'rank': 'species', 'ancestor_ids': [1, 2, 3]},
    ]

    seen = {'taxon_id': None}

    def fake_user_identifications_windowed_best_effort(user, taxon_id, start=None, end=None):
        seen['taxon_id'] = taxon_id
        return my_ids, {
            'partial_results': False,
            'warning': None,
            'loaded_proposal_count': len(my_ids),
            'oldest_loaded_proposed_at': my_ids[0]['created_at'],
            'fetched_pages': 1,
            'order': 'desc',
        }

    monkeypatch.setattr(analyzer.client, 'user_identifications_windowed_best_effort', fake_user_identifications_windowed_best_effort)
    monkeypatch.setattr(analyzer.client, 'identifications_for_observations', lambda obs_ids: all_ids)
    monkeypatch.setattr(analyzer.client, 'observations_by_ids', lambda obs_ids: obs)
    monkeypatch.setattr(analyzer.client, 'taxa_by_ids', lambda taxon_ids: taxa)

    out = analyzer.assess_taxon('demo-user', taxon_id=5001, print_report=False)

    assert len(out['proposals']) == 1
    assert len(out['summary_by_rank']) >= 1
    assert 'taxon_reliability' in out
    assert 'analysis_meta' in out
    assert seen['taxon_id'] == 5001
    prop = out['proposals'].iloc[0]
    assert prop['taxon_id'] == 5001
    assert prop['outcome'] == 'confirmed'

    rel = out['taxon_reliability']
    assert not rel.empty
    assert rel['confirmed_count'].sum() == 1


def test_assess_taxon_handles_mixed_timezone_timestamps(monkeypatch, tmp_path):
    analyzer = cm.Analyzer(cache_dir=str(tmp_path))

    my_ids = [
        {
            'id': 1,
            'created_at': '2024-01-01T00:00:00Z',
            'current': True,
            'observation': {'id': 2001},
            'user': {'id': 11, 'login': 'demo-user'},
            'taxon': {'id': 7001, 'name': 'Spec Z', 'rank': 'species'},
        }
    ]
    all_ids = [
        {
            'id': 1,
            'created_at': '2024-01-01T00:00:00Z',
            'current': True,
            'disagreement': False,
            'observation': {'id': 2001},
            'user': {'id': 11, 'login': 'demo-user'},
            'taxon': {'id': 7001, 'name': 'Spec Z', 'rank': 'species'},
        },
        {
            'id': 2,
            'created_at': '2024-01-01T00:10:00-05:00',
            'current': True,
            'disagreement': False,
            'observation': {'id': 2001},
            'user': {'id': 22, 'login': 'other-user'},
            'taxon': {'id': 7001, 'name': 'Spec Z', 'rank': 'species'},
        },
    ]
    obs = [
        {
            'id': 2001,
            'observed_on': '2024-01-01',
            'created_at': '2024-01-01T00:00:00Z',
            'updated_at': '2024-01-01T01:00:00Z',
            'quality_grade': 'research',
            'geojson': {'coordinates': [0.0, 0.0]},
            'place_ids': [],
            'community_taxon': {'id': 7001, 'name': 'Spec Z', 'rank': 'species'},
        }
    ]
    taxa = [
        {'id': 7001, 'name': 'Spec Z', 'rank': 'species', 'ancestor_ids': [1, 2, 3]},
    ]

    monkeypatch.setattr(
        analyzer.client,
        'user_identifications_windowed_best_effort',
        lambda user, taxon_id, start=None, end=None: (
            my_ids,
            {
                'partial_results': False,
                'warning': None,
                'loaded_proposal_count': len(my_ids),
                'oldest_loaded_proposed_at': my_ids[0]['created_at'],
                'fetched_pages': 1,
                'order': 'desc',
            },
        ),
    )
    monkeypatch.setattr(analyzer.client, 'identifications_for_observations', lambda obs_ids: all_ids)
    monkeypatch.setattr(analyzer.client, 'observations_by_ids', lambda obs_ids: obs)
    monkeypatch.setattr(analyzer.client, 'taxa_by_ids', lambda taxon_ids: taxa)

    out = analyzer.assess_taxon('demo-user', taxon_id=7001, start='2024-01-01', end='2024-01-01', print_report=False)

    assert len(out['proposals']) == 1
    assert out['proposals'].iloc[0]['outcome'] == 'confirmed'


def test_assess_taxon_returns_empty_when_no_matching_taxon(monkeypatch, tmp_path):
    analyzer = cm.Analyzer(cache_dir=str(tmp_path))

    my_ids = [
        {
            'id': 1,
            'created_at': '2024-01-01T00:00:00Z',
            'current': True,
            'observation': {'id': 1001},
            'user': {'id': 11, 'login': 'demo-user'},
            'taxon': {'id': 9999, 'name': 'Other', 'rank': 'species'},
        }
    ]

    monkeypatch.setattr(
        analyzer.client,
        'user_identifications_windowed_best_effort',
        lambda user, taxon_id, start=None, end=None: (
            my_ids,
            {
                'partial_results': False,
                'warning': None,
                'loaded_proposal_count': len(my_ids),
                'oldest_loaded_proposed_at': my_ids[0]['created_at'],
                'fetched_pages': 1,
                'order': 'desc',
            },
        ),
    )
    monkeypatch.setattr(
        analyzer.client,
        'identifications_for_observations',
        lambda obs_ids: (_ for _ in ()).throw(AssertionError('should not fetch timelines for empty taxon match')),
    )

    out = analyzer.assess_taxon('demo-user', taxon_id=5001, print_report=False)

    assert out['proposals'].empty
    assert out['summary_by_species'].empty
    assert out['summary_by_rank'].empty
    assert out['taxon_reliability'].empty
    assert out['analysis_meta']['partial_results'] is False


def test_assess_taxon_reports_partial_results_on_403(monkeypatch, tmp_path):
    analyzer = cm.Analyzer(cache_dir=str(tmp_path))

    my_ids = [
        {
            'id': 1,
            'created_at': '2024-01-15T00:00:00Z',
            'current': True,
            'observation': {'id': 4001},
            'user': {'id': 11, 'login': 'demo-user'},
            'taxon': {'id': 8001, 'name': 'Spec P', 'rank': 'species'},
        }
    ]

    monkeypatch.setattr(
        analyzer.client,
        'user_identifications_windowed_best_effort',
        lambda user, taxon_id, start=None, end=None: (
            my_ids,
            {
                'partial_results': True,
                'warning': 'Partial results due to HTTP 403 while paging identifications. Loaded proposals=1; oldest_loaded_proposed_at=2024-01-15T00:00:00+00:00.',
                'loaded_proposal_count': 1,
                'oldest_loaded_proposed_at': '2024-01-15T00:00:00+00:00',
                'fetched_pages': 1,
                'order': 'desc',
            },
        ),
    )
    monkeypatch.setattr(analyzer.client, 'identifications_for_observations', lambda obs_ids: [])
    monkeypatch.setattr(analyzer.client, 'observations_by_ids', lambda obs_ids: [])
    monkeypatch.setattr(analyzer.client, 'taxa_by_ids', lambda taxon_ids: [])

    out = analyzer.assess_taxon('demo-user', taxon_id=8001, print_report=False)

    assert out['analysis_meta']['partial_results'] is True
    assert out['analysis_meta']['loaded_proposal_count'] == 1
    assert out['analysis_meta']['oldest_loaded_proposed_at'] is not None
