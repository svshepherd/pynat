import pytest
import time

from inat_api import INatAPIClient


def test_inat_api_client_uses_v2_base_url():
    client = INatAPIClient(api_version='v2')
    assert client.endpoint('observations') == 'https://api.inaturalist.org/v2/observations'


def test_inat_api_client_get_json_retries_on_429():
    class FakeResp:
        def __init__(self, status_code, payload, headers=None):
            self.status_code = status_code
            self._payload = payload
            self.headers = headers or {}

        def raise_for_status(self):
            if self.status_code >= 400 and self.status_code not in {403, 429, 500, 502, 503, 504}:
                raise RuntimeError('HTTP error')

        def json(self):
            return self._payload

    class FakeSession:
        def __init__(self):
            self.calls = 0

        def get(self, url, params=None, timeout=30):
            self.calls += 1
            if self.calls == 1:
                return FakeResp(429, {'results': []}, headers={'Retry-After': '0'})
            return FakeResp(200, {'results': [{'id': 1}]})

    client = INatAPIClient(session=FakeSession(), api_version='v1', max_retries=1, min_request_interval=0)
    payload = client.observations(params={'user_id': 'demo'})
    assert payload['results'][0]['id'] == 1


def test_rate_limiter_enforces_min_interval():
    """Verify that min_request_interval causes at least that delay between requests."""

    class FakeResp:
        status_code = 200
        headers = {}

        def raise_for_status(self):
            pass

        def json(self):
            return {'results': []}

    class TimingSession:
        def __init__(self):
            self.timestamps = []

        def get(self, url, params=None, timeout=30):
            self.timestamps.append(time.monotonic())
            return FakeResp()

    session = TimingSession()
    interval = 0.15
    client = INatAPIClient(session=session, api_version='v2', min_request_interval=interval)

    client.get_json('observations')
    client.get_json('observations')
    client.get_json('observations')

    assert len(session.timestamps) == 3
    for i in range(1, len(session.timestamps)):
        elapsed = session.timestamps[i] - session.timestamps[i - 1]
        assert elapsed >= interval * 0.9, f'Gap {elapsed:.3f}s < expected {interval}s'


def test_rate_limiter_zero_disables_throttle():
    """min_request_interval=0 should not add any sleep."""

    class FakeResp:
        status_code = 200
        headers = {}

        def raise_for_status(self):
            pass

        def json(self):
            return {'results': []}

    class TimingSession:
        def __init__(self):
            self.call_count = 0

        def get(self, url, params=None, timeout=30):
            self.call_count += 1
            return FakeResp()

    session = TimingSession()
    client = INatAPIClient(session=session, api_version='v2', min_request_interval=0)

    start = time.monotonic()
    for _ in range(5):
        client.get_json('observations')
    elapsed = time.monotonic() - start

    assert session.call_count == 5
    assert elapsed < 1.0, f'5 requests with interval=0 took {elapsed:.2f}s, expected <1s'


def test_observation_histogram_method():
    """Verify observation_histogram() hits the correct endpoint."""

    class FakeResp:
        status_code = 200
        headers = {}

        def raise_for_status(self):
            pass

        def json(self):
            return {'results': {'month_of_year': {'1': 10, '2': 20}}}

    class FakeSession:
        def __init__(self):
            self.last_url = None

        def get(self, url, params=None, timeout=30):
            self.last_url = url
            return FakeResp()

    session = FakeSession()
    client = INatAPIClient(session=session, api_version='v2', min_request_interval=0)
    payload = client.observation_histogram(params={'taxon_id': 47157, 'interval': 'month_of_year'})

    assert 'observations/histogram' in session.last_url
    assert '/v2/' in session.last_url
    assert payload['results']['month_of_year']['1'] == 10
