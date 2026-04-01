import pytest

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

    client = INatAPIClient(session=FakeSession(), api_version='v1', max_retries=1)
    payload = client.observations(params={'user_id': 'demo'})
    assert payload['results'][0]['id'] == 1
