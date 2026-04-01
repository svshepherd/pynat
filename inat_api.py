"""Shared iNaturalist API adapter with lightweight versioning and retries.

This module centralizes API URL construction and basic retry behavior so
callers can migrate endpoint-by-endpoint from v1 to v2 without duplicating
network logic.
"""

from __future__ import annotations

from typing import Any, Optional
import time

import requests


class INatAPIClient:
    """Minimal HTTP client for iNaturalist API endpoints.

    Parameters
    ----------
    session:
        Optional requests session. If omitted, a new session is created.
    api_version:
        API version segment, e.g. ``"v1"`` or ``"v2"``.
    max_retries:
        Number of retries for transient status codes and request errors.
    backoff_base_seconds:
        Base delay for exponential backoff.
    """

    def __init__(
        self,
        session: Optional[requests.Session] = None,
        api_version: str = "v1",
        max_retries: int = 3,
        backoff_base_seconds: float = 1.5,
        retry_statuses: Optional[set[int]] = None,
    ) -> None:
        self.session = session or requests.Session()
        self.api_version = str(api_version).strip("/") or "v1"
        self.base_url = f"https://api.inaturalist.org/{self.api_version}"
        self.max_retries = max(0, int(max_retries))
        self.backoff_base_seconds = float(backoff_base_seconds)
        self.retry_statuses = retry_statuses or {403, 429, 500, 502, 503, 504}

    def endpoint(self, path: str) -> str:
        return f"{self.base_url}/{path.lstrip('/')}"

    def get_json(self, path: str, params: Optional[dict[str, Any]] = None, timeout: int = 30) -> dict[str, Any]:
        """Execute GET with simple retry/backoff and return decoded JSON."""
        url = self.endpoint(path)
        last_exc: Optional[Exception] = None

        for attempt in range(self.max_retries + 1):
            try:
                response = self.session.get(url, params=params, timeout=timeout)
            except requests.RequestException as exc:
                last_exc = exc
                if attempt >= self.max_retries:
                    break
                time.sleep(self.backoff_base_seconds * (2 ** attempt))
                continue

            status_code = int(getattr(response, "status_code", 200) or 200)
            headers = getattr(response, "headers", {}) or {}

            if status_code in self.retry_statuses:
                if attempt >= self.max_retries:
                    response.raise_for_status()
                retry_after = headers.get("Retry-After")
                if retry_after is not None:
                    try:
                        delay = max(0.0, float(retry_after))
                    except ValueError:
                        delay = self.backoff_base_seconds * (2 ** attempt)
                else:
                    delay = self.backoff_base_seconds * (2 ** attempt)
                time.sleep(delay)
                continue

            response.raise_for_status()
            payload = response.json()
            if isinstance(payload, dict):
                return payload
            return {"results": payload if isinstance(payload, list) else []}

        raise RuntimeError(f"iNaturalist request failed after retries: {url} params={params}") from last_exc

    def observations(self, params: Optional[dict[str, Any]] = None, timeout: int = 30) -> dict[str, Any]:
        """Query ``/observations`` with the given filter parameters."""
        return self.get_json("observations", params=params, timeout=timeout)

    def observation_species_counts(self, params: Optional[dict[str, Any]] = None, timeout: int = 30) -> dict[str, Any]:
        """Query ``/observations/species_counts`` for per-taxon observation totals."""
        return self.get_json("observations/species_counts", params=params, timeout=timeout)

    def places(self, place_id: Optional[int] = None, params: Optional[dict[str, Any]] = None, timeout: int = 30) -> dict[str, Any]:
        """Fetch places via ``/places`` (search) or ``/places/{id}`` (detail)."""
        path = "places" if place_id is None else f"places/{int(place_id)}"
        return self.get_json(path, params=params, timeout=timeout)

    def places_nearby(self, params: Optional[dict[str, Any]] = None, timeout: int = 30) -> dict[str, Any]:
        """Geocode coordinates to nearby places via ``/places/nearby``."""
        return self.get_json("places/nearby", params=params, timeout=timeout)

    def places_autocomplete(self, params: Optional[dict[str, Any]] = None, timeout: int = 30) -> dict[str, Any]:
        """Search places by name via ``/places/autocomplete``."""
        return self.get_json("places/autocomplete", params=params, timeout=timeout)
