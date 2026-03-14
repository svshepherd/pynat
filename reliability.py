#!/usr/bin/env python3
"""Lean reliability analyzer for one iNaturalist user and one target taxon.

Primary goal
Estimate how reliably a user's proposals for a specific taxon predict the
resolved community identification.

Primary workflow
1) Fetch identifications for the user scoped to one target taxon.
2) Build proposal events from timeline data (including withdrawn proposals).
3) Classify proposals as confirmed, disconfirmed, or unresolved.
4) Report overlap depth and confirmed/disconfirmed ratios by taxonomic level.

This file still contains deprecated cache-backed ingest/summarize methods for
backward compatibility, but the taxon-scoped assess path is the intended
lightweight entrypoint.
"""
from __future__ import annotations

import argparse
import collections
import dataclasses
import datetime as dt
import itertools
import json
import math
import os
import sys
import time
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple

import pandas as pd
import requests

try:
    from tqdm import tqdm
except Exception:  # pragma: no cover
    def tqdm(x, **kwargs):
        return x

BASE = "https://api.inaturalist.org/v1"
PER_PAGE = 200
SLEEP = 0.2  # seconds between paginated calls
DEFAULT_CACHE_DIR = os.path.join(os.path.dirname(__file__), "data")
REQUIRED_CACHE_FILES = (
    "my_identifications.parquet",
    "all_identifications.parquet",
    "observations.parquet",
    "taxa.parquet",
)

# -------------------------
# Utilities
# -------------------------

def log(msg: str):
    print(msg, file=sys.stderr)


def chunks(seq: List, n: int) -> Iterable[List]:
    for i in range(0, len(seq), n):
        yield seq[i : i + n]


# -------------------------
# API Client (v1)
# -------------------------

class INatClient:
    """Thin iNaturalist API v1 client with pagination helpers.

    The client intentionally exposes a narrow subset of endpoints used by this
    analysis pipeline. Each public method returns raw JSON-like dictionaries so
    normalization logic remains centralized in ``Analyzer.ingest``.
    """

    def __init__(
        self,
        base=BASE,
        per_page=PER_PAGE,
        sleep=SLEEP,
        max_retries: int = 4,
        backoff_base_seconds: float = 1.5,
        backoff_cap_seconds: float = 90.0,
        retry_statuses: Optional[Set[int]] = None,
    ):
        self.base = base
        self.per_page = per_page
        self.sleep = sleep
        self.max_retries = max_retries
        self.backoff_base_seconds = backoff_base_seconds
        self.backoff_cap_seconds = backoff_cap_seconds
        self.retry_statuses = retry_statuses or {403, 429, 500, 502, 503, 504}
        self.session = requests.Session()
        self.session.headers.update({"User-Agent": "inat-reliability/0.2 (+https://github.com/svshepherd/pynat)"})

    def _retry_delay_seconds(self, attempt_number: int, response: Optional[requests.Response] = None) -> float:
        if response is not None:
            retry_after = response.headers.get("Retry-After")
            if retry_after:
                try:
                    return max(0.0, float(retry_after))
                except ValueError:
                    pass
        exp = self.backoff_base_seconds * (2 ** max(0, attempt_number - 1))
        return min(self.backoff_cap_seconds, exp)

    def _get(self, endpoint: str, params: Dict) -> Dict:
        url = f"{self.base}/{endpoint}"
        last_exc: Optional[Exception] = None
        for attempt in range(self.max_retries + 1):
            try:
                r = self.session.get(url, params=params, timeout=60)
            except requests.RequestException as e:
                last_exc = e
                if attempt >= self.max_retries:
                    break
                delay = self._retry_delay_seconds(attempt + 1)
                log(f"Request error for {endpoint}; retrying in {delay:.1f}s ({attempt + 1}/{self.max_retries})")
                time.sleep(delay)
                continue

            if r.status_code in self.retry_statuses:
                if attempt >= self.max_retries:
                    r.raise_for_status()
                delay = self._retry_delay_seconds(attempt + 1, response=r)
                log(
                    f"HTTP {r.status_code} for {endpoint}; retrying in {delay:.1f}s "
                    f"({attempt + 1}/{self.max_retries})"
                )
                time.sleep(delay)
                continue

            r.raise_for_status()
            return r.json()

        raise RuntimeError(
            f"iNaturalist API request failed after retries for endpoint '{endpoint}' with params {params}."
        ) from last_exc

    def paged(self, endpoint: str, params: Dict) -> Iterable[Dict]:
        page = 1
        total_pages = None
        while True:
            p = dict(params)
            p.update({"page": page, "per_page": self.per_page})
            j = self._get(endpoint, p)
            results = j.get("results", [])
            if not results:
                break
            for item in results:
                yield item
            total_results = j.get("total_results")
            if total_results is not None:
                total_pages = math.ceil(total_results / self.per_page)
            page += 1
            if total_pages and page > total_pages:
                break
            time.sleep(self.sleep)

    # ---- Specific endpoints ----
    def taxa_by_query(self, q: str, per_page=1) -> List[Dict]:
        return self._get("taxa", {"q": q, "per_page": per_page}).get("results", [])

    def taxa_by_ids(self, taxon_ids: List[int]) -> List[Dict]:
        out = []
        for batch in chunks(list(dict.fromkeys(taxon_ids)), 100):
            j = self._get("taxa", {"id": ",".join(map(str, batch)), "per_page": 100})
            out.extend(j.get("results", []))
            time.sleep(self.sleep)
        return out

    def user_identifications(self, user: str, taxon_id: Optional[int] = None) -> List[Dict]:
        """Return a user's identifications, optionally scoped to one taxon."""
        params = {
            "user_id": user,
            "current": "any",  # include withdrawn
            "order_by": "created_at",
            "order": "asc",
        }
        if taxon_id is not None:
            params["taxon_id"] = int(taxon_id)
        return list(self.paged("identifications", params))

    def user_identifications_windowed_best_effort(
        self,
        user: str,
        taxon_id: int,
        start: Optional[str] = None,
        end: Optional[str] = None,
        include_withdrawn: bool = True,
    ) -> Tuple[List[Dict], Dict[str, Any]]:
        """Fetch user identifications newest-first in a time window.

        If a 403 occurs mid-pagination, returns already-loaded results and
        metadata describing partial coverage instead of failing hard.
        """
        start_utc = pd.to_datetime(start, utc=True, errors="coerce") if start else None
        end_utc = pd.to_datetime(end, utc=True, errors="coerce") if end else None
        if start_utc is not None and pd.notna(start_utc):
            start_utc = start_utc.normalize()
        else:
            start_utc = None
        if end_utc is not None and pd.notna(end_utc):
            end_exclusive = end_utc.normalize() + pd.Timedelta(days=1)
        else:
            end_exclusive = None

        page = 1
        total_pages = None
        out: List[Dict] = []
        partial_results = False
        fetched_pages = 0
        stop_due_to_window = False
        oldest_loaded_ts = None

        while True:
            params = {
                "user_id": user,
                "taxon_id": int(taxon_id),
                "current": "any" if include_withdrawn else "true",
                "order_by": "created_at",
                "order": "desc",
                "page": page,
                "per_page": self.per_page,
            }
            try:
                j = self._get("identifications", params)
            except requests.HTTPError as e:
                if e.response is not None and e.response.status_code == 403:
                    partial_results = True
                    break
                raise

            results = j.get("results", [])
            if not results:
                break

            fetched_pages += 1
            for item in results:
                created = pd.to_datetime(item.get("created_at"), utc=True, errors="coerce")
                if pd.isna(created):
                    continue
                if end_exclusive is not None and created >= end_exclusive:
                    continue
                if start_utc is not None and created < start_utc:
                    stop_due_to_window = True
                    continue
                out.append(item)
                if oldest_loaded_ts is None or created < oldest_loaded_ts:
                    oldest_loaded_ts = created

            if stop_due_to_window:
                break

            total_results = j.get("total_results")
            if total_results is not None:
                total_pages = math.ceil(total_results / self.per_page)
            page += 1
            if total_pages and page > total_pages:
                break
            time.sleep(self.sleep)

        warning_message = None
        if partial_results:
            oldest_txt = oldest_loaded_ts.isoformat() if oldest_loaded_ts is not None else "none"
            warning_message = (
                "Partial results due to HTTP 403 while paging identifications. "
                f"Loaded proposals={len(out)}; oldest_loaded_proposed_at={oldest_txt}."
            )

        return out, {
            "partial_results": partial_results,
            "warning": warning_message,
            "loaded_proposal_count": int(len(out)),
            "oldest_loaded_proposed_at": oldest_loaded_ts.isoformat() if oldest_loaded_ts is not None else None,
            "fetched_pages": fetched_pages,
            "order": "desc",
        }

    def identifications_for_observations(self, obs_ids: List[int]) -> List[Dict]:
        out = []
        # identifications supports comma-separated observation_id
        for batch in tqdm(list(chunks(obs_ids, 200)), desc="identifications(obs)"):
            j = self._get("identifications", {"observation_id": ",".join(map(str, batch)), "per_page": 200, "order": "asc"})
            out.extend(j.get("results", []))
            time.sleep(self.sleep)
        return out

    def observations_by_ids(self, obs_ids: List[int]) -> List[Dict]:
        out = []
        for batch in tqdm(list(chunks(obs_ids, 200)), desc="observations"):
            j = self._get("observations", {"id": ",".join(map(str, batch)), "per_page": 200})
            out.extend(j.get("results", []))
            time.sleep(self.sleep)
        return out


# -------------------------
# Data structures
# -------------------------

@dataclasses.dataclass
class Taxon:
    id: int
    name: str
    rank: str
    ancestor_ids: Tuple[int, ...]

    @staticmethod
    def from_api(j: Dict) -> "Taxon":
        return Taxon(
            id=j.get("id"),
            name=j.get("name"),
            rank=(j.get("rank") or ""),
            ancestor_ids=tuple(j.get("ancestor_ids") or ()),
        )


@dataclasses.dataclass
class Proposal:
    obs_id: int
    proposer_user_id: int
    proposer_login: str
    proposal_index: int  # 1,2,3... within this observation
    proposed_at: str
    taxon_id: int
    taxon_name: str
    taxon_rank: str
    disagree: Optional[bool]
    withdrawn: bool
    # Filled after timeline scan
    confirmed_by_others: bool = False
    confirming_user_ids: Tuple[int, ...] = dataclasses.field(default_factory=tuple)
    time_to_confirm_days: Optional[float] = None
    # Observation context
    observed_on: Optional[str] = None
    latitude: Optional[float] = None
    longitude: Optional[float] = None
    place_ids: Tuple[int, ...] = dataclasses.field(default_factory=tuple)
    final_ct_id: Optional[int] = None
    final_ct_name: Optional[str] = None
    final_ct_rank: Optional[str] = None
    # Derived
    status: str = "unknown"  # vindicated | undecided_support | overruled | withdrawn | unknown
    correctness_depth: str = "no_ct"  # species | genus | family | higher | wrong | no_ct


# -------------------------
# Taxon utils
# -------------------------

RANK_ORDER = [
    "stateofmatter", "kingdom", "phylum", "subphylum", "class", "subclass", "infraclass",
    "superorder", "order", "suborder", "infraorder", "parvorder",
    "superfamily", "epifamily", "family", "subfamily", "tribe", "subtribe",
    "genus", "subgenus", "section", "species", "subspecies", "variety", "form"
]
RANK_INDEX = {r: i for i, r in enumerate(RANK_ORDER)}


def lineage_ids(t: Optional[Taxon]) -> Set[int]:
    if not t:
        return set()
    s = set(t.ancestor_ids)
    s.add(t.id)
    return s


def common_rank_depth(a: Taxon, b: Taxon) -> str:
    """Return depth of agreement by clade: species/genus/family/higher/wrong."""
    la, lb = set(a.ancestor_ids) | {a.id}, set(b.ancestor_ids) | {b.id}
    if a.id == b.id:
        # exact match: assess by rank
        r = a.rank or b.rank or "species"
        if r in ("species", "subspecies", "variety", "form"):
            return "species"
        if r in ("genus", "subgenus", "section"):
            return "genus"
    if la & lb:
        # find deepest shared rank using a simple ladder
        # (coarse; relies on name ranks not being missing)
        shared = la & lb
        # Heuristically pick a family/genus break using taxon ranks not names
        # We need taxa for both; assume we will call this with cached taxa that include ranks
        # We approximate: if they share any genus-level ancestor -> genus; else if share family -> family; else higher
        # This requires a map id->Taxon to check ranks; will be injected at runtime
        return "higher"  # Placeholder; replaced in Analyzer where we have id->Taxon
    return "wrong"


# -------------------------
# Analyzer
# -------------------------

class Analyzer:
    """Coordinator for cache-backed reliability analytics.

    Design intent
    - Keep ingest and analysis decoupled to support notebook iteration.
    - Treat cached parquet files as the source of truth for offline rebuilds.
    - Return DataFrames from key operations so callers can build custom views.

    Expected cache artifacts
    - my_identifications.parquet
    - all_identifications.parquet
    - observations.parquet
    - taxa.parquet
    - proposals.parquet (optional derived artifact)
    """

    def __init__(self, cache_dir: str):
        self.cache_dir = cache_dir
        os.makedirs(cache_dir, exist_ok=True)
        self.client = INatClient()
        self._taxa_cache: Dict[int, Taxon] = {}

    # ---------- Cache I/O ----------
    def _path(self, name: str) -> str:
        return os.path.join(self.cache_dir, name)

    def _missing_cache_files(self) -> List[str]:
        """Return required ingest cache files that are currently missing."""
        return [name for name in REQUIRED_CACHE_FILES if not os.path.exists(self._path(name))]

    def _ensure_cache_ready(self, user_login: Optional[str] = None):
        """Validate that ingest cache exists before running offline analytics."""
        missing = self._missing_cache_files()
        if not missing:
            return
        missing_lines = "\n - ".join([""] + missing)
        who = f" for user '{user_login}'" if user_login else ""
        raise FileNotFoundError(
            "Missing cached ingest files in cache_dir"
            f"{who}:{missing_lines}\n\n"
            "Run Analyzer.ingest(user_login) first, or call Analyzer.run(..., ingest=True)."
        )

    def save_parquet(self, df: pd.DataFrame, name: str):
        try:
            df.to_parquet(self._path(name), index=False)
        except Exception as e:
            msg = str(e)
            if "pyarrow" in msg or "fastparquet" in msg or "Unable to find a usable engine" in msg:
                raise RuntimeError(
                    "Parquet support is required but no engine was found. "
                    "Install pyarrow (recommended) or fastparquet, then retry."
                ) from e
            raise

    def load_parquet(self, name: str) -> pd.DataFrame:
        try:
            return pd.read_parquet(self._path(name))
        except Exception as e:
            msg = str(e)
            if "pyarrow" in msg or "fastparquet" in msg or "Unable to find a usable engine" in msg:
                raise RuntimeError(
                    "Parquet support is required but no engine was found. "
                    "Install pyarrow (recommended) or fastparquet, then retry."
                ) from e
            raise

    @staticmethod
    def _observation_updated_map(df_obs: pd.DataFrame) -> Dict[int, str]:
        out: Dict[int, str] = {}
        if df_obs.empty or not {"id", "updated_at"}.issubset(df_obs.columns):
            return out
        for _, r in df_obs[["id", "updated_at"]].dropna(subset=["id"]).iterrows():
            out[int(r["id"])] = str(r.get("updated_at"))
        return out

    def ingest_progress_report(self, user_login: str, mode: str = "incremental") -> Dict[str, Any]:
        """Return a dry-run report describing expected ingest work.

        This method performs read-only API/cache checks and does not write any
        cache artifacts. It is intended for notebook/CLI progress previews.
        """
        if mode not in {"incremental", "full"}:
            raise ValueError("ingest mode must be one of: incremental, full")

        missing = self._missing_cache_files()
        effective_mode = mode
        fallback_to_full = False
        if mode == "incremental" and missing:
            effective_mode = "full"
            fallback_to_full = True

        report: Dict[str, Any] = {
            "user_login": user_login,
            "requested_mode": mode,
            "effective_mode": effective_mode,
            "fallback_to_full": fallback_to_full,
            "missing_cache_files": missing,
            "cache_dir": self.cache_dir,
            "api_batches": {},
        }

        my_ids = self.client.user_identifications(user_login)
        report["my_identifications_count"] = len(my_ids)
        if not my_ids:
            report["message"] = "No identifications found for user."
            return report

        df_my = pd.json_normalize(my_ids)
        obs_ids = sorted(df_my["observation.id"].dropna().astype(int).unique().tolist())
        report["current_observation_count"] = len(obs_ids)
        report["api_batches"]["user_identifications_pages"] = int(math.ceil(len(my_ids) / self.client.per_page))

        if effective_mode == "full":
            report.update(
                {
                    "new_observation_count": len(obs_ids),
                    "retained_observation_count": 0,
                    "removed_observation_count": 0,
                    "changed_observation_count": len(obs_ids),
                    "timelines_to_refresh_count": len(obs_ids),
                }
            )
            report["api_batches"]["observations_by_ids_batches"] = int(math.ceil(len(obs_ids) / 200))
            report["api_batches"]["identifications_by_observation_batches"] = int(math.ceil(len(obs_ids) / 200))
            return report

        df_obs_existing = self.load_parquet("observations.parquet")
        existing_obs_ids: Set[int] = set()
        if not df_obs_existing.empty and "id" in df_obs_existing.columns:
            existing_obs_ids = set(df_obs_existing["id"].dropna().astype(int).tolist())

        current_obs_ids = set(obs_ids)
        new_obs_ids = current_obs_ids - existing_obs_ids
        removed_obs_ids = existing_obs_ids - current_obs_ids
        retained_obs_ids = current_obs_ids & existing_obs_ids

        obs = self.client.observations_by_ids(obs_ids)
        df_obs = pd.json_normalize(obs)
        old_updated = self._observation_updated_map(df_obs_existing)
        new_updated = self._observation_updated_map(df_obs)

        changed_obs_ids = {
            oid for oid in current_obs_ids
            if oid not in old_updated or oid not in new_updated or old_updated[oid] != new_updated[oid]
        }
        refresh_obs_ids = new_obs_ids | changed_obs_ids

        report.update(
            {
                "new_observation_count": len(new_obs_ids),
                "retained_observation_count": len(retained_obs_ids),
                "removed_observation_count": len(removed_obs_ids),
                "changed_observation_count": len(changed_obs_ids),
                "timelines_to_refresh_count": len(refresh_obs_ids),
            }
        )
        report["api_batches"]["observations_by_ids_batches"] = int(math.ceil(len(obs_ids) / 200))
        report["api_batches"]["identifications_by_observation_batches"] = int(math.ceil(len(refresh_obs_ids) / 200))
        return report

    # ---------- Ingest ----------
    def ingest(self, user_login: str, mode: str = "incremental"):
        """Fetch and persist all raw inputs needed for offline analysis.

        This method performs network I/O. It intentionally stores denormalized,
        mostly raw tables from the iNaturalist API so proposal logic can be
        revised later without another download.

        Modes:
        - incremental (default): first-run fallback to full; otherwise refresh
          user IDs and only refetch timelines for observations that are new or
          whose observation ``updated_at`` changed.
        - full: refetch all observation timelines for the user's current
          observation footprint.
        """
        if mode not in {"incremental", "full"}:
            raise ValueError("ingest mode must be one of: incremental, full")

        missing = self._missing_cache_files()
        if mode == "incremental" and missing:
            missing_txt = ", ".join(missing)
            log(f"Incremental ingest requested but cache is incomplete ({missing_txt}); falling back to full ingest.")
            mode = "full"

        log(f"Ingest mode: {mode}")
        log(f"Fetching identifications for user {user_login}…")
        my_ids = self.client.user_identifications(user_login)
        if not my_ids:
            log("No identifications found.")
            return
        df_my = pd.json_normalize(my_ids)
        # Extract observation ids
        obs_ids = sorted(df_my["observation.id"].dropna().astype(int).unique().tolist())
        log(f"Unique observations with your IDs: {len(obs_ids)}")

        if mode == "full":
            log("Fetching ALL identifications on those observations…")
            all_ids = self.client.identifications_for_observations(obs_ids)
            df_all_ids = pd.json_normalize(all_ids)

            log("Fetching observation shells…")
            obs = self.client.observations_by_ids(obs_ids)
            df_obs = pd.json_normalize(obs)
            existing_taxa: Dict[int, Taxon] = {}
        else:
            df_all_existing = self.load_parquet("all_identifications.parquet")
            df_obs_existing = self.load_parquet("observations.parquet")
            df_taxa_existing = self.load_parquet("taxa.parquet")

            current_obs_ids = set(obs_ids)
            existing_obs_ids: Set[int] = set()
            if not df_obs_existing.empty and "id" in df_obs_existing.columns:
                existing_obs_ids = set(df_obs_existing["id"].dropna().astype(int).tolist())

            new_obs_ids = sorted(current_obs_ids - existing_obs_ids)
            removed_obs_ids = sorted(existing_obs_ids - current_obs_ids)
            log(
                "Incremental observation footprint: "
                f"new={len(new_obs_ids)}, retained={len(current_obs_ids) - len(new_obs_ids)}, removed={len(removed_obs_ids)}"
            )

            log("Refreshing observation shells to detect timeline changes…")
            obs = self.client.observations_by_ids(obs_ids)
            df_obs = pd.json_normalize(obs)

            old_updated: Dict[int, str] = {}
            if not df_obs_existing.empty and {"id", "updated_at"}.issubset(df_obs_existing.columns):
                for _, r in df_obs_existing[["id", "updated_at"]].dropna(subset=["id"]).iterrows():
                    old_updated[int(r["id"])] = str(r.get("updated_at"))

            new_updated: Dict[int, str] = {}
            if not df_obs.empty and {"id", "updated_at"}.issubset(df_obs.columns):
                for _, r in df_obs[["id", "updated_at"]].dropna(subset=["id"]).iterrows():
                    new_updated[int(r["id"])] = str(r.get("updated_at"))

            changed_obs_ids = [
                oid for oid in obs_ids
                if oid not in old_updated or oid not in new_updated or old_updated[oid] != new_updated[oid]
            ]
            refresh_obs_ids = sorted(set(new_obs_ids) | set(changed_obs_ids))
            log(f"Observation timelines needing refresh: {len(refresh_obs_ids)}")

            if refresh_obs_ids:
                log("Fetching refreshed identifications for changed/new observations…")
                all_ids_refreshed = self.client.identifications_for_observations(refresh_obs_ids)
                df_all_refreshed = pd.json_normalize(all_ids_refreshed)
            else:
                df_all_refreshed = pd.DataFrame(columns=df_all_existing.columns)

            keep_obs_ids = set(obs_ids) - set(refresh_obs_ids)
            if not df_all_existing.empty and "observation.id" in df_all_existing.columns:
                df_all_base = df_all_existing[df_all_existing["observation.id"].isin(keep_obs_ids)].copy()
            else:
                df_all_base = pd.DataFrame()
            df_all_ids = pd.concat([df_all_base, df_all_refreshed], ignore_index=True)

            existing_taxa = {}
            if not df_taxa_existing.empty and {"id", "name", "rank", "ancestor_ids"}.issubset(df_taxa_existing.columns):
                for _, r in df_taxa_existing.iterrows():
                    tid = int(r["id"])
                    ancestors = r["ancestor_ids"]
                    if not isinstance(ancestors, (list, tuple)):
                        ancestors = []
                    existing_taxa[tid] = Taxon(
                        id=tid,
                        name=str(r.get("name") or ""),
                        rank=str(r.get("rank") or ""),
                        ancestor_ids=tuple(int(x) for x in ancestors if pd.notna(x)),
                    )

        # Taxa cache: gather all taxon ids appearing in IDs and community taxa
        taxon_ids: Set[int] = set(
            x for x in pd.concat([
                df_all_ids.get("taxon.id"), df_obs.get("community_taxon.id")
            ], axis=0).dropna().astype(int).unique().tolist()
        )
        missing_taxon_ids = sorted(taxon_ids - set(existing_taxa.keys()))
        if missing_taxon_ids:
            log(f"Fetching {len(missing_taxon_ids)} new taxa…")
            taxa = self.client.taxa_by_ids(missing_taxon_ids)
            fetched_taxa = {t["id"]: Taxon.from_api(t) for t in taxa}
        else:
            fetched_taxa = {}
        taxa_map = {tid: existing_taxa[tid] for tid in taxon_ids if tid in existing_taxa}
        taxa_map.update(fetched_taxa)
        self._taxa_cache = dict(taxa_map)

        # Save raw caches
        self.save_parquet(df_my, "my_identifications.parquet")
        self.save_parquet(df_all_ids, "all_identifications.parquet")
        self.save_parquet(df_obs, "observations.parquet")
        # Save taxa as a small table
        df_taxa = pd.DataFrame([
            {"id": t.id, "name": t.name, "rank": t.rank, "ancestor_ids": list(t.ancestor_ids)}
            for t in taxa_map.values()
        ])
        self.save_parquet(df_taxa, "taxa.parquet")
        log("Ingest complete.")

    # ---------- Proposal building & timeline scan ----------
    def _load_taxa(self) -> Dict[int, Taxon]:
        """Load taxa from cache into memory when available."""
        if self._taxa_cache:
            return self._taxa_cache
        path = self._path("taxa.parquet")
        if not os.path.exists(path):
            return {}
        df = pd.read_parquet(path)
        cache = {}
        for _, r in df.iterrows():
            cache[int(r["id"])] = Taxon(id=int(r["id"]), name=r["name"], rank=r["rank"], ancestor_ids=tuple(r["ancestor_ids"]))
        self._taxa_cache = cache
        return cache

    def _taxon(self, taxon_id: Optional[int]) -> Optional[Taxon]:
        """Return a cached Taxon by id, or None when missing."""
        if taxon_id is None:
            return None
        return self._load_taxa().get(int(taxon_id))

    def _rank_id(self, t: Taxon, target: str) -> Optional[int]:
        """Return the nearest ancestor id (or self) that matches a rank."""
        if t.rank == target:
            return t.id
        for aid in reversed(t.ancestor_ids):
            at = self._taxa_cache.get(aid)
            if at and at.rank == target:
                return at.id
        return None

    def _deepest_shared_rank(self, a: Optional[Taxon], b: Optional[Taxon]) -> str:
        """Return deepest shared rank label between two taxa or 'none'."""
        if not a or not b:
            return "none"
        a_line = set(a.ancestor_ids) | {a.id}
        b_line = set(b.ancestor_ids) | {b.id}
        shared = a_line & b_line
        if not shared:
            return "none"
        best_rank = "none"
        best_idx = -1
        for tid in shared:
            t = self._taxa_cache.get(tid)
            if not t:
                continue
            idx = RANK_INDEX.get(t.rank, -1)
            if idx > best_idx:
                best_idx = idx
                best_rank = t.rank
        return best_rank

    def _depth_vs_ct(self, prop_taxon: Optional[Taxon], ct_taxon: Optional[Taxon]) -> str:
        """Classify clade agreement depth vs final community taxon.

        Returns one of: no_ct, wrong, species, genus, family, higher.
        """
        if not ct_taxon:
            return "no_ct"
        if not prop_taxon:
            return "wrong"
        # Build lineage sets
        la = set(prop_taxon.ancestor_ids) | {prop_taxon.id}
        lb = set(ct_taxon.ancestor_ids) | {ct_taxon.id}
        if prop_taxon.id == ct_taxon.id or ct_taxon.id in la or prop_taxon.id in lb:
            # exact or ancestor/descendant within species boundary ⇒ species
            # (descendant allows subspecies/forms)
            # If either rank is species-ish, treat as species
            ranks_species = {"species", "subspecies", "variety", "form"}
            if (prop_taxon.rank in ranks_species) or (ct_taxon.rank in ranks_species):
                return "species"
        # genus
        # find genus ancestor for each and compare
        g1 = self._rank_id(prop_taxon, "genus")
        g2 = self._rank_id(ct_taxon, "genus")
        if g1 and g2 and g1 == g2:
            return "genus"
        # family
        f1 = self._rank_id(prop_taxon, "family")
        f2 = self._rank_id(ct_taxon, "family")
        if f1 and f2 and f1 == f2:
            return "family"
        # higher shared?
        if la & lb:
            return "higher"
        return "wrong"

    def _matches_confirmation_boundary(self, prop_taxon: Taxon, other_taxon: Taxon, disagree: Optional[bool]) -> bool:
        """Evaluate whether another user's later ID confirms a proposal.

        Boundary rules are rank-aware and slightly stricter for disagreements.
        This intentionally encodes policy in one place so it can be tuned
        without touching timeline assembly code.

                Decision table (conceptual)
                - proposal rank: species-ish (species/subspecies/variety/form)
                    confirmation when: exact match OR ancestor/descendant relation
                - proposal rank: genus
                    disagree=False: any taxon that resolves to same genus
                    disagree=True: must be explicit genus-level ID for same genus
                - proposal rank: family
                    disagree=False: any taxon resolving to same family
                    disagree=True: must be explicit family-level ID for same family
                - proposal rank: higher than family or unusual rank
                    disagree=False: any lineage overlap
                    disagree=True: same rank AND lineage overlap

                Why this shape exists
                The rules attempt to preserve two practical properties:
                1) Non-disagreeing IDs can confirm more specific downstream agreement.
                2) Disagreeing IDs require stricter same-rank evidence to avoid
                     over-counting broad or ambiguous agreement as true confirmation.
        """
        # Species-level proposal ⇒ need same species or descendant (e.g., ssp.)
        speciesish = {"species", "subspecies", "variety", "form"}
        if prop_taxon.rank in speciesish:
            return (other_taxon.id == prop_taxon.id) or (other_taxon.id in set(prop_taxon.ancestor_ids)) or (prop_taxon.id in set(other_taxon.ancestor_ids))
        # Genus-level proposal ⇒
        # - if disagreeing: require genus match exactly
        # - if non-disagreeing: allow any descendant within genus
        if prop_taxon.rank == "genus":
            g_prop = self._rank_id(prop_taxon, "genus")
            g_other = self._rank_id(other_taxon, "genus")
            if not g_prop or not g_other:
                return False
            if disagree:
                return g_other == g_prop and other_taxon.rank == "genus"
            else:
                return g_other == g_prop
        # Family-level and above: require same family (or descendant if non-disagreeing)
        if prop_taxon.rank == "family":
            f_prop = self._rank_id(prop_taxon, "family")
            f_other = self._rank_id(other_taxon, "family")
            return bool(f_prop and f_other and f_prop == f_other and (not disagree or other_taxon.rank == "family"))
        # Otherwise, treat as higher-only: require overlap within lineage; if disagreeing, require same-rank match
        la = set(prop_taxon.ancestor_ids) | {prop_taxon.id}
        lb = set(other_taxon.ancestor_ids) | {other_taxon.id}
        if disagree:
            return prop_taxon.rank == other_taxon.rank and bool(la & lb)
        return bool(la & lb)

    def _build_proposals_from_frames(
        self,
        user_login: str,
        df_all: pd.DataFrame,
        df_obs: pd.DataFrame,
        start: Optional[str] = None,
        end: Optional[str] = None,
        place_ids: Optional[List[int]] = None,
        bbox: Optional[Tuple[float, float, float, float]] = None,
    ) -> pd.DataFrame:
        """Build proposal rows from in-memory observation and timeline tables.

        Date/time handling in this method is normalized to UTC.
        """
        if df_all.empty or df_obs.empty:
            return pd.DataFrame()
        if "id" not in df_obs.columns or "observation.id" not in df_all.columns:
            return pd.DataFrame()

        obs_meta = df_obs.set_index("id")
        mask_obs = pd.Series(True, index=obs_meta.index)

        if place_ids:
            place_set = set(place_ids)

            def has_place(x):
                try:
                    return bool(set(x) & place_set)
                except Exception:
                    return False

            m = obs_meta.get("place_ids").apply(has_place)
            m = m.fillna(False)
            mask_obs &= m

        if bbox:
            minlon, minlat, maxlon, maxlat = bbox
            lat = obs_meta.get("geojson.coordinates").apply(
                lambda v: v[1] if isinstance(v, list) and len(v) >= 2 else None
            )
            lon = obs_meta.get("geojson.coordinates").apply(
                lambda v: v[0] if isinstance(v, list) and len(v) >= 2 else None
            )
            mask_bbox = (lat >= minlat) & (lat <= maxlat) & (lon >= minlon) & (lon <= maxlon)
            mask_obs &= mask_bbox.fillna(False)

        if start or end:
            dates = pd.to_datetime(obs_meta.get("observed_on"), utc=True, errors="coerce")
            fallback = pd.to_datetime(obs_meta.get("created_at"), utc=True, errors="coerce")
            when = dates.fillna(fallback)
            if start:
                start_utc = pd.to_datetime(start, utc=True, errors="coerce")
                if pd.notna(start_utc):
                    mask_obs &= when >= start_utc.normalize()
            if end:
                end_utc = pd.to_datetime(end, utc=True, errors="coerce")
                if pd.notna(end_utc):
                    end_exclusive = end_utc.normalize() + pd.Timedelta(days=1)
                    mask_obs &= when < end_exclusive

        obs_keep = set(obs_meta.index[mask_obs])
        df_all = df_all[df_all["observation.id"].isin(obs_keep)].copy()
        if df_all.empty:
            return pd.DataFrame()

        df_all["_created_at_utc"] = pd.to_datetime(df_all.get("created_at"), utc=True, errors="coerce")
        df_all = df_all[df_all["_created_at_utc"].notna()].copy()
        df_all.sort_values(["observation.id", "_created_at_utc"], inplace=True)
        proposals: List[Proposal] = []

        for obs_id, g in tqdm(df_all.groupby("observation.id"), desc="proposals(obs)"):
            g = g.reset_index(drop=True)

            om = obs_meta.loc[obs_id] if obs_id in obs_meta.index else None
            observed_on = om.get("observed_on") if om is not None else None
            coords = om.get("geojson.coordinates") if om is not None else None
            lat = coords[1] if isinstance(coords, list) and len(coords) >= 2 else None
            lon = coords[0] if isinstance(coords, list) and len(coords) >= 2 else None
            places = tuple(om.get("place_ids") or ()) if om is not None else tuple()
            ct_id = om.get("community_taxon.id") if om is not None else None
            ct_name = om.get("community_taxon.name") if om is not None else None
            ct_rank = om.get("community_taxon.rank") if om is not None else None

            mine = g[g["user.login"] == user_login]
            if mine.empty:
                continue
            mine = mine.sort_values("_created_at_utc")

            current_taxon = None
            prop_idx = 0
            for _, r in mine.iterrows():
                tid = r.get("taxon.id")
                if pd.isna(tid):
                    continue
                tid = int(tid)
                if current_taxon == tid:
                    continue

                prop_idx += 1
                current_taxon = tid
                taxon_name = r.get("taxon.name")
                if not isinstance(taxon_name, str) or not taxon_name.strip():
                    taxon_obj = self._taxon(tid)
                    taxon_name = taxon_obj.name if taxon_obj else ""
                prop = Proposal(
                    obs_id=int(obs_id),
                    proposer_user_id=int(r.get("user.id")) if not pd.isna(r.get("user.id")) else -1,
                    proposer_login=user_login,
                    proposal_index=prop_idx,
                    proposed_at=str(r.get("_created_at_utc")),
                    taxon_id=tid,
                    taxon_name=taxon_name,
                    taxon_rank=r.get("taxon.rank") or "",
                    disagree=bool(r.get("disagreement")) if "disagreement" in r else None,
                    withdrawn=not bool(r.get("current")) if "current" in r else False,
                    observed_on=str(observed_on) if observed_on is not None else None,
                    latitude=lat,
                    longitude=lon,
                    place_ids=places,
                    final_ct_id=int(ct_id) if pd.notna(ct_id) else None,
                    final_ct_name=ct_name if isinstance(ct_name, str) else None,
                    final_ct_rank=ct_rank if isinstance(ct_rank, str) else None,
                )
                proposals.append(prop)

            for prop in proposals[-prop_idx:]:
                prop_time = pd.to_datetime(prop.proposed_at, utc=True, errors="coerce")
                if pd.isna(prop_time):
                    continue
                prop_tax = self._taxon(prop.taxon_id)
                if not prop_tax:
                    continue
                confirming_users: Set[int] = set()
                first_confirm_time = None
                for _, r in g.iterrows():
                    row_time = r.get("_created_at_utc")
                    if pd.isna(row_time) or row_time <= prop_time:
                        continue
                    if r.get("user.login") == user_login:
                        continue
                    other_tid = r.get("taxon.id")
                    if pd.isna(other_tid):
                        continue
                    other_tax = self._taxon(int(other_tid))
                    if not other_tax:
                        continue
                    if self._matches_confirmation_boundary(prop_tax, other_tax, prop.disagree):
                        uid = int(r.get("user.id")) if pd.notna(r.get("user.id")) else None
                        if uid:
                            confirming_users.add(uid)
                        if first_confirm_time is None:
                            first_confirm_time = row_time
                if confirming_users:
                    prop.confirmed_by_others = True
                    prop.confirming_user_ids = tuple(sorted(confirming_users))
                    try:
                        t0 = pd.to_datetime(prop_time, utc=True, errors="coerce")
                        t1 = pd.to_datetime(first_confirm_time, utc=True, errors="coerce")
                        if pd.isna(t0) or pd.isna(t1):
                            raise ValueError("invalid timestamp")
                        prop.time_to_confirm_days = float((t1 - t0).total_seconds()) / 86400.0
                    except Exception:
                        prop.time_to_confirm_days = None

            for prop in proposals[-prop_idx:]:
                ct_tax = self._taxon(prop.final_ct_id) if prop.final_ct_id else None
                my_tax = self._taxon(prop.taxon_id)
                prop.correctness_depth = self._depth_vs_ct(my_tax, ct_tax)
                if prop.confirmed_by_others and prop.correctness_depth != "wrong" and prop.correctness_depth != "no_ct":
                    prop.status = "vindicated"
                elif prop.confirmed_by_others and prop.correctness_depth == "no_ct":
                    prop.status = "undecided_support"
                else:
                    la = set(my_tax.ancestor_ids) | {my_tax.id} if my_tax else set()
                    lb = set(ct_tax.ancestor_ids) | {ct_tax.id} if ct_tax else set()
                    if ct_tax and not (la & lb):
                        prop.status = "overruled"
                    elif prop.withdrawn and not prop.confirmed_by_others:
                        prop.status = "withdrawn"
                    else:
                        prop.status = "unknown"

        return pd.DataFrame(dataclasses.asdict(p) for p in proposals)

    def build_proposals(self, user_login: str, start: Optional[str] = None, end: Optional[str] = None,
                        place_ids: Optional[List[int]] = None, bbox: Optional[Tuple[float,float,float,float]] = None,
                        save_outputs: bool = True) -> pd.DataFrame:
        """Build proposal events and per-proposal outcomes from local cache.

        Processing flow
        1. Load cached identification/observation/taxa tables.
        2. Apply observation-level filters (place, bbox, date window).
        3. Derive proposal boundaries from the target user's ID sequence.
        4. Scan later IDs by others for confirmations.
        5. Assign correctness depth and status for each proposal.

                Status assignment rules (explicit)
                Let C = confirmed_by_others, D = correctness_depth, W = withdrawn.

                - If C and D not in {"wrong", "no_ct"} => status = "vindicated"
                - Else if C and D == "no_ct" => status = "undecided_support"
                - Else if final CT exists and proposal lineage has no overlap with CT
                    lineage => status = "overruled"
                - Else if W and not C => status = "withdrawn"
                - Else => status = "unknown"

                Ordering is intentional and important: the ladder prioritizes positive
                confirmation outcomes before overrule/withdrawn fallbacks.

        Args:
            user_login: iNaturalist login to analyze.
            start: Optional inclusive date lower bound (YYYY-MM-DD).
            end: Optional inclusive date upper bound (YYYY-MM-DD).
            place_ids: Optional list of iNat place IDs; observation must match any.
            bbox: Optional (minlon, minlat, maxlon, maxlat) bounding box.
            save_outputs: When True, writes ``proposals.parquet`` in cache dir.

        Returns:
            DataFrame with one row per proposal.
        """
        self._ensure_cache_ready(user_login=user_login)
        df_all = self.load_parquet("all_identifications.parquet")
        df_obs = self.load_parquet("observations.parquet")
        self._load_taxa()

        df_props = self._build_proposals_from_frames(
            user_login=user_login,
            df_all=df_all,
            df_obs=df_obs,
            start=start,
            end=end,
            place_ids=place_ids,
            bbox=bbox,
        )
        if save_outputs:
            self.save_parquet(df_props, "proposals.parquet")
        return df_props

    def assess_taxon(
        self,
        user_login: str,
        taxon_id: int,
        start: Optional[str] = None,
        end: Optional[str] = None,
        place_ids: Optional[List[int]] = None,
        bbox: Optional[Tuple[float, float, float, float]] = None,
        include_withdrawn: bool = True,
        print_report: bool = False,
    ) -> Dict[str, pd.DataFrame]:
        """Assess one user+taxon reliability with minimal API footprint.

        Semantics for proposal outcomes in this mode:
        - confirmed: research-grade observation and at least one later
          confirmatory ID by another user.
        - disconfirmed: research-grade observation and final community taxon is
          different from the proposed taxon.
        - unresolved: all other proposals.

                Optional `place_ids` and `bbox` can be used for geographic subsets,
                but are not required for the lean default workflow.

                The assessment may proceed with partial input data if upstream paging
                encounters HTTP 403. In that case, warning metadata is returned.
        """
        if taxon_id is None:
            raise ValueError("taxon_id is required")

        target_taxon_id = int(taxon_id)
        log(
            f"[assess_taxon:scope] Fetching identifications for user '{user_login}' "
            f"within taxon {target_taxon_id} (target + descendants)."
        )
        my_ids, fetch_meta = self.client.user_identifications_windowed_best_effort(
            user=user_login,
            taxon_id=target_taxon_id,
            start=start,
            end=end,
            include_withdrawn=include_withdrawn,
        )
        fetch_meta = dict(fetch_meta)
        fetch_meta["target_taxon_id"] = target_taxon_id
        fetch_meta["target_scope"] = "target_plus_descendants"
        fetch_meta["include_withdrawn"] = bool(include_withdrawn)
        if fetch_meta.get("warning"):
            log(fetch_meta["warning"])
        if not my_ids:
            return {
                "proposals": pd.DataFrame(),
                "summary_by_species": pd.DataFrame(),
                "summary_by_rank": pd.DataFrame(),
                "taxon_reliability": pd.DataFrame(),
                "analysis_meta": fetch_meta,
            }

        df_my = pd.json_normalize(my_ids)
        if "taxon.id" not in df_my.columns or "observation.id" not in df_my.columns:
            return {
                "proposals": pd.DataFrame(),
                "summary_by_species": pd.DataFrame(),
                "summary_by_rank": pd.DataFrame(),
                "taxon_reliability": pd.DataFrame(),
                "analysis_meta": fetch_meta,
            }

        my_taxon = pd.to_numeric(df_my["taxon.id"], errors="coerce")
        my_taxon_ids = sorted(set(my_taxon.dropna().astype(int).tolist()))
        scope_taxa_ids = sorted(set(my_taxon_ids + [target_taxon_id]))
        scope_taxa = self.client.taxa_by_ids(scope_taxa_ids) if scope_taxa_ids else []
        scope_taxa_map: Dict[int, Taxon] = {t["id"]: Taxon.from_api(t) for t in scope_taxa if "id" in t}

        target_scope_taxon = scope_taxa_map.get(target_taxon_id)
        if target_scope_taxon:
            fetch_meta["target_taxon_name"] = target_scope_taxon.name
            fetch_meta["target_taxon_rank"] = target_scope_taxon.rank

            def in_target_lineage(raw_tid: Any) -> bool:
                if pd.isna(raw_tid):
                    return False
                tid = int(raw_tid)
                t = scope_taxa_map.get(tid)
                if not t:
                    # If taxonomy lookup is missing, trust API taxon scoping.
                    return True
                return tid == target_taxon_id or target_taxon_id in set(t.ancestor_ids)

            lineage_mask = my_taxon.map(in_target_lineage)
        else:
            # If target lineage lookup fails, avoid dropping API-scoped rows.
            lineage_mask = my_taxon.notna()

        df_my_taxon = df_my[lineage_mask.fillna(False)].copy()
        fetch_meta["candidate_identification_count"] = int(len(df_my))
        fetch_meta["lineage_filtered_identification_count"] = int(len(df_my_taxon))
        if df_my_taxon.empty:
            return {
                "proposals": pd.DataFrame(),
                "summary_by_species": pd.DataFrame(),
                "summary_by_rank": pd.DataFrame(),
                "taxon_reliability": pd.DataFrame(),
                "analysis_meta": fetch_meta,
            }

        obs_ids = sorted(df_my_taxon["observation.id"].dropna().astype(int).unique().tolist())
        fetch_meta["observation_scope_count"] = int(len(obs_ids))
        if not obs_ids:
            return {
                "proposals": pd.DataFrame(),
                "summary_by_species": pd.DataFrame(),
                "summary_by_rank": pd.DataFrame(),
                "taxon_reliability": pd.DataFrame(),
                "analysis_meta": fetch_meta,
            }

        log(
            f"[assess_taxon:scope] Retained {len(df_my_taxon)} proposals across {len(obs_ids)} "
            f"observations for target lineage."
        )
        log("[assess_taxon:timeline] Fetching full identification timelines for scoped observations.")
        all_ids = self.client.identifications_for_observations(obs_ids)
        obs = self.client.observations_by_ids(obs_ids)
        df_all = pd.json_normalize(all_ids)
        df_obs = pd.json_normalize(obs)

        taxon_series = []
        if "taxon.id" in df_all.columns:
            taxon_series.append(df_all["taxon.id"])
        if "community_taxon.id" in df_obs.columns:
            taxon_series.append(df_obs["community_taxon.id"])
        if taxon_series:
            all_taxon_ids = set(pd.concat(taxon_series, axis=0).dropna().astype(int).tolist())
        else:
            all_taxon_ids = set()
        all_taxon_ids.update(scope_taxa_map.keys())
        all_taxon_ids.add(target_taxon_id)

        taxa = self.client.taxa_by_ids(sorted(all_taxon_ids)) if all_taxon_ids else []
        self._taxa_cache = {t["id"]: Taxon.from_api(t) for t in taxa if "id" in t}
        target_taxon = self._taxon(target_taxon_id)
        if target_taxon:
            fetch_meta["target_taxon_name"] = target_taxon.name
            fetch_meta["target_taxon_rank"] = target_taxon.rank

        log("[assess_taxon:build] Building proposal outcomes from observation timelines.")
        df_props = self._build_proposals_from_frames(
            user_login=user_login,
            df_all=df_all,
            df_obs=df_obs,
            start=start,
            end=end,
            place_ids=place_ids,
            bbox=bbox,
        )
        log(f"[assess_taxon:build] Built {len(df_props)} proposals for downstream summaries.")
        sp, rk = self.summarize_from_proposals(df_props, csv=False, save_outputs=False, print_report=print_report)

        qg_map: Dict[int, str] = {}
        if not df_obs.empty and {"id", "quality_grade"}.issubset(df_obs.columns):
            for _, r in df_obs[["id", "quality_grade"]].dropna(subset=["id"]).iterrows():
                qg_map[int(r["id"])] = str(r.get("quality_grade") or "")

        if df_props.empty:
            df_taxon_rel = pd.DataFrame()
            fetch_meta["proposal_rows_generated"] = 0
            fetch_meta["proposal_observation_count"] = 0
            fetch_meta["unique_proposed_taxa_count"] = 0
        else:
            df_props = df_props.copy()
            df_props["is_research_grade"] = df_props["obs_id"].map(lambda oid: qg_map.get(int(oid), "") == "research")
            df_props["is_disconfirmed"] = (
                df_props["is_research_grade"]
                & df_props["final_ct_id"].notna()
                & (df_props["final_ct_id"].astype("Int64") != df_props["taxon_id"].astype("Int64"))
            )
            df_props["is_confirmed"] = df_props["is_research_grade"] & df_props["confirmed_by_others"].fillna(False)
            df_props["outcome"] = "unresolved"
            df_props.loc[df_props["is_confirmed"], "outcome"] = "confirmed"
            df_props.loc[df_props["is_disconfirmed"], "outcome"] = "disconfirmed"
            df_props["proposed_taxon"] = df_props["taxon_name"]
            df_props["proposed_taxon_id"] = pd.to_numeric(df_props["taxon_id"], errors="coerce").astype("Int64")
            df_props["community_taxon"] = df_props["final_ct_name"].fillna("")
            df_props["community_taxon_id"] = pd.to_numeric(df_props["final_ct_id"], errors="coerce").astype("Int64")
            df_props["confirmed_status"] = df_props["confirmed_by_others"].map(
                lambda v: "confirmed_after_proposal" if bool(v) else "no_confirming_id_seen"
            )

            target_lineage_ids = set(target_taxon.ancestor_ids) | {target_taxon.id} if target_taxon else set()

            def overlap_level_for_target_lineage(tid: Any) -> str:
                t = self._taxon(int(tid)) if pd.notna(tid) else None
                if not t or not target_lineage_ids:
                    return "none"
                shared = (set(t.ancestor_ids) | {t.id}) & target_lineage_ids
                if not shared:
                    return "none"
                best_rank = "none"
                best_idx = -1
                for sid in shared:
                    st = self._taxa_cache.get(sid)
                    if not st:
                        continue
                    idx = RANK_INDEX.get(st.rank, -1)
                    if idx > best_idx:
                        best_idx = idx
                        best_rank = st.rank
                return best_rank

            df_props["taxonomic_level"] = df_props["taxon_id"].map(overlap_level_for_target_lineage)

            grp = df_props.groupby("taxonomic_level", dropna=False)
            df_counts = grp.agg(
                n_props=("obs_id", "count"),
                confirmed_count=("outcome", lambda s: int((s == "confirmed").sum())),
                disconfirmed_count=("outcome", lambda s: int((s == "disconfirmed").sum())),
                unresolved_count=("outcome", lambda s: int((s == "unresolved").sum())),
            ).reset_index()

            target_levels: List[str] = []
            if target_taxon:
                rank_pairs: List[Tuple[int, str]] = []
                for tid in list(target_taxon.ancestor_ids) + [target_taxon.id]:
                    tt = self._taxa_cache.get(tid)
                    if not tt:
                        continue
                    rank_pairs.append((RANK_INDEX.get(tt.rank, -1), tt.rank))
                target_levels = [r for _, r in sorted(rank_pairs, key=lambda x: x[0]) if r]
                target_levels = list(dict.fromkeys(target_levels))

            if target_levels:
                df_levels = pd.DataFrame({"taxonomic_level": target_levels})
                df_taxon_rel = df_levels.merge(df_counts, on="taxonomic_level", how="left")
                for c in ["n_props", "confirmed_count", "disconfirmed_count", "unresolved_count"]:
                    df_taxon_rel[c] = df_taxon_rel[c].fillna(0).astype(int)
                extra = df_counts[~df_counts["taxonomic_level"].isin(target_levels)]
                if not extra.empty:
                    df_taxon_rel = pd.concat([df_taxon_rel, extra], ignore_index=True)
            else:
                df_taxon_rel = df_counts

            df_taxon_rel["confirmed_to_disconfirmed_ratio"] = df_taxon_rel["confirmed_count"] / df_taxon_rel[
                "disconfirmed_count"
            ].where(df_taxon_rel["disconfirmed_count"] > 0, 1)

            if target_taxon:
                rank_name_map: Dict[str, str] = {}
                rank_id_map: Dict[str, int] = {}
                for tid in list(target_taxon.ancestor_ids) + [target_taxon.id]:
                    tt = self._taxa_cache.get(tid)
                    if not tt or not tt.rank:
                        continue
                    rank_name_map[tt.rank] = tt.name
                    rank_id_map[tt.rank] = tt.id

                df_taxon_rel["target_taxon_name"] = df_taxon_rel["taxonomic_level"].map(rank_name_map).fillna("")
                df_taxon_rel["target_taxon_id"] = pd.to_numeric(
                    df_taxon_rel["taxonomic_level"].map(rank_id_map), errors="coerce"
                ).astype("Int64")
                df_taxon_rel["target_taxon_label"] = df_taxon_rel.apply(
                    lambda r: (
                        f"{r['target_taxon_name']} ({int(r['target_taxon_id'])})"
                        if r["target_taxon_name"] and pd.notna(r["target_taxon_id"])
                        else str(r["taxonomic_level"])
                    ),
                    axis=1,
                )
            else:
                df_taxon_rel["target_taxon_name"] = ""
                df_taxon_rel["target_taxon_id"] = pd.Series([pd.NA] * len(df_taxon_rel), dtype="Int64")
                df_taxon_rel["target_taxon_label"] = df_taxon_rel["taxonomic_level"]

            df_props["community_overlap_depth"] = df_props["correctness_depth"]
            fetch_meta["proposal_rows_generated"] = int(len(df_props))
            fetch_meta["proposal_observation_count"] = int(df_props["obs_id"].nunique())
            fetch_meta["unique_proposed_taxa_count"] = int(df_props["taxon_id"].nunique())

        return {
            "proposals": df_props,
            "summary_by_species": sp,
            "summary_by_rank": rk,
            "taxon_reliability": df_taxon_rel,
            "analysis_meta": fetch_meta,
        }

    # ---------- Summaries ----------
    def summarize_from_proposals(self, df_props: pd.DataFrame, csv: bool = False,
                                 save_outputs: bool = True, print_report: bool = True) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Aggregate proposal-level rows into species and rank summaries.

        This method is pure relative to proposal construction and can be used
        in notebooks to rerun aggregation after custom filtering on ``df_props``.
        """
        if df_props.empty:
            log("No proposals after filtering.")
            return pd.DataFrame(), pd.DataFrame()
        # Per-species (proposed)
        sp = (df_props[df_props["taxon_rank"] == "species"].groupby(["taxon_id", "taxon_name"]).agg(
            n_props=("obs_id", "count"),
            n_vindicated=("status", lambda s: (s == "vindicated").sum()),
            n_undecided=("status", lambda s: (s == "undecided_support").sum()),
            n_overruled=("status", lambda s: (s == "overruled").sum()),
            n_withdrawn=("status", lambda s: (s == "withdrawn").sum()),
            n_unknown=("status", lambda s: (s == "unknown").sum()),
            species_correct=("correctness_depth", lambda s: (s == "species").sum()),
            genus_only=("correctness_depth", lambda s: (s == "genus").sum()),
            family_only=("correctness_depth", lambda s: (s == "family").sum()),
            higher_only=("correctness_depth", lambda s: (s == "higher").sum()),
            wrong_clade=("correctness_depth", lambda s: (s == "wrong").sum()),
        ).reset_index())
        # Rates
        denom = pd.to_numeric(sp["n_props"], errors="coerce").where(sp["n_props"] > 0, 1).astype(float)
        for col in ["n_vindicated", "n_undecided", "n_overruled", "n_withdrawn", "n_unknown", "species_correct", "genus_only", "family_only", "higher_only", "wrong_clade"]:
            numer = pd.to_numeric(sp[col], errors="coerce").astype(float)
            sp[col + "_rate"] = numer.div(denom)

        # Per-rank reliability
        rk = (df_props.groupby(["taxon_rank"]).agg(
            n_props=("obs_id", "count"),
            vindicated_rate=("status", lambda s: (s == "vindicated").mean()),
            overruled_rate=("status", lambda s: (s == "overruled").mean()),
            withdrawn_rate=("status", lambda s: (s == "withdrawn").mean()),
            undecided_rate=("status", lambda s: (s == "undecided_support").mean()),
        ).reset_index())

        if save_outputs:
            out_dir = self.cache_dir
            sp_path = os.path.join(out_dir, "summary_by_species.parquet")
            rk_path = os.path.join(out_dir, "summary_by_rank.parquet")
            sp.to_parquet(sp_path, index=False)
            rk.to_parquet(rk_path, index=False)
            if csv:
                sp.to_csv(os.path.join(out_dir, "summary_by_species.csv"), index=False)
                rk.to_csv(os.path.join(out_dir, "summary_by_rank.csv"), index=False)
            log(f"Wrote {sp_path} and {rk_path}")

        if print_report:
            print("\nTop species by vindication rate (n>=3):")
            sp2 = sp[sp["n_props"] >= 3].sort_values(["n_vindicated_rate", "n_props"], ascending=[False, False]).head(20)
            with pd.option_context('display.max_rows', None, 'display.max_columns', None):
                print(sp2[['taxon_name','n_props','n_vindicated','n_overruled','n_withdrawn','n_undecided','n_unknown','n_vindicated_rate']])

        return sp, rk

    def summarize(self, user_login: str, start: Optional[str], end: Optional[str],
                  place_ids: Optional[List[int]], bbox: Optional[Tuple[float,float,float,float]],
                  csv: bool = False, save_outputs: bool = True,
                  print_report: bool = True) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Convenience wrapper: build proposals then aggregate summaries."""
        # Rebuild proposals with filters (from cache; no API)
        df_props = self.build_proposals(user_login, start, end, place_ids, bbox, save_outputs=save_outputs)
        sp, rk = self.summarize_from_proposals(df_props, csv=csv, save_outputs=save_outputs, print_report=print_report)
        return df_props, sp, rk

    def run(self, user_login: str, ingest: bool = False, ingest_mode: str = "incremental",
            start: Optional[str] = None, end: Optional[str] = None,
            place_ids: Optional[List[int]] = None, bbox: Optional[Tuple[float,float,float,float]] = None,
            csv: bool = False, save_outputs: bool = False,
            print_report: bool = False) -> Dict[str, pd.DataFrame]:
        """Notebook-first end-to-end entrypoint.

        Typical notebook pattern:
        - First run with ``ingest=True`` to seed cache.
        - Subsequent runs with ``ingest=False`` while iterating on filters.

        Args:
            user_login: iNaturalist login.
            ingest: If True, refresh cache from API before analysis.
            ingest_mode: One of incremental or full. Ignored when ingest=False.
            start/end/place_ids/bbox: Same semantics as ``build_proposals``.
            csv: Also emit CSV summaries when ``save_outputs`` is True.
            save_outputs: Persist derived parquet/csv outputs to cache dir.
            print_report: Print compact console summary table.

        Returns:
            Dict with ``proposals``, ``summary_by_species``, ``summary_by_rank``.
        """
        if ingest:
            self.ingest(user_login, mode=ingest_mode)
        df_props = self.build_proposals(user_login, start, end, place_ids, bbox, save_outputs=save_outputs)
        sp, rk = self.summarize_from_proposals(df_props, csv=csv, save_outputs=save_outputs, print_report=print_report)
        return {
            "proposals": df_props,
            "summary_by_species": sp,
            "summary_by_rank": rk,
        }


# -------------------------
# CLI
# -------------------------

def parse_bbox(s: str) -> Tuple[float,float,float,float]:
    """Parse --bbox CLI value ``minlon,minlat,maxlon,maxlat``."""
    parts = [float(x) for x in s.split(',')]
    if len(parts) != 4:
        raise argparse.ArgumentTypeError("--bbox must be minlon,minlat,maxlon,maxlat")
    return parts[0], parts[1], parts[2], parts[3]


def main():
    """CLI entrypoint for primary taxon assessment and deprecated cache workflows."""
    ap = argparse.ArgumentParser(description="iNat user+taxon reliability analyzer (assess-taxon is primary)")
    sub = ap.add_subparsers(dest="cmd", required=True)

    ap_ing = sub.add_parser("ingest", help="DEPRECATED: download and cache data for a user")
    ap_ing.add_argument("--user", required=True, help="iNat login (e.g., schizoform)")
    ap_ing.add_argument("--out", default=DEFAULT_CACHE_DIR, help="Cache/output directory")
    ap_ing.add_argument(
        "--mode",
        choices=["incremental", "full"],
        default="incremental",
        help="Ingest strategy: incremental (default, first-run falls back to full) or full",
    )

    ap_rep = sub.add_parser("ingest-report", help="DEPRECATED: dry-run ingest workload report")
    ap_rep.add_argument("--user", required=True, help="iNat login")
    ap_rep.add_argument("--out", default=DEFAULT_CACHE_DIR, help="Cache/output directory")
    ap_rep.add_argument(
        "--mode",
        choices=["incremental", "full"],
        default="incremental",
        help="Report strategy assumptions: incremental (default) or full",
    )

    ap_sum = sub.add_parser("summarize", help="DEPRECATED: build proposals and summaries from cache")
    ap_sum.add_argument("--user", required=True, help="iNat login")
    ap_sum.add_argument("--out", default=DEFAULT_CACHE_DIR, help="Cache/output directory")
    ap_sum.add_argument("--start", help="Start date (YYYY-MM-DD)")
    ap_sum.add_argument("--end", help="End date (YYYY-MM-DD)")
    ap_sum.add_argument("--place-id", help="Comma-separated iNat place_ids to include")
    ap_sum.add_argument("--bbox", type=parse_bbox, help="minlon,minlat,maxlon,maxlat")
    ap_sum.add_argument("--csv", action="store_true", help="Also write CSV outputs")

    ap_assess = sub.add_parser("assess-taxon", help="Primary lightweight assessment for one user and one taxon")
    ap_assess.add_argument("--user", required=True, help="iNat login")
    ap_assess.add_argument("--taxon-id", required=True, type=int, help="Target taxon id")
    ap_assess.add_argument("--out", default=DEFAULT_CACHE_DIR, help="Cache/output directory (unused for no-cache path)")
    ap_assess.add_argument("--start", help="Start date (YYYY-MM-DD)")
    ap_assess.add_argument("--end", help="End date (YYYY-MM-DD)")
    ap_assess.add_argument("--place-id", help="Comma-separated iNat place_ids to include")
    ap_assess.add_argument("--bbox", type=parse_bbox, help="minlon,minlat,maxlon,maxlat")

    args = ap.parse_args()

    analyzer = Analyzer(cache_dir=args.out)

    if args.cmd == "ingest":
        analyzer.ingest(args.user, mode=args.mode)
        return

    if args.cmd == "ingest-report":
        report = analyzer.ingest_progress_report(args.user, mode=args.mode)
        print(json.dumps(report, indent=2, sort_keys=True))
        return

    if args.cmd == "summarize":
        place_ids = None
        if args.place_id:
            place_ids = [int(x) for x in args.place_id.split(',') if x.strip()]
        bbox = args.bbox if args.bbox else None
        analyzer.summarize(args.user, args.start, args.end, place_ids, bbox, csv=args.csv)
        return

    if args.cmd == "assess-taxon":
        place_ids = None
        if args.place_id:
            place_ids = [int(x) for x in args.place_id.split(',') if x.strip()]
        bbox = args.bbox if args.bbox else None
        out = analyzer.assess_taxon(
            user_login=args.user,
            taxon_id=args.taxon_id,
            start=args.start,
            end=args.end,
            place_ids=place_ids,
            bbox=bbox,
            print_report=True,
        )
        print(
            json.dumps(
                {
                    "n_proposals": int(len(out["proposals"])),
                    "n_species_rows": int(len(out["summary_by_species"])),
                    "n_rank_rows": int(len(out["summary_by_rank"])),
                    "n_reliability_rows": int(len(out["taxon_reliability"])),
                },
                indent=2,
                sort_keys=True,
            )
        )
        return


if __name__ == "__main__":
    main()
