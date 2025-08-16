#!/usr/bin/env python3
"""
Compute proposal-level identification reliability for an iNaturalist user, with
post-hoc filtering by time and location and taxon-level summaries.

Design highlights
- One-time ingest/cache from iNat API v1: your identifications, all IDs on the
  same observations, observation shells (dates/coords/place_ids/CT), taxa.
- Rebuild metrics locally without re-downloading: time windows, place filters
  (place_id or bounding box), and rank-aware confirmation and correctness.
- Status ladder per *proposal* (a user's taxon change on an observation):
    Vindicated > Undecided (with support) > Overruled > Withdrawn (unsupported) > Unknown
- Correctness depth vs final community taxon: species / genus / family / higher / wrong / no_ct

CLI
  Ingest:     python inat_reliability.py ingest --user schizoform [--out data]
  Summarize:  python inat_reliability.py summarize --user schizoform \
                [--start 2022-01-01 --end 2025-08-11] \
                [--place-id 47,123] [--bbox minlon,minlat,maxlon,maxlat] \
                [--rank-threshold genus] [--out data] [--csv]

Requirements: Python 3.9+, requests, pandas; optional: pyarrow (for Parquet),
              tqdm (nicer progress). No internet calls during summarize.
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
from typing import Dict, Iterable, List, Optional, Set, Tuple

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
    def __init__(self, base=BASE, per_page=PER_PAGE, sleep=SLEEP):
        self.base = base
        self.per_page = per_page
        self.sleep = sleep
        self.session = requests.Session()
        self.session.headers.update({"User-Agent": "inat-reliability/0.1"})

    def _get(self, endpoint: str, params: Dict) -> Dict:
        url = f"{self.base}/{endpoint}"
        r = self.session.get(url, params=params, timeout=60)
        r.raise_for_status()
        return r.json()

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

    def user_identifications(self, user: str) -> List[Dict]:
        params = {
            "user_id": user,
            "current": "any",  # include withdrawn
            "order_by": "created_at",
            "order": "asc",
        }
        return list(self.paged("identifications", params))

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
    def __init__(self, cache_dir: str):
        self.cache_dir = cache_dir
        os.makedirs(cache_dir, exist_ok=True)
        self.client = INatClient()
        self._taxa_cache: Dict[int, Taxon] = {}

    # ---------- Cache I/O ----------
    def _path(self, name: str) -> str:
        return os.path.join(self.cache_dir, name)

    def save_parquet(self, df: pd.DataFrame, name: str):
        df.to_parquet(self._path(name), index=False)

    def load_parquet(self, name: str) -> pd.DataFrame:
        return pd.read_parquet(self._path(name))

    # ---------- Ingest ----------
    def ingest(self, user_login: str):
        log(f"Fetching identifications for user {user_login}…")
        my_ids = self.client.user_identifications(user_login)
        if not my_ids:
            log("No identifications found.")
            return
        df_my = pd.json_normalize(my_ids)
        # Extract observation ids
        obs_ids = sorted(df_my["observation.id"].dropna().astype(int).unique().tolist())
        log(f"Unique observations with your IDs: {len(obs_ids)}")

        log("Fetching ALL identifications on those observations…")
        all_ids = self.client.identifications_for_observations(obs_ids)
        df_all_ids = pd.json_normalize(all_ids)

        log("Fetching observation shells…")
        obs = self.client.observations_by_ids(obs_ids)
        df_obs = pd.json_normalize(obs)

        # Taxa cache: gather all taxon ids appearing in IDs and community taxa
        taxon_ids: Set[int] = set(
            x for x in pd.concat([
                df_all_ids.get("taxon.id"), df_obs.get("community_taxon.id")
            ], axis=0).dropna().astype(int).unique().tolist()
        )
        log(f"Fetching {len(taxon_ids)} taxa…")
        taxa = self.client.taxa_by_ids(sorted(taxon_ids))
        taxa_map = {t["id"]: Taxon.from_api(t) for t in taxa}
        self._taxa_cache.update(taxa_map)

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
        if taxon_id is None:
            return None
        return self._load_taxa().get(int(taxon_id))

    def _depth_vs_ct(self, prop_taxon: Optional[Taxon], ct_taxon: Optional[Taxon]) -> str:
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
        def first_rank(t: Taxon, target: str) -> Optional[int]:
            # Walk ancestors from top; pick deepest with that rank
            # We'll just check if any ancestor id has that rank via map
            for aid in reversed(t.ancestor_ids):
                at = self._taxa_cache.get(aid)
                if at and at.rank == target:
                    return at.id
            return None
        g1 = first_rank(prop_taxon, "genus")
        g2 = first_rank(ct_taxon, "genus")
        if g1 and g2 and g1 == g2:
            return "genus"
        # family
        f1 = first_rank(prop_taxon, "family")
        f2 = first_rank(ct_taxon, "family")
        if f1 and f2 and f1 == f2:
            return "family"
        # higher shared?
        if la & lb:
            return "higher"
        return "wrong"

    def _matches_confirmation_boundary(self, prop_taxon: Taxon, other_taxon: Taxon, disagree: Optional[bool]) -> bool:
        # Species-level proposal ⇒ need same species or descendant (e.g., ssp.)
        speciesish = {"species", "subspecies", "variety", "form"}
        if prop_taxon.rank in speciesish:
            return (other_taxon.id == prop_taxon.id) or (other_taxon.id in set(prop_taxon.ancestor_ids)) or (prop_taxon.id in set(other_taxon.ancestor_ids))
        # Genus-level proposal ⇒
        # - if disagreeing: require genus match exactly
        # - if non-disagreeing: allow any descendant within genus
        def genus_id(t: Taxon) -> Optional[int]:
            if t.rank == "genus":
                return t.id
            for aid in reversed(t.ancestor_ids):
                at = self._taxa_cache.get(aid)
                if at and at.rank == "genus":
                    return at.id
            return None
        if prop_taxon.rank == "genus":
            g_prop = genus_id(prop_taxon)
            g_other = genus_id(other_taxon)
            if not g_prop or not g_other:
                return False
            if disagree:
                return g_other == g_prop and other_taxon.rank == "genus"
            else:
                return g_other == g_prop
        # Family-level and above: require same family (or descendant if non-disagreeing)
        def rank_id(t: Taxon, target: str) -> Optional[int]:
            if t.rank == target:
                return t.id
            for aid in reversed(t.ancestor_ids):
                at = self._taxa_cache.get(aid)
                if at and at.rank == target:
                    return at.id
            return None
        if prop_taxon.rank == "family":
            f_prop = rank_id(prop_taxon, "family")
            f_other = rank_id(other_taxon, "family")
            return bool(f_prop and f_other and f_prop == f_other and (not disagree or other_taxon.rank == "family"))
        # Otherwise, treat as higher-only: require overlap within lineage; if disagreeing, require same-rank match
        la = set(prop_taxon.ancestor_ids) | {prop_taxon.id}
        lb = set(other_taxon.ancestor_ids) | {other_taxon.id}
        if disagree:
            return prop_taxon.rank == other_taxon.rank and bool(la & lb)
        return bool(la & lb)

    def build_proposals(self, user_login: str, start: Optional[str] = None, end: Optional[str] = None,
                        place_ids: Optional[List[int]] = None, bbox: Optional[Tuple[float,float,float,float]] = None,
                        save_outputs: bool = True) -> pd.DataFrame:
        df_my = self.load_parquet("my_identifications.parquet")
        df_all = self.load_parquet("all_identifications.parquet")
        df_obs = self.load_parquet("observations.parquet")
        self._load_taxa()

        # Index helpers
        obs_meta = df_obs.set_index("id")
        # Filter observations by place/bbox/time (based on observation fields)
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
            lat = obs_meta.get("geojson.coordinates").apply(lambda v: v[1] if isinstance(v, list) and len(v)>=2 else None)
            lon = obs_meta.get("geojson.coordinates").apply(lambda v: v[0] if isinstance(v, list) and len(v)>=2 else None)
            mask_bbox = (lat >= minlat) & (lat <= maxlat) & (lon >= minlon) & (lon <= maxlon)
            mask_obs &= mask_bbox.fillna(False)
        if start or end:
            # Use observed_on if present, else created_at
            dates = pd.to_datetime(obs_meta.get("observed_on"))
            fallback = pd.to_datetime(obs_meta.get("created_at"))
            when = dates.fillna(fallback)
            if start:
                mask_obs &= (when >= pd.to_datetime(start))
            if end:
                mask_obs &= (when <= pd.to_datetime(end))
        obs_keep = set(obs_meta.index[mask_obs])

        # Prepare IDs per observation sorted by time
        df_all = df_all[df_all["observation.id"].isin(obs_keep)].copy()
        df_all.sort_values(["observation.id", "created_at"], inplace=True)

        # Map login->id to tag confirmers
        user_id_by_login = {}
        if "user.login" in df_all.columns and "user.id" in df_all.columns:
            tmp = df_all[["user.login", "user.id"]].dropna().drop_duplicates()
            user_id_by_login = dict(zip(tmp["user.login"], tmp["user.id"]))

        proposals: List[Proposal] = []

        for obs_id, g in tqdm(df_all.groupby("observation.id"), desc="proposals(obs)"):
            g = g.reset_index(drop=True)
            # Observation context
            om = obs_meta.loc[obs_id] if obs_id in obs_meta.index else None
            observed_on = om.get("observed_on") if om is not None else None
            coords = om.get("geojson.coordinates") if om is not None else None
            lat = coords[1] if isinstance(coords, list) and len(coords) >= 2 else None
            lon = coords[0] if isinstance(coords, list) and len(coords) >= 2 else None
            places = tuple(om.get("place_ids") or ()) if om is not None else tuple()
            ct_id = om.get("community_taxon.id") if om is not None else None
            ct_name = om.get("community_taxon.name") if om is not None else None
            ct_rank = om.get("community_taxon.rank") if om is not None else None

            # Build your proposal list
            mine = g[g["user.login"] == user_login]
            if mine.empty:
                continue
            mine = mine.sort_values("created_at")
            current_taxon = None
            prop_idx = 0
            for _, r in mine.iterrows():
                tid = r.get("taxon.id")
                if pd.isna(tid):
                    continue
                tid = int(tid)
                if current_taxon == tid:
                    # Same taxon as your previous ID -> not a new proposal
                    continue
                prop_idx += 1
                current_taxon = tid
                prop = Proposal(
                    obs_id=int(obs_id),
                    proposer_user_id=int(r.get("user.id")) if not pd.isna(r.get("user.id")) else -1,
                    proposer_login=user_login,
                    proposal_index=prop_idx,
                    proposed_at=str(r.get("created_at")),
                    taxon_id=tid,
                    taxon_name=r.get("taxon.name") or "",
                    taxon_rank=r.get("taxon.rank") or "",
                    disagree=bool(r.get("disagreement")) if "disagreement" in r else None,
                    withdrawn=not bool(r.get("current")) if "current" in r else False,
                    observed_on=str(observed_on) if observed_on is not None else None,
                    latitude=lat, longitude=lon,
                    place_ids=places,
                    final_ct_id=int(ct_id) if pd.notna(ct_id) else None,
                    final_ct_name=ct_name if isinstance(ct_name, str) else None,
                    final_ct_rank=ct_rank if isinstance(ct_rank, str) else None,
                )
                proposals.append(prop)

            # Now scan confirmations for each proposal using the full timeline `g`
            # Only check IDs *after* the proposal time, by *other* users
            for prop in proposals[-prop_idx:]:
                prop_time = prop.proposed_at
                prop_tax = self._taxon(prop.taxon_id)
                if not prop_tax:
                    continue
                confirming_users: Set[int] = set()
                first_confirm_time = None
                for _, r in g.iterrows():
                    if r.get("created_at") <= prop_time:
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
                            first_confirm_time = r.get("created_at")
                if confirming_users:
                    prop.confirmed_by_others = True
                    prop.confirming_user_ids = tuple(sorted(confirming_users))
                    try:
                        t0 = pd.to_datetime(prop_time)
                        t1 = pd.to_datetime(first_confirm_time)
                        prop.time_to_confirm_days = float((t1 - t0).total_seconds()) / 86400.0
                    except Exception:
                        prop.time_to_confirm_days = None

            # Final: assign status and correctness
            for prop in proposals[-prop_idx:]:
                ct_tax = self._taxon(prop.final_ct_id) if prop.final_ct_id else None
                my_tax = self._taxon(prop.taxon_id)
                prop.correctness_depth = self._depth_vs_ct(my_tax, ct_tax)
                # Status ladder
                if prop.confirmed_by_others and prop.correctness_depth != "wrong" and prop.correctness_depth != "no_ct":
                    prop.status = "vindicated"
                elif prop.confirmed_by_others and (prop.correctness_depth == "no_ct"):
                    prop.status = "undecided_support"
                else:
                    # overruled if final CT exists and is outside proposal lineage
                    la = set(my_tax.ancestor_ids) | {my_tax.id} if my_tax else set()
                    lb = set(ct_tax.ancestor_ids) | {ct_tax.id} if ct_tax else set()
                    if ct_tax and not (la & lb):
                        prop.status = "overruled"
                    elif prop.withdrawn and not prop.confirmed_by_others:
                        prop.status = "withdrawn"
                    else:
                        prop.status = "unknown"

        # Convert to DataFrame
        df_props = pd.DataFrame(dataclasses.asdict(p) for p in proposals)
        if save_outputs:
            self.save_parquet(df_props, "proposals.parquet")
        return df_props

    # ---------- Summaries ----------
    def summarize(self, user_login: str, start: Optional[str], end: Optional[str], place_ids: Optional[List[int]], bbox: Optional[Tuple[float,float,float,float]], csv: bool = False):
        # Rebuild proposals with filters (from cache; no API)
        df_props = self.build_proposals(user_login, start, end, place_ids, bbox, save_outputs=False)
        if df_props.empty:
            log("No proposals after filtering.")
            return
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
        for col in ["n_vindicated", "n_undecided", "n_overruled", "n_withdrawn", "n_unknown", "species_correct", "genus_only", "family_only", "higher_only", "wrong_clade"]:
            sp[col + "_rate"] = sp[col] / sp["n_props"].where(sp["n_props"] > 0, 1)

        # Per-rank reliability
        rk = (df_props.groupby(["taxon_rank"]).agg(
            n_props=("obs_id", "count"),
            vindicated_rate=("status", lambda s: (s == "vindicated").mean()),
            overruled_rate=("status", lambda s: (s == "overruled").mean()),
            withdrawn_rate=("status", lambda s: (s == "withdrawn").mean()),
            undecided_rate=("status", lambda s: (s == "undecided_support").mean()),
        ).reset_index())

        # Save
        out_dir = self.cache_dir
        sp_path = os.path.join(out_dir, "summary_by_species.parquet")
        rk_path = os.path.join(out_dir, "summary_by_rank.parquet")
        sp.to_parquet(sp_path, index=False)
        rk.to_parquet(rk_path, index=False)
        if csv:
            sp.to_csv(os.path.join(out_dir, "summary_by_species.csv"), index=False)
            rk.to_csv(os.path.join(out_dir, "summary_by_rank.csv"), index=False)
        log(f"Wrote {sp_path} and {rk_path}")
        print("\nTop species by vindication rate (n>=3):")
        sp2 = sp[sp["n_props"] >= 3].sort_values(["n_vindicated_rate", "n_props"], ascending=[False, False]).head(20)
        with pd.option_context('display.max_rows', None, 'display.max_columns', None):
            print(sp2[['taxon_name','n_props','n_vindicated','n_overruled','n_withdrawn','n_undecided','n_unknown','n_vindicated_rate']])


# -------------------------
# CLI
# -------------------------

def parse_bbox(s: str) -> Tuple[float,float,float,float]:
    parts = [float(x) for x in s.split(',')]
    if len(parts) != 4:
        raise argparse.ArgumentTypeError("--bbox must be minlon,minlat,maxlon,maxlat")
    return parts[0], parts[1], parts[2], parts[3]


def main():
    ap = argparse.ArgumentParser(description="iNat proposal-level reliability")
    sub = ap.add_subparsers(dest="cmd", required=True)

    ap_ing = sub.add_parser("ingest", help="Download and cache data for a user")
    ap_ing.add_argument("--user", required=True, help="iNat login (e.g., schizoform)")
    ap_ing.add_argument("--out", default="data", help="Cache/output directory")

    ap_sum = sub.add_parser("summarize", help="Build proposals and summaries from cache")
    ap_sum.add_argument("--user", required=True, help="iNat login")
    ap_sum.add_argument("--out", default="data", help="Cache/output directory")
    ap_sum.add_argument("--start", help="Start date (YYYY-MM-DD)")
    ap_sum.add_argument("--end", help="End date (YYYY-MM-DD)")
    ap_sum.add_argument("--place-id", help="Comma-separated iNat place_ids to include")
    ap_sum.add_argument("--bbox", type=parse_bbox, help="minlon,minlat,maxlon,maxlat")
    ap_sum.add_argument("--csv", action="store_true", help="Also write CSV outputs")

    args = ap.parse_args()

    analyzer = Analyzer(cache_dir=args.out)

    if args.cmd == "ingest":
        analyzer.ingest(args.user)
        return

    if args.cmd == "summarize":
        place_ids = None
        if args.place_id:
            place_ids = [int(x) for x in args.place_id.split(',') if x.strip()]
        bbox = args.bbox if args.bbox else None
        analyzer.summarize(args.user, args.start, args.end, place_ids, bbox, csv=args.csv)
        return


if __name__ == "__main__":
    main()
