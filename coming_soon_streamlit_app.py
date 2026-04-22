"""Streamlit frontend for the Coming Soon Near You iNaturalist query.

Run with:  streamlit run coming_soon_streamlit_app.py
"""
import streamlit as st
import pandas as pd

from helpers import (
    coming_soon,
    get_inat_session,
    load_api_key,
    _lookup_place_name,
    _search_places,
)

st.set_page_config(page_title="Coming Soon Near You", page_icon="🌿", layout="wide")
st.title("Coming Soon Near You")

# ---------------------------------------------------------------------------
# Session / auth (created once per Streamlit session)
# ---------------------------------------------------------------------------

if "session" not in st.session_state:
    token = load_api_key()
    st.session_state["session"] = get_inat_session(token=token)

session = st.session_state["session"]

# ---------------------------------------------------------------------------
# Sidebar controls
# ---------------------------------------------------------------------------

st.sidebar.header("Query settings")

KIND_OPTIONS = [
    "any", "plants", "flowers", "fruits", "mushrooms",
    "animals", "wugs", "fish", "herps", "birds", "mammals",
    "butterflies", "caterpillars",
]
kind = st.sidebar.selectbox("Kind", KIND_OPTIONS, index=KIND_OPTIONS.index("flowers"))

place_mode = st.sidebar.radio("Location mode", ["Place ID", "Coordinates"])

if place_mode == "Place ID":
    search_query = st.sidebar.text_input(
        "Search place name", placeholder="e.g. Shenandoah", key="place_search"
    )
    if search_query and len(search_query.strip()) >= 3:
        search_results = _search_places(session, search_query)
        if search_results:
            result_labels = [r["display_name"] for r in search_results]
            chosen_idx = result_labels.index(
                st.sidebar.selectbox("Select place", result_labels, key="place_select")
            )
            st.session_state["place_id_input"] = search_results[chosen_idx]["id"]
        else:
            st.sidebar.caption("No places found.")

    place_id = st.sidebar.number_input(
        "Place ID", min_value=1, value=160915, step=1, key="place_id_input"
    )
    place_name = _cached_lookup_place_name(session, int(place_id))
    if place_name:
        st.sidebar.caption(f"📍 {place_name}")
    location_key = ("place", int(place_id))
else:
    lat = st.sidebar.number_input("Latitude", value=37.66933, format="%.5f")
    lon = st.sidebar.number_input("Longitude", value=-77.81001, format="%.5f")
    radius_km = st.sidebar.number_input("Radius (km)", min_value=1.0, value=5.0, step=1.0)
    location_key = ("coords", lat, lon, radius_km)

st.sidebar.markdown("---")
st.sidebar.subheader("Normalization & filtering")

NORM_OPTIONS = ["none", "time", "place", "overall"]
norm = st.sidebar.selectbox("Normalization", NORM_OPTIONS, index=NORM_OPTIONS.index("time"))

LINEAGE_OPTIONS = ["any", "native_endemic", "introduced"]
lineage_filter = st.sidebar.selectbox(
    "Lineage filter", LINEAGE_OPTIONS, index=LINEAGE_OPTIONS.index("native_endemic"),
)

limit = st.sidebar.slider("Species limit", min_value=1, max_value=25, value=7)
show_images = st.sidebar.checkbox("Show images", value=True)

# ---------------------------------------------------------------------------
# Run query
# ---------------------------------------------------------------------------


@st.cache_data(ttl=3600)
def _run_query(_session, kind, location_key, norm, limit, lineage_filter):
    """Cached wrapper around coming_soon().

    ``location_key`` is a hashable tuple: ("place", id) or ("coords", lat, lon, km).
    """
    mode, *vals = location_key
    loc_kwargs = {"places": [vals[0]]} if mode == "place" else {"loc": tuple(vals)}
    return coming_soon(
        kind=kind,
        **loc_kwargs,
        norm=None if norm == "none" else norm,
        limit=limit,
        fetch_images=False,
        use_cache=True,
        lineage_filter=lineage_filter,
        nativity_place_id='auto',
        session=_session,
    )


@st.cache_data(ttl=3600)
def _cached_lookup_place_name(_session, place_id: int):
    """Cached wrapper so place-name lookups don't fire on every widget change."""
    return _lookup_place_name(_session, place_id)


if st.sidebar.button("Run Query", type="primary", use_container_width=True):
    st.session_state["run"] = True
    st.session_state["query_pending"] = True

status_slot = st.sidebar.empty()

if st.session_state.get("run"):
    if st.session_state.get("query_pending"):
        status_slot.caption("Results... coming soon ⏳")
    try:
        res = _run_query(
            session,
            kind,
            location_key,
            norm,
            limit,
            lineage_filter,
        )
    except Exception as exc:
        exc_str = str(exc)
        if "are in the [columns]" in exc_str:
            st.warning(
                "No observations were found for this location and group combination. "
                "This usually means the place is too small or too specific to have enough iNaturalist data."
            )
            st.info(
                "**Try a larger place** — a county, state/national park, or state tends to work well. "
                "Places larger than a state (e.g. a country or continent) are usually too broad "
                "to give meaningful 'coming soon' results."
            )
        else:
            st.error(f"Query failed: {exc}")
            st.info("Verify your Place ID or coordinates and try a different group.")
        st.session_state["query_pending"] = False
        status_slot.warning("Query failed ⚠️")
        st.stop()

    if res is None or res.empty:
        st.warning("No results found. Try a larger radius or different group.")
        st.session_state["query_pending"] = False
        status_slot.warning("No results found ⚠️")
        st.stop()

    st.session_state["query_pending"] = False
    status_slot.success("Results ready ✅")
    st.success("Results ready ✅")
    st.subheader(f"Found {len(res)} taxa")

    if show_images:
        cols_per_row = min(len(res), 4)
        for row_start in range(0, len(res), cols_per_row):
            cols = st.columns(cols_per_row)
            for col_idx, (_, row) in enumerate(
                res.iloc[row_start : row_start + cols_per_row].iterrows()
            ):
                with cols[col_idx]:
                    common = row.get("taxon.preferred_common_name") or ""
                    scientific = row.get("taxon.name") or "Unknown"
                    photo_url = row.get("taxon.default_photo.medium_url")
                    wiki_url = row.get("taxon.wikipedia_url")
                    count = int(row.get("count", 0))

                    if (
                        photo_url
                        and isinstance(photo_url, str)
                        and photo_url.strip()
                    ):
                        st.image(photo_url, use_container_width=True)
                    else:
                        st.markdown("*No photo available*")

                    label = f"**{common}**" if common else ""
                    if (
                        wiki_url
                        and isinstance(wiki_url, str)
                        and wiki_url.strip()
                    ):
                        label = f"[{common or scientific}]({wiki_url})"
                    st.markdown(label)
                    nativity = row.get("nativity") or ""
                    nativity_icon = {
                        "Native": "🟢",
                        "Endemic": "🔵",
                        "Introduced": "🔴",
                        "Unknown": "⬜",
                    }.get(nativity, "")
                    if nativity:
                        nat_pid = row.get("nativity_place_id")
                        nat_place = (
                            _cached_lookup_place_name(session, int(nat_pid))
                            if nat_pid is not None and not pd.isna(nat_pid)
                            else None
                        )
                        scope = f" ({nat_place})" if nat_place else ""
                        nativity_str = f" · {nativity_icon} {nativity}{scope}"
                    else:
                        nativity_str = ""
                    st.caption(f"*{scientific}* · {count} obs{nativity_str}")
    else:
        show_cols = [
            "count",
            "taxon.name",
            "taxon.preferred_common_name",
            "nativity",
            "nativity_place_id",
            "taxon.wikipedia_url",
        ]
        safe_cols = [c for c in show_cols if c in res.columns]
        st.dataframe(res[safe_cols], use_container_width=True, hide_index=True)
else:
    st.info(
        "Configure settings in the sidebar and click **Run Query**.\n\n"
        "📱 On a small screen (phone), open the sidebar using the **>>** icon at the upper left."
    )