from __future__ import annotations

import json
import math
import os

# Prevent macOS Accelerate BLAS bugs from crashing NumPy on import
os.environ.setdefault("NPY_DISABLE_MAC_OS_ACCELERATE", "1")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")
os.environ.setdefault("VECLIB_DEFAULT_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")
os.environ.setdefault("ACCELERATE_DISABLE", "1")

from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd
import plotly.express as px
# PyDeck disabled due to segfault on macOS
# try:
#     import pydeck as pdk
#     PYDECK_AVAILABLE = True
# except (ImportError, OSError):
pdk = None
PYDECK_AVAILABLE = False
import streamlit as st
from sqlalchemy import create_engine, text

try:
    import pytds  # type: ignore
    from types import SimpleNamespace

    if not hasattr(pytds, "tds_session"):
        from pytds import tds as _pytds_tds  # type: ignore

        pytds.tds_session = SimpleNamespace(_token_map=_pytds_tds._token_map)  # type: ignore[attr-defined]

    import sqlalchemy_pytds  # noqa: F401  # Ensure pytds dialect is registered
    PYTDS_AVAILABLE = True
except ImportError:
    PYTDS_AVAILABLE = False

# SHAPELY DISABLED - Causes segfault on macOS with certain geometry data
# try:
#     from shapely import wkb, wkt
#     from shapely.geometry import shape, mapping
#     SHAPELY_AVAILABLE = True
# except (ImportError, OSError) as e:  # pragma: no cover
print("⚠️ Shapely disabled due to segfault issues on macOS")
wkb = None  # type: ignore
wkt = None  # type: ignore
shape = None  # type: ignore
mapping = None  # type: ignore
SHAPELY_AVAILABLE = False

import folium
from streamlit.components.v1 import html as components_html
import time

BOROUGH_CENTERS: Dict[str, Tuple[float, float]] = {
    "Manhattan": (40.7831, -73.9712),
    "Brooklyn": (40.6782, -73.9442),
    "Queens": (40.7282, -73.7949),
    "Bronx": (40.8448, -73.8648),
    "Staten Island": (40.5795, -74.1502),
    "EWR": (40.6895, -74.1745),
}

HERO_ZONE_COORDS: Dict[str, Tuple[float, float]] = {
    "JFK Airport": (40.6413, -73.7781),
    "LaGuardia Airport": (40.7769, -73.8740),
    "Midtown Center": (40.7549, -73.9840),
    "Times Sq/Theatre District": (40.7580, -73.9855),
    "Upper East Side South": (40.7736, -73.9566),
    "Upper West Side South": (40.7870, -73.9754),
    "Financial District South": (40.7014, -74.0122),
    "Harlem": (40.8116, -73.9465),
    "Long Island City/Queens Plaza": (40.7440, -73.9489),
    "Williamsburg (North)": (40.7223, -73.9570),
}

st.set_page_config(
    page_title="NYC Promise System Studio",
    page_icon="🚕",
    layout="wide",
    initial_sidebar_state="expanded",
)

CUSTOM_CSS = """
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
    html, body, [class*="css"], .stApp {
        font-family: 'Inter', sans-serif;
        background: radial-gradient(circle at top left, rgba(0, 198, 255, 0.12), transparent 45%),
                    radial-gradient(circle at bottom right, rgba(122, 98, 246, 0.18), transparent 50%),
                    #030b18;
        color: #f2f6fb;
    }
    .metric-box {
        background: rgba(7, 26, 51, 0.9);
        border-radius: 18px;
        padding: 18px 22px;
        border: 1px solid rgba(255,255,255,0.08);
        box-shadow: 0 18px 32px rgba(5, 20, 44, 0.35);
    }
    .metric-title {
        font-size: 0.85rem;
        color: #9db6d2;
        text-transform: uppercase;
        letter-spacing: 0.08em;
    }
    .metric-value {
        font-size: 2rem;
        font-weight: 600;
        color: #f7fafc;
    }
    .chip {
        display: inline-flex;
        align-items: center;
        gap: 8px;
        padding: 8px 14px;
        border-radius: 999px;
        background: rgba(0, 198, 255, 0.12);
        border: 1px solid rgba(0, 198, 255, 0.25);
        margin-right: 8px;
        font-size: 0.85rem;
    }
    .legend-swatch {
        width: 16px;
        height: 16px;
        border-radius: 4px;
        display: inline-block;
        margin-right: 6px;
    }
</style>
"""

st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

st.title("🚕 NYC Promise System • Streamlit Studio")
st.caption(
    "Interactive command center for late-arrival risk across boroughs, promises, and corridors."
)


@st.cache_resource(show_spinner=False)
def get_engine():
    if not PYTDS_AVAILABLE:
        raise RuntimeError(
            "sqlalchemy-pytds is not installed. Install it with `pip install sqlalchemy-pytds python-tds`."
        )

    server = os.getenv("PROMISE_DB_SERVER", "localhost")
    database = os.getenv("PROMISE_DB_DATABASE", "NYC_Promise_System")
    username = os.getenv("PROMISE_DB_USERNAME", "SA")
    password = os.getenv("PROMISE_DB_PASSWORD", "Allah248012!")
    port = int(os.getenv("PROMISE_DB_PORT", "1433"))

    url = f"mssql+pytds://{username}:{password}@{server}:{port}/{database}?charset=utf8"
    return create_engine(url, pool_pre_ping=True, connect_args={"timeout": 5})


@dataclass
class MapDataset:
    months: List[Dict[str, object]]
    boroughs: List[str]
    zones_geojson: List[Dict[str, object]]
    trip_aggregates: pd.DataFrame
    policy_curves: Dict[int, Dict[int, float]]


def _detect_geometry_column(columns: Iterable[str]) -> Optional[str]:
    tokens = ["geometry", "geom", "the_geom", "shape", "wkt", "wkb", "geojson", "polygon", "multipolygon"]
    for col in columns:
        lower = col.lower()
        if any(token in lower for token in tokens):
            return col
    return None


def _choose_coordinate_column(columns: Iterable[str], axis: str) -> Optional[str]:
    best_score = -1
    best_col: Optional[str] = None
    for col in columns:
        lower = col.lower()
        if any(ex in lower for ex in ["pickup", "dropoff", "pu_", "do_"]):
            continue
        score = 0
        if "centroid" in lower:
            score += 5
        if axis == "lat":
            if "latitude" in lower:
                score += 4
            if "lat" in lower or lower.endswith("_y"):
                score += 2
        else:
            if "longitude" in lower:
                score += 4
            if any(token in lower for token in ["lon", "lng", "long"]) or lower.endswith("_x"):
                score += 2
        if score > best_score and score > 0:
            best_score = score
            best_col = col
    return best_col


def _safe_float(value: object) -> Optional[float]:
    try:
        if value is None or (isinstance(value, float) and math.isnan(value)):
            return None
        return float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None


def _load_zone_features(zones_df: pd.DataFrame) -> List[Dict[str, object]]:
    geom_col = _detect_geometry_column(zones_df.columns)
    lat_col = _choose_coordinate_column(zones_df.columns, "lat")
    lon_col = _choose_coordinate_column(zones_df.columns, "lon")

    if lat_col is None:
        lat_col = "centroid_lat"
        zones_df[lat_col] = np.nan
    if lon_col is None:
        lon_col = "centroid_lon"
        zones_df[lon_col] = np.nan

    geometry_objects: List[object] = []
    if geom_col and shape is not None:
        for val in zones_df[geom_col]:
            geom = None
            if val is None or (isinstance(val, float) and math.isnan(val)):
                geometry_objects.append(None)
                continue
            try:
                if isinstance(val, (dict, list)):
                    geom = shape(val)
                elif isinstance(val, (bytes, bytearray, memoryview)) and wkb is not None:
                    geom = wkb.loads(bytes(val))
                elif isinstance(val, str) and val.strip():
                    txt = val.strip()
                    if txt.startswith("{") and shape is not None:
                        geom = shape(json.loads(txt))
                    elif wkt is not None:
                        geom = wkt.loads(txt)
                elif shape is not None:
                    geom = shape(val)
            except Exception:
                geom = None
            geometry_objects.append(geom)
    else:
        geometry_objects = [None] * len(zones_df)

    zone_id_candidates = [col for col in zones_df.columns if col.lower() in {"locationid", "location_id", "zone_id", "id"}]
    zone_id_col = zone_id_candidates[0] if zone_id_candidates else "LocationID"

    features: List[Dict[str, object]] = []
    for idx, row in zones_df.iterrows():
        zone_name = row.get("Zone", "")
        borough = row.get("Borough", "")
        lat = _safe_float(row.get(lat_col))
        lon = _safe_float(row.get(lon_col))
        geom = geometry_objects[idx]

        # Geometry centroid disabled due to macOS segfault
        # if (lat is None or lon is None) and geom is not None:
        #     centroid = geom.centroid
        #     lat, lon = centroid.y, centroid.x
        if (lat is None or lon is None) and zone_name in HERO_ZONE_COORDS:
            lat, lon = HERO_ZONE_COORDS[zone_name]
        if (lat is None or lon is None) and borough in BOROUGH_CENTERS:
            lat, lon = BOROUGH_CENTERS[borough]

        feature_geometry = mapping(geom) if geom is not None and mapping is not None else None
        features.append(
            {
                "type": "Feature",
                "properties": {
                    "location_id": int(row[zone_id_col]),
                    "zone": zone_name,
                    "borough": borough,
                    "centroid_lat": lat,
                    "centroid_lon": lon,
                },
                "geometry": feature_geometry,
            }
        )
    return features


def _build_policy_lookup(policy_df: pd.DataFrame) -> Dict[int, Dict[int, float]]:
    lookup: Dict[int, Dict[int, float]] = {}
    if policy_df.empty:
        return lookup

    cols = {col.lower(): col for col in policy_df.columns}
    loc_col = cols.get("pulocationid") or cols.get("pickup_location_id") or cols.get("locationid")
    perc_col = next((cols[key] for key in cols if "percentile" in key), None)
    late_col = next((cols[key] for key in cols if "late" in key and ("rate" in key or "pct" in key or "prob" in key)), None)

    if not (loc_col and perc_col and late_col):
        return lookup

    for _, row in policy_df.iterrows():
        loc = row.get(loc_col)
        perc = row.get(perc_col)
        late = row.get(late_col)
        if pd.isna(loc) or pd.isna(perc) or pd.isna(late):
            continue
        zone_lookup = lookup.setdefault(int(loc), {})
        zone_lookup[int(perc)] = float(late)
    return lookup


def load_dataset() -> MapDataset:
    print("🔄 Starting database connection...")
    engine = get_engine()
    print("✅ Engine created")
    with engine.connect() as conn:
        print("🔄 Loading zones from database...")
        # Load zones WITHOUT geometry column to avoid crash
        try:
            zones_df = pd.read_sql(text("SELECT LocationID, Zone, Borough, service_zone FROM dim_zone"), conn)
        except:
            # Fallback to all columns if specific ones don't exist
            zones_df = pd.read_sql(text("SELECT LocationID, Zone, Borough FROM dim_zone"), conn)
        print(f"✅ Loaded {len(zones_df)} zones")
        trips_df = pd.read_sql(
            text(
                """
                SELECT
                    t.date,
                    t.hour,
                    t.PULocationID,
                    t.DOLocationID,
                    t.trips,
                    t.median_eta,
                    t.p90_eta,
                    t.late_rate,
                    t.avg_distance,
                    t.wet_flag,
                    t.tavg,
                    t.prcp,
                    pz.Zone AS pickup_zone,
                    pz.Borough AS pickup_borough,
                    dz.Zone AS dropoff_zone,
                    dz.Borough AS dropoff_borough
                FROM fact_trip_agg t
                JOIN dim_zone pz ON t.PULocationID = pz.LocationID
                JOIN dim_zone dz ON t.DOLocationID = dz.LocationID
                WHERE t.date BETWEEN '2025-05-01' AND '2025-07-31'
                """
            ),
            conn,
        )
        policy_df = pd.read_sql(text("SELECT * FROM fact_policy_curve"), conn)

    trips_df["date"] = pd.to_datetime(trips_df["date"])
    trips_df["month_key"] = trips_df["date"].dt.strftime("%Y-%m")
    trips_df["month_label"] = trips_df["date"].dt.strftime("%b %Y")
    trips_df["month_name"] = trips_df["date"].dt.strftime("%B")
    trips_df["month_num"] = trips_df["date"].dt.month
    trips_df["hour"] = trips_df["hour"].astype(int)
    trips_df["wet_label"] = trips_df["wet_flag"].map({0: "Dry", 1: "Wet"}).fillna("Unknown")

    print("🔄 Processing zone features (geometry data)...")
    # Geometry processing disabled - using simple coordinate-based features
    zone_features = []
    for idx, row in zones_df.iterrows():
        zone_id = int(row.get('LocationID', idx))
        zone_name = str(row.get('Zone', f'Zone {zone_id}'))
        borough = str(row.get('Borough', 'Unknown'))

        # Get coordinates from hardcoded lookup
        lat, lon = None, None
        if zone_name in HERO_ZONE_COORDS:
            lat, lon = HERO_ZONE_COORDS[zone_name]
        elif borough in BOROUGH_CENTERS:
            lat, lon = BOROUGH_CENTERS[borough]

        zone_features.append({
            "type": "Feature",
            "properties": {
                "location_id": zone_id,
                "zone": zone_name,
                "borough": borough,
                "centroid_lat": lat,
                "centroid_lon": lon,
            },
            "geometry": None,  # No geometry to avoid crashes
        })
    print(f"✅ Created {len(zone_features)} zone features without geometry")
    print(f"✅ Processed {len(zone_features)} zone features")
    months_meta = (
        trips_df[["month_key", "month_label", "month_name", "month_num"]]
        .drop_duplicates()
        .sort_values("month_key")
        .to_dict(orient="records")
    )

    policy_lookup = _build_policy_lookup(policy_df)

    return MapDataset(
        months=months_meta,
        boroughs=sorted(trips_df["pickup_borough"].dropna().unique().tolist()),
        zones_geojson=zone_features,
        trip_aggregates=trips_df,
        policy_curves=policy_lookup,
    )


def _filter_trips(
    dataset: MapDataset,
    month: str,
    hours: Tuple[int, int],
    boroughs: List[str],
    weather: str,
    apply_hour_filter: bool = True,
) -> pd.DataFrame:
    df = dataset.trip_aggregates.copy()
    if month:
        df = df[df["month_key"] == month]
    if apply_hour_filter:
        start, end = hours
        df = df[(df["hour"] >= start) & (df["hour"] <= end)]
    if boroughs:
        df = df[df["pickup_borough"].isin(boroughs)]
    if weather == "Dry Only":
        df = df[df["wet_flag"] == 0]
    elif weather == "Wet Only":
        df = df[df["wet_flag"] == 1]
    return df


def _weighted_average(values: pd.Series, weights: pd.Series) -> float:
    if weights.sum() == 0:
        return float(values.mean()) if len(values) else 0.0
    return float(np.average(values, weights=weights))


def _compute_zone_stats(filtered: pd.DataFrame) -> pd.DataFrame:
    if filtered.empty:
        return pd.DataFrame(
            columns=[
                "zoneId",
                "zoneName",
                "borough",
                "trips",
                "p50",
                "p90",
                "lateRate",
                "avgDistance",
            ]
        )

    grouped = filtered.groupby("PULocationID")
    rows = []
    for zone_id, group in grouped:
        weights = group["trips"].replace(0, np.nan)
        trips = float(group["trips"].sum())
        if not np.isfinite(trips) or trips <= 0:
            weights = np.ones(len(group))
            trips = float(len(group))
        rows.append(
            {
                "zoneId": int(zone_id),
                "zoneName": group["pickup_zone"].iloc[0],
                "borough": group["pickup_borough"].iloc[0],
                "trips": trips,
                "p50": _weighted_average(group["median_eta"], weights),
                "p90": _weighted_average(group["p90_eta"], weights),
                "lateRate": _weighted_average(group["late_rate"], weights),
                "avgDistance": _weighted_average(group["avg_distance"], weights),
            }
        )
    return pd.DataFrame(rows)


def _compute_corridors(filtered: pd.DataFrame) -> pd.DataFrame:
    if filtered.empty:
        return pd.DataFrame(
            columns=[
                "from",
                "to",
                "fromZone",
                "toZone",
                "fromBorough",
                "toBorough",
                "trips",
                "lateRate",
                "p50",
                "p90",
            ]
        )

    grouped = filtered.groupby(["PULocationID", "DOLocationID"])
    rows = []
    for (pu, do), group in grouped:
        weights = group["trips"].replace(0, np.nan)
        trips = float(group["trips"].sum())
        if not np.isfinite(trips) or trips <= 0:
            weights = np.ones(len(group))
            trips = float(len(group))
        rows.append(
            {
                "from": int(pu),
                "to": int(do),
                "fromZone": group["pickup_zone"].iloc[0],
                "toZone": group["dropoff_zone"].iloc[0],
                "fromBorough": group["pickup_borough"].iloc[0],
                "toBorough": group["dropoff_borough"].iloc[0],
                "trips": trips,
                "lateRate": _weighted_average(group["late_rate"], weights),
                "p50": _weighted_average(group["median_eta"], weights),
                "p90": _weighted_average(group["p90_eta"], weights),
            }
        )
    return pd.DataFrame(rows)


def _compute_scale(values: Iterable[float], percentile: int) -> Dict[str, float]:
    clean = [val for val in values if isinstance(val, (int, float)) and np.isfinite(val)]
    if not clean:
        return {"low": 0.15, "medium": 0.25, "high": 0.35}
    arr = np.array(clean)
    low = float(np.percentile(arr, max(0, percentile - 20)))
    medium = float(np.percentile(arr, max(0, percentile - 10)))
    high = float(np.percentile(arr, percentile))
    return {"low": low, "medium": medium, "high": high}


def _get_policy_value(policy_lookup: Dict[int, Dict[int, float]], zone_id: int, percentile: int) -> Optional[float]:
    zone_policy = policy_lookup.get(zone_id)
    if not zone_policy:
        return None
    if percentile in zone_policy:
        return zone_policy[percentile]
    closest_key = min(zone_policy.keys(), key=lambda key: abs(key - percentile))
    return zone_policy.get(closest_key)


def _format_minutes(value: float) -> str:
    return f"{value:.1f}" if np.isfinite(value) else "--"


def _format_percent(value: float) -> str:
    return f"{value * 100:.1f}%" if np.isfinite(value) else "--"


def _format_trips(value: float) -> str:
    if value >= 1_000_000:
        return f"{value / 1_000_000:.1f}M"
    if value >= 1_000:
        return f"{value / 1_000:.1f}K"
    return f"{int(value):,}"


def _build_route_tooltip(
    from_zone: str,
    to_zone: str,
    from_borough: str,
    to_borough: str,
    trips: float,
    late_rate: float,
    p50: float,
    p90: float,
    month_label: str = "",
    hour_range: str = "",
) -> str:
    """
    Build a detailed, professional tooltip for route corridors.

    Args:
        from_zone: Origin zone name
        to_zone: Destination zone name
        from_borough: Origin borough
        to_borough: Destination borough
        trips: Total trip count
        late_rate: Late arrival rate (0-1)
        p50: Median ETA in minutes
        p90: 90th percentile ETA in minutes
        month_label: Optional month label (e.g., "Jul 2025")
        hour_range: Optional hour range (e.g., "00:00-23:00")

    Returns:
        HTML string for tooltip
    """
    # Determine status color based on late rate
    if late_rate <= 0.15:
        status_color = "#27ae60"  # green
        status_text = "EXCELLENT"
    elif late_rate <= 0.25:
        status_color = "#f39c12"  # yellow
        status_text = "GOOD"
    elif late_rate <= 0.35:
        status_color = "#e67e22"  # orange
        status_text = "FAIR"
    else:
        status_color = "#e74c3c"  # red
        status_text = "POOR"

    tooltip = (
        f"<div style='background:#ffffff; padding:16px 20px; border-radius:14px; "
        f"box-shadow:0 8px 24px rgba(0,0,0,0.25), 0 2px 8px rgba(0,0,0,0.15); "
        f"border-left:5px solid {status_color}; min-width:280px; font-family:Inter,sans-serif;'>"

        # Header: Origin → Destination
        f"<div style='font-size:15px; font-weight:700; color:#2c3e50; margin-bottom:10px; "
        f"border-bottom:2px solid #ecf0f1; padding-bottom:8px;'>"
        f"<span style='color:#27ae60;'>🟢</span> {from_zone}<br>"
        f"<span style='margin:0 8px; color:#95a5a6;'>→</span><br>"
        f"<span style='color:#3498db;'>🔵</span> {to_zone}"
        f"</div>"

        # Borough info
        f"<div style='font-size:11px; color:#7f8c8d; margin-bottom:12px; font-style:italic;'>"
        f"{from_borough} → {to_borough}"
        f"</div>"

        # Stats section
        f"<div style='font-size:12px; color:#34495e; line-height:1.9;'>"

        # Trips
        f"<div style='margin-bottom:6px;'>"
        f"<span style='color:#7f8c8d;'>📊 Total Trips:</span> "
        f"<b style='color:#2c3e50;'>{_format_trips(trips)}</b>"
        f"</div>"

        # Late Rate with status
        f"<div style='margin-bottom:6px;'>"
        f"<span style='color:#7f8c8d;'>⚠️ Late Rate:</span> "
        f"<b style='color:{status_color}; font-weight:700;'>{_format_percent(late_rate)}</b> "
        f"<span style='font-size:10px; padding:2px 6px; background:{status_color}; color:#fff; "
        f"border-radius:4px; font-weight:600;'>{status_text}</span>"
        f"</div>"

        # ETA P50
        f"<div style='margin-bottom:6px;'>"
        f"<span style='color:#7f8c8d;'>⏱ Avg Duration (P50):</span> "
        f"<b style='color:#3498db;'>{_format_minutes(p50)} min</b>"
        f"</div>"

        # ETA P90
        f"<div style='margin-bottom:6px;'>"
        f"<span style='color:#7f8c8d;'>⏱ Slow Duration (P90):</span> "
        f"<b style='color:#e67e22;'>{_format_minutes(p90)} min</b>"
        f"</div>"
    )

    # Add filter info if provided
    if month_label or hour_range:
        tooltip += (
            f"<div style='margin-top:12px; padding-top:10px; border-top:1px solid #ecf0f1; "
            f"font-size:10px; color:#95a5a6;'>"
        )
        if month_label:
            tooltip += f"📅 {month_label}"
        if hour_range:
            tooltip += f" • 🕐 {hour_range}"
        tooltip += "</div>"

    tooltip += "</div></div>"
    return tooltip


def _build_zone_popup(row: pd.Series, percentile: int, policy_lookup: Dict[int, Dict[int, float]]) -> str:
    policy_val = _get_policy_value(policy_lookup, int(row.zoneId), percentile)
    parts = [
        f"<h4 style='margin-bottom:4px;'>{row.zoneName}</h4>",
        f"<div style='color:#95a5a6;margin-bottom:6px;'> {row.borough} • {_format_trips(row.trips)} trips</div>",
        f"<div>Late Rate: <b>{_format_percent(row.lateRate)}</b></div>",
        f"<div>P50 ETA: <b>{_format_minutes(row.p50)} min</b></div>",
        f"<div>P90 ETA: <b>{_format_minutes(row.p90)} min</b></div>",
    ]
    if policy_val is not None:
        parts.append(
            f"<div style='margin-top:6px;color:#2ecc71;'>If Promise = P{percentile} → Late = {_format_percent(policy_val)}</div>"
        )
    return "".join(parts)


def _color_for_value(value: Optional[float], scale: Dict[str, float]) -> str:
    if value is None or not np.isfinite(value):
        return "#7f8c8d"
    if value <= scale["low"]:
        return "#1abc9c"
    if value <= scale["medium"]:
        return "#f6c744"
    if value <= scale["high"]:
        return "#e67e22"
    return "#c0392b"


def _color_rgb(value: Optional[float], scale: Dict[str, float]) -> List[int]:
    hex_color = _color_for_value(value, scale).lstrip('#')
    if len(hex_color) != 6:
        return [127, 140, 141]
    return [int(hex_color[i:i+2], 16) for i in (0, 2, 4)]


def build_map(
    zone_stats: pd.DataFrame,
    corridor_stats: pd.DataFrame,
    zone_features: Dict[int, Dict[str, object]],
    scale: Dict[str, float],
    show_layers: List[str],
    min_trips: int,
    percentile: int,
    policy_lookup: Dict[int, Dict[int, float]],
) -> folium.Map:
    base_map = folium.Map(location=[40.7128, -73.94], zoom_start=11, tiles="cartodbpositron")
    zone_stats_map = {row.zoneId: row for row in zone_stats.itertuples()}

    if "Choropleth" in show_layers:
        features_with_geom = []
        for zone_id, feature in zone_features.items():
            if feature.get("geometry") and zone_id in zone_stats_map:
                feature_copy = json.loads(json.dumps(feature))
                feature_copy["properties"]["lateRate"] = zone_stats_map[zone_id].lateRate
                features_with_geom.append(feature_copy)
        if features_with_geom:
            folium.GeoJson(
                {
                    "type": "FeatureCollection",
                    "features": features_with_geom,
                },
                style_function=lambda feat: {
                    "fillColor": _color_for_value(feat["properties"].get("lateRate"), scale),
                    "color": "#0d1f33",
                    "fillOpacity": 0.75,
                    "weight": 1,
                },
                highlight_function=lambda feat: {"weight": 2, "color": "#2ecc71"},
                tooltip=folium.features.GeoJsonTooltip(
                    fields=["zone", "borough", "lateRate"],
                    aliases=["Zone", "Borough", "Late Rate"],
                    localize=True,
                    labels=True,
                ),
            ).add_to(base_map)

    if "Bubbles" in show_layers and not zone_stats.empty:
        max_trips = zone_stats["trips"].max() or 1
        for _, row in zone_stats.iterrows():
            feature = zone_features.get(int(row.zoneId))
            if not feature:
                continue
            lat = feature["properties"].get("centroid_lat")
            lon = feature["properties"].get("centroid_lon")
            if not isinstance(lat, (int, float)) or not isinstance(lon, (int, float)):
                continue
            radius = 6 + (row.trips / max_trips) * 20
            popup_html = _build_zone_popup(row, percentile, policy_lookup)
            folium.CircleMarker(
                location=[lat, lon],
                radius=radius,
                color=_color_for_value(row.lateRate, scale),
                fill=True,
                fill_color=_color_for_value(row.lateRate, scale),
                fill_opacity=0.85,
                weight=1.2,
                popup=folium.Popup(popup_html, max_width=320),
            ).add_to(base_map)

    if "Corridors" in show_layers and not corridor_stats.empty:
        eligible = corridor_stats[corridor_stats["trips"] >= min_trips]
        eligible = eligible.sort_values(["lateRate", "trips"], ascending=[False, False]).head(10)
        for _, row in eligible.iterrows():
            start = zone_features.get(int(row["from"]))
            end = zone_features.get(int(row["to"]))
            if not start or not end:
                continue
            start_lat = start["properties"].get("centroid_lat")
            start_lon = start["properties"].get("centroid_lon")
            end_lat = end["properties"].get("centroid_lat")
            end_lon = end["properties"].get("centroid_lon")
            if not all(isinstance(val, (int, float)) for val in [start_lat, start_lon, end_lat, end_lon]):
                continue
            popup_html = (
                f"<b>{row['fromZone']} → {row['toZone']}</b><br>"
                f"Trips: {_format_trips(row['trips'])}<br>"
                f"Late: {_format_percent(row['lateRate'])}<br>"
                f"P90: {_format_minutes(row['p90'])} min"
            )
            folium.PolyLine(
                locations=[
                    [float(start_lat), float(start_lon)],
                    [float(end_lat), float(end_lon)],
                ],
                color=_color_for_value(row["lateRate"], scale),
                weight=4,
                opacity=0.65,
                dash_array="6, 8",
                popup=folium.Popup(popup_html, max_width=320),
            ).add_to(base_map)

    folium.LayerControl(collapsed=False).add_to(base_map)
    return base_map


def _prepare_policy_chart_data(
    policy_lookup: Dict[int, Dict[int, float]],
    zone_ids: Iterable[int],
    fallback_df: pd.DataFrame,
) -> pd.DataFrame:
    rows = []
    for zone_id in zone_ids:
        entries = policy_lookup.get(int(zone_id), {})
        for percentile, late in entries.items():
            rows.append({"percentile": percentile, "late_rate": late})
    if rows:
        df = pd.DataFrame(rows)
        return df.groupby("percentile", as_index=False)["late_rate"].mean()

    if fallback_df.empty or fallback_df['late_rate'].dropna().empty:
        return pd.DataFrame(columns=['percentile', 'late_rate'])

    percentiles = list(range(50, 96, 5))
    values = fallback_df['late_rate'].dropna().values
    fallback_rows = [
        {"percentile": p, "late_rate": float(np.percentile(values, p))}
        for p in percentiles
    ]
    return pd.DataFrame(fallback_rows)


def _render_legend(scale: Dict[str, float]):
    st.markdown(
        f"""
        <div class="chip"><span class="legend-swatch" style="background:#1abc9c;"></span>≤ {_format_percent(scale['low'])}</div>
        <div class="chip"><span class="legend-swatch" style="background:#f6c744;"></span>≤ {_format_percent(scale['medium'])}</div>
        <div class="chip"><span class="legend-swatch" style="background:#e67e22;"></span>≤ {_format_percent(scale['high'])}</div>
        <div class="chip"><span class="legend-swatch" style="background:#c0392b;"></span>> {_format_percent(scale['high'])}</div>
        """,
        unsafe_allow_html=True,
    )


def _render_insight_chips(zone_count: int, corridor_count: int, record_count: int):
    cols = st.columns(3)
    cols[0].markdown(f"<div class='chip'>Pickup Zones: <strong>{zone_count}</strong></div>", unsafe_allow_html=True)
    cols[1].markdown(f"<div class='chip'>Corridors: <strong>{corridor_count}</strong></div>", unsafe_allow_html=True)
    cols[2].markdown(f"<div class='chip'>Records: <strong>{record_count}</strong></div>", unsafe_allow_html=True)


try:
    with st.spinner("🔄 Loading data from database..."):
        dataset = load_dataset()
except Exception as e:
    st.error(f"❌ Failed to load data: {str(e)}")
    st.info("Please check if the SQL Server database is running and accessible.")
    st.code(f"Error details:\n{str(e)}", language="text")
    st.stop()

with st.sidebar:
    st.header("Time Controls")
    month_options = [item["month_key"] for item in dataset.months]
    month_labels = {item["month_key"]: item["month_label"] for item in dataset.months}
    default_month = month_options[-1] if month_options else ""
    month_key = st.selectbox(
        "Month",
        options=month_options,
        format_func=lambda key: month_labels.get(key, key),
        index=month_options.index(default_month) if default_month in month_options else 0,
    )
    hour_range = st.slider("Hour Range", min_value=0, max_value=23, value=(0, 23))

    st.header("Filters")
    boroughs = st.multiselect(
        "Pickup Boroughs",
        options=dataset.boroughs,
        default=dataset.boroughs,
    )
    weather = st.radio("Weather", options=["All", "Dry Only", "Wet Only"], index=0)
    min_trips = st.number_input("Minimum Trips", min_value=0, max_value=1000, value=10, step=5)

    st.header("Promise Policy")
    percentile = st.slider("Promise Percentile", min_value=50, max_value=95, value=90, step=5)
    layer_options = ["Choropleth", "Bubbles", "Corridors"]
    show_layers = st.multiselect("Map Layers", options=layer_options, default=layer_options)

filtered_for_map = _filter_trips(dataset, month_key, hour_range, boroughs, weather, apply_hour_filter=True)
filtered_full_window = _filter_trips(dataset, month_key, hour_range, boroughs, weather, apply_hour_filter=False)

zone_stats = _compute_zone_stats(filtered_for_map).sort_values("lateRate", ascending=False)
corridor_stats = _compute_corridors(filtered_for_map)
scale = _compute_scale(zone_stats["lateRate"].tolist(), percentile)
zone_lookup = {feature["properties"]["location_id"]: feature for feature in dataset.zones_geojson}

st.subheader(
    f"Summer 2025 Performance • {month_key} | Hours {hour_range[0]:02d}:00 – {hour_range[1]:02d}:00"
)

if zone_stats.empty:
    st.warning("No data available for the current filter set. Adjust filters to explore more activity.")
else:
    total_trips = float(zone_stats["trips"].sum())
    weighted_p50 = _weighted_average(zone_stats["p50"], zone_stats["trips"])
    weighted_p90 = _weighted_average(zone_stats["p90"], zone_stats["trips"])
    weighted_late = _weighted_average(zone_stats["lateRate"], zone_stats["trips"])

    kpi_cols = st.columns(4)
    for col, title, value in zip(
        kpi_cols,
        ["P50 ETA (min)", "P90 ETA (min)", "Late Arrivals", "Trips"],
        [
            _format_minutes(weighted_p50),
            _format_minutes(weighted_p90),
            _format_percent(weighted_late),
            _format_trips(total_trips),
        ],
    ):
        col.markdown(
            f"<div class='metric-box'><div class='metric-title'>{title}</div><div class='metric-value'>{value}</div></div>",
            unsafe_allow_html=True,
        )

    tab_2d, tab_3d, tab_live = st.tabs(["2D Atlas", "3D Cityscape", "🔴 Live Stream"])

    with tab_2d:
        st.info("💡 Click the button below to render the interactive 2D map")

        if st.button("🗺️ Load Interactive Map", key="load_map_btn"):
            try:
                with st.spinner("Rendering map..."):
                    map_object = build_map(
                        zone_stats,
                        corridor_stats,
                        zone_lookup,
                        scale,
                        show_layers,
                        min_trips,
                        percentile,
                        dataset.policy_curves,
                    )
                    components_html(map_object.get_root().render(), height=700)
                    st.success("✅ Map loaded successfully!")
            except Exception as e:
                st.error(f"⚠️ Failed to render map: {str(e)}")
                st.info("Map rendering failed - this is a known issue with folium on some macOS systems")

        _render_legend(scale)

    with tab_3d:
        st.markdown("### 🌃 3D NYC Promise Cityscape - Enhanced Interactive Edition")
        st.caption("✨ Professional visualization with smart tooltips, 3D columns, and dynamic color gradients")

        # Feature explanation card
        with st.expander("ℹ️ What's New in this Enhanced 3D View", expanded=False):
            st.markdown("""
            **🎨 Visual Enhancements:**
            - **Smart Tooltips:** Hover over zones to see rich, styled tooltips with gradient backgrounds and shadows
            - **3D Column Buildings:** Top 15 zones displayed as vertical columns - height = trip volume
            - **Color Gradients:** Zones colored by late rate (🟢 Green → 🟡 Yellow → 🔴 Red)
            - **Size Scaling:** Marker size represents trip volume (15px to 80px range)

            **🚦 Route Features:**
            - **Corridor Lines:** Top 20 routes with gradient colors based on late rate
            - **Start/End Pins:** Green pins (🟢) for pickup, red pins (🔴) for dropoff
            - **Enhanced Tooltips:** Rich hover details for each route segment

            **💫 Interactive Elements:**
            - **Pulse Effects:** Double-layer glow animations that follow your animation speed
            - **Hover Highlights:** Objects glow when you hover over them
            - **3D Buildings:** Trip density visualized as stacked column segments

            **🎯 Controls:**
            - Adjust map style, pulse speed, building height, and route visibility using the controls above
            """)

        st.markdown("---")

        # Enhanced 3D view controls with better descriptions
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            view_mode = st.selectbox(
                "🗺️ Map Style",
                ["Skyline", "Satellite", "Navigation", "Dark Mode"],
                help="Choose the base map style for your 3D visualization"
            )
        with col2:
            animation_speed = st.slider(
                "💫 Pulse Speed",
                0.5, 3.0, 1.5, 0.1,
                help="Control the glow/pulse animation speed around markers"
            )
        with col3:
            building_height = st.slider(
                "🏢 3D Column Height",
                10, 100, 25, 5,
                help="Adjust the height of 3D columns representing trip density"
            )
        with col4:
            show_routes = st.checkbox(
                "🚦 Show Routes",
                value=True,
                help="Display top 20 routes with start/end markers"
            )

        zone_stats_3d = zone_stats.copy()
        zone_stats_3d["lat"] = zone_stats_3d["zoneId"].apply(
            lambda z: zone_lookup.get(int(z), {}).get("properties", {}).get("centroid_lat")
        )
        zone_stats_3d["lon"] = zone_stats_3d["zoneId"].apply(
            lambda z: zone_lookup.get(int(z), {}).get("properties", {}).get("centroid_lon")
        )
        zone_stats_3d = zone_stats_3d.dropna(subset=["lat", "lon"])

        if zone_stats_3d.empty:
            st.info("3D görünüm için yeterli koordinat verisi bulunamadı. Filtreleri genişletmeyi deneyin.")
        else:
            zone_stats_3d["latePct"] = (zone_stats_3d["lateRate"] * 100).round(1)
            zone_stats_3d["tripsFormatted"] = zone_stats_3d["trips"].apply(lambda x: f"{int(x):,}")
            zone_stats_3d["p90Formatted"] = zone_stats_3d["p90"].apply(lambda x: f"{x:.1f}")

            # Enhanced size scaling based on trip volume - larger range for better visibility
            min_trips = float(zone_stats_3d["trips"].min() or 0)
            max_trips = float(zone_stats_3d["trips"].max() or 1)
            if max_trips - min_trips <= 0:
                zone_stats_3d["size"] = 28
            else:
                # More dynamic sizing: 15px to 80px based on trip volume
                zone_stats_3d["size"] = np.interp(
                    zone_stats_3d["trips"],
                    (min_trips, max_trips),
                    (15, 80),
                )

            hero_names = set(HERO_ZONE_COORDS.keys())
            hottest_cutoff = zone_stats_3d["lateRate"].quantile(0.9)
            marker_symbols = zone_stats_3d.apply(
                lambda row: "star" if row["zoneName"] in hero_names else ("fire-station" if row["lateRate"] >= hottest_cutoff else "marker"),
                axis=1,
            ).tolist()

            # Enhanced tooltip with professional styling - mor/gri gradient + shadow
            hover_template = (
                "<div style='background:linear-gradient(145deg,#4a148c,#311b92,#1a237e);"
                " padding:16px 20px; border-radius:16px; "
                " box-shadow:0 8px 32px rgba(0,0,0,0.6), 0 0 20px rgba(156,39,176,0.5);"
                " border:2px solid rgba(186,104,200,0.8); min-width:240px;'>"
                "<div style='font-size:17px;font-weight:700;margin-bottom:8px;color:#f3e5f5;"
                " text-shadow:0 2px 4px rgba(0,0,0,0.4);'>✨ %{text}</div>"
                "<div style='font-size:13px;color:#e1bee7;line-height:1.8;'>"
                "📍 <b style='color:#ce93d8;'>Borough:</b> %{customdata[3]}<br>"
                "🚕 <b style='color:#ce93d8;'>Trips:</b> <span style='color:#ffeb3b;'>%{customdata[0]}</span><br>"
                "⏱ <b style='color:#ce93d8;'>Late Rate:</b> <span style='color:#ff5252;font-weight:600;'>%{customdata[1]}%</span><br>"
                "🎯 <b style='color:#ce93d8;'>P90 ETA:</b> <span style='color:#80deea;'>%{customdata[2]} min</span>"
                "</div>"
                "<div style='margin-top:10px;padding-top:8px;border-top:1px solid rgba(186,104,200,0.4);"
                " font-size:11px;color:#ba68c8;font-style:italic;'>Hover to explore • Click for details</div>"
                "</div><extra></extra>"
            )

            customdata = zone_stats_3d[["tripsFormatted", "latePct", "p90Formatted", "borough"]].to_numpy()

            import plotly.graph_objects as go

            plotly_styles = {
                "Skyline": "carto-darkmatter",
                "Satellite": "open-street-map",
                "Navigation": "carto-positron",
                "Dark Mode": "carto-darkmatter",
            }
            selected_style = plotly_styles.get(view_mode, "carto-darkmatter")

            fig = go.Figure()
            fig.add_trace(
                go.Scattermapbox(
                    lat=zone_stats_3d["lat"],
                    lon=zone_stats_3d["lon"],
                    mode="markers",
                    text=zone_stats_3d["zoneName"],
                    customdata=customdata,
                    marker=dict(
                        size=zone_stats_3d["size"],
                        sizemode="diameter",
                        symbol=marker_symbols,
                        color=zone_stats_3d["lateRate"],
                        # Enhanced green -> yellow -> red gradient for late rate
                        colorscale=[
                            [0, "#00e676"],      # Bright green (0% late)
                            [0.15, "#66ff66"],   # Light green (15% late)
                            [0.25, "#fdd835"],   # Yellow (25% late)
                            [0.35, "#ffb300"],   # Amber (35% late)
                            [0.5, "#ff6f00"],    # Orange (50% late)
                            [0.7, "#e64a19"],    # Deep orange (70% late)
                            [1, "#c62828"]       # Dark red (100% late)
                        ],
                        cmin=float(zone_stats_3d["lateRate"].min()),
                        cmax=float(zone_stats_3d["lateRate"].max()),
                        showscale=True,
                        colorbar=dict(
                            title="Late Rate",
                            tickformat=".0%",
                            bgcolor="rgba(0,0,0,0.5)",
                            bordercolor="rgba(186,104,200,0.6)",
                            borderwidth=2,
                            thickness=20,
                            len=0.6,
                        ),
                        opacity=0.95,
                    ),
                    hovertemplate=hover_template,
                    hoverlabel=dict(
                        bgcolor="rgba(6,19,38,0.94)",
                        font=dict(color="#ecf6ff", family="Inter", size=13),
                        bordercolor="rgba(0,198,255,0.65)",
                        align="left",
                    ),
                    name="Zones",
                )
            )

            # Animated pulse/glow effect on hover (size increases with animation speed)
            fig.add_trace(
                go.Scattermapbox(
                    lat=zone_stats_3d["lat"],
                    lon=zone_stats_3d["lon"],
                    mode="markers",
                    marker=dict(
                        size=zone_stats_3d["size"] + animation_speed * 8,
                        color="rgba(186,104,200,0.25)",  # Purple glow matching tooltip theme
                        sizemode="diameter",
                    ),
                    hoverinfo="skip",
                    showlegend=False,
                    name="Hover Glow",
                )
            )

            # Second outer glow layer for enhanced visibility
            fig.add_trace(
                go.Scattermapbox(
                    lat=zone_stats_3d["lat"],
                    lon=zone_stats_3d["lon"],
                    mode="markers",
                    marker=dict(
                        size=zone_stats_3d["size"] + animation_speed * 14,
                        color="rgba(156,39,176,0.12)",  # Softer purple outer glow
                        sizemode="diameter",
                    ),
                    hoverinfo="skip",
                    showlegend=False,
                    name="Outer Glow",
                )
            )

            # Add 3D column markers for top zones (visualizing trip density as building heights)
            top_density_zones = zone_stats_3d.nlargest(15, "trips")
            if not top_density_zones.empty:
                # Create vertical columns using multiple stacked markers
                for _, zone in top_density_zones.iterrows():
                    # Calculate column height based on trip volume (scaled for visual appeal)
                    normalized_height = (zone["trips"] - min_trips) / (max_trips - min_trips) if (max_trips - min_trips) > 0 else 0.5
                    num_segments = int(3 + normalized_height * building_height / 5)  # 3 to ~25 segments based on slider

                    column_lats = []
                    column_lons = []
                    column_sizes = []
                    column_colors = []

                    for i in range(num_segments):
                        # Stack segments vertically by slightly offsetting latitude
                        # This creates a pseudo-3D column effect
                        offset = (i * 0.0003)  # Small offset to create stacking effect
                        column_lats.append(zone["lat"] + offset)
                        column_lons.append(zone["lon"])

                        # Size decreases as we go up (tapered column)
                        size = max(8, zone["size"] * (1 - i / (num_segments * 1.5)))
                        column_sizes.append(size)

                        # Color intensity decreases with height (fade effect)
                        alpha = max(0.2, 0.9 - i / num_segments)
                        late_rate = float(zone['lateRate'])
                        if late_rate <= 0.15:
                            color = f"rgba(46,204,113,{alpha})"
                        elif late_rate <= 0.25:
                            color = f"rgba(241,196,15,{alpha})"
                        elif late_rate <= 0.35:
                            color = f"rgba(230,126,34,{alpha})"
                        else:
                            color = f"rgba(231,76,60,{alpha})"
                        column_colors.append(color)

                    if column_lats:
                        fig.add_trace(
                            go.Scattermapbox(
                                lat=column_lats,
                                lon=column_lons,
                                mode="markers",
                                marker=dict(
                                    size=column_sizes,
                                    color=column_colors,
                                    sizemode="diameter",
                                ),
                                hoverinfo="skip",
                                showlegend=False,
                                name=f"Column {zone['zoneName']}",
                            )
                        )

            hero_df = zone_stats_3d[zone_stats_3d["zoneName"].isin(hero_names)]
            if not hero_df.empty:
                fig.add_trace(
                    go.Scattermapbox(
                        lat=hero_df["lat"],
                        lon=hero_df["lon"],
                        mode="markers+text",
                        marker=dict(size=hero_df["size"] * 0.6, color="rgba(255,215,0,0.75)", symbol="star"),
                        text=hero_df["zoneName"],
                        textfont=dict(color="#ffd369", size=12, family="Inter"),
                        textposition="top center",
                        hoverinfo="skip",
                        name="Landmarks",
                    )
                )

            if show_routes and not corridor_stats.empty:
                route_lat: List[Optional[float]] = []
                route_lon: List[Optional[float]] = []
                route_text: List[Optional[str]] = []
                route_colors: List[str] = []
                route_widths: List[float] = []

                top_routes = corridor_stats.sort_values("trips", ascending=False).head(20)

                # Calculate route properties based on trips and late rate
                max_route_trips = top_routes["trips"].max() or 1

                # Get filter info for tooltips
                month_label = month_labels.get(month_key, month_key)
                hour_range_str = f"{hour_range[0]:02d}:00 - {hour_range[1]:02d}:00"

                for _, route in top_routes.iterrows():
                    start_zone = zone_lookup.get(int(route["from"]), {}).get("properties", {})
                    end_zone = zone_lookup.get(int(route["to"]), {}).get("properties", {})
                    if start_zone.get("centroid_lat") and end_zone.get("centroid_lat"):
                        # Color based on late rate (green -> yellow -> red)
                        late_rate = float(route['lateRate'])
                        if late_rate <= 0.15:
                            color = "rgba(46,204,113,0.8)"  # green
                        elif late_rate <= 0.25:
                            color = "rgba(241,196,15,0.8)"  # yellow
                        elif late_rate <= 0.35:
                            color = "rgba(230,126,34,0.8)"  # orange
                        else:
                            color = "rgba(231,76,60,0.9)"   # red

                        # Width based on trip volume
                        width = 2 + (route['trips'] / max_route_trips) * 8

                        # Build professional tooltip with all details
                        text_value = _build_route_tooltip(
                            from_zone=str(route['fromZone']),
                            to_zone=str(route['toZone']),
                            from_borough=str(route['fromBorough']),
                            to_borough=str(route['toBorough']),
                            trips=float(route['trips']),
                            late_rate=float(route['lateRate']),
                            p50=float(route['p50']),
                            p90=float(route['p90']),
                            month_label=month_label,
                            hour_range=hour_range_str,
                        )

                        route_lat.extend([start_zone["centroid_lat"], end_zone["centroid_lat"], None])
                        route_lon.extend([start_zone["centroid_lon"], end_zone["centroid_lon"], None])
                        route_text.extend([text_value, text_value, None])

                if route_lat:
                    # Add enhanced arc-style corridors with gradient colors
                    fig.add_trace(
                        go.Scattermapbox(
                            lat=route_lat,
                            lon=route_lon,
                            mode="lines",
                            line=dict(width=5, color="rgba(255,140,0,0.85)"),
                            opacity=0.8,
                            hoverinfo="text",
                            text=route_text,
                            name="🚦 Corridors",
                        )
                    )

                    # Add start/end point markers (pins)
                    start_lats = []
                    start_lons = []
                    end_lats = []
                    end_lons = []
                    start_texts = []
                    end_texts = []

                    for _, route in top_routes.iterrows():
                        start_zone = zone_lookup.get(int(route["from"]), {}).get("properties", {})
                        end_zone = zone_lookup.get(int(route["to"]), {}).get("properties", {})
                        if start_zone.get("centroid_lat") and end_zone.get("centroid_lat"):
                            start_lats.append(start_zone["centroid_lat"])
                            start_lons.append(start_zone["centroid_lon"])
                            end_lats.append(end_zone["centroid_lat"])
                            end_lons.append(end_zone["centroid_lon"])
                            start_texts.append(f"🟢 START: {route['fromZone']}")
                            end_texts.append(f"🔴 END: {route['toZone']}")

                    # Start point markers (green pins)
                    fig.add_trace(
                        go.Scattermapbox(
                            lat=start_lats,
                            lon=start_lons,
                            mode="markers",
                            marker=dict(size=14, color="rgba(46,204,113,0.95)", symbol="marker"),
                            text=start_texts,
                            hoverinfo="text",
                            showlegend=False,
                        )
                    )

                    # End point markers (red flags)
                    fig.add_trace(
                        go.Scattermapbox(
                            lat=end_lats,
                            lon=end_lons,
                            mode="markers",
                            marker=dict(size=14, color="rgba(231,76,60,0.95)", symbol="marker"),
                            text=end_texts,
                            hoverinfo="text",
                            showlegend=False,
                        )
                    )

            fig.update_layout(
                hovermode="closest",
                mapbox=dict(
                    style=selected_style,
                    center=dict(lat=40.7128, lon=-73.94),
                    zoom=11.5,
                    pitch=56,
                    bearing=18,
                ),
                margin=dict(l=0, r=0, t=0, b=0),
                legend=dict(
                    orientation="h",
                    yanchor="bottom",
                    y=0.02,
                    xanchor="center",
                    x=0.5,
                    bgcolor="rgba(3,11,24,0.6)",
                    bordercolor="rgba(0,198,255,0.35)",
                    borderwidth=1,
                ),
            )

            fig.update_traces(
                selector=dict(name="Zones"),
                hovertemplate=hover_template,
                hoverlabel=dict(
                    bgcolor="rgba(6,19,38,0.94)",
                    font=dict(color="#ecf6ff", family="Inter", size=13),
                    bordercolor="rgba(0,198,255,0.65)",
                    align="left",
                ),
            )

            st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

            # Add live stats below the 3D map
            st.markdown("#### 🔥 Live 3D Insights")
            insight_cols = st.columns(4)
            max_trips_zone = zone_stats_3d.loc[zone_stats_3d["trips"].idxmax()]
            max_late_zone = zone_stats_3d.loc[zone_stats_3d["lateRate"].idxmax()]

            insight_cols[0].metric(
                "🏗️ Tallest Building",
                max_trips_zone["zoneName"],
                f"{_format_trips(max_trips_zone['trips'])} trips"
            )
            insight_cols[1].metric(
                "🚨 Hottest Zone",
                max_late_zone["zoneName"],
                f"{_format_percent(max_late_zone['lateRate'])} late"
            )
            insight_cols[2].metric(
                "🌆 Active Zones",
                len(zone_stats_3d),
                f"Showing {len(zone_stats_3d)} districts"
            )
            insight_cols[3].metric(
                "📊 Data Density",
                f"{zone_stats_3d['trips'].sum():.0f}",
                "Total trip volume"
            )

    with tab_live:
        st.markdown("### 🔴 Live Promise Monitor")
        st.markdown("*Real-time simulation of NYC traffic patterns*")

        # Live simulation controls
        sim_cols = st.columns(4)
        with sim_cols[0]:
            auto_refresh = st.checkbox("Auto Refresh", value=False)
        with sim_cols[1]:
            refresh_interval = st.selectbox("Refresh Rate (seconds)", [1, 2, 5, 10], index=1)
        with sim_cols[2]:
            simulation_mode = st.selectbox("Simulation", ["Rush Hour", "Normal", "Late Night", "Weather Event"])
        with sim_cols[3]:
            manual_refresh = st.button("🔄 Refresh Now")

        # Initialize session state for live data
        if "live_iteration" not in st.session_state:
            st.session_state.live_iteration = 0

        # Update iteration counter when refresh is triggered
        if auto_refresh or manual_refresh:
            st.session_state.live_iteration += 1

        # Create animated/live view
        zone_stats_live = zone_stats.copy()
        if not zone_stats_live.empty:
            # Use iteration counter for smooth animation without full page reloads
            time_factor = time.time()

            # Smooth but noticeable variations using trigonometric functions
            zone_stats_live["live_trips"] = zone_stats_live["trips"] * (0.85 + 0.3 * np.sin(time_factor * 1.2))
            zone_stats_live["live_late_rate"] = zone_stats_live["lateRate"] * (0.9 + 0.2 * np.cos(time_factor * 0.8))
            zone_stats_live["pulse"] = 0.8 + 0.4 * np.sin(time_factor * 3 + zone_stats_live.index * 0.2)

            # Enhanced live 3D visualization
            live_3d_data = zone_stats_live.copy()
            live_3d_data["lat"] = live_3d_data["zoneId"].apply(
                lambda z: zone_lookup.get(int(z), {}).get("properties", {}).get("centroid_lat")
            )
            live_3d_data["lon"] = live_3d_data["zoneId"].apply(
                lambda z: zone_lookup.get(int(z), {}).get("properties", {}).get("centroid_lon")
            )
            live_3d_data = live_3d_data.dropna(subset=["lat", "lon"])

            if not live_3d_data.empty:
                live_3d_data["live_trips_fmt"] = live_3d_data["live_trips"].apply(lambda x: f"{int(x):,}")
                live_3d_data["live_late_pct"] = (live_3d_data["live_late_rate"] * 100).round(1)

                live_min = float(live_3d_data["live_trips"].min() or 0)
                live_max = float(live_3d_data["live_trips"].max() or 1)
                if live_max - live_min <= 0:
                    live_3d_data["size"] = 24
                else:
                    live_3d_data["size"] = np.interp(
                        live_3d_data["live_trips"],
                        (live_min, live_max),
                        (18, 58),
                    )

                # Enhanced live tooltip with pulsing effect styling
                hover_template_live = (
                    "<div style='background:linear-gradient(145deg,#4a148c,#7b1fa2,#c2185b);"
                    " padding:16px 20px; border-radius:16px; "
                    " box-shadow:0 8px 32px rgba(0,0,0,0.7), 0 0 24px rgba(255,105,180,0.6);"
                    " border:2px solid rgba(255,161,255,0.9); min-width:240px;"
                    " animation:pulse 2s infinite;'>"
                    "<div style='font-size:17px;font-weight:700;margin-bottom:8px;color:#fff0ff;"
                    " text-shadow:0 2px 6px rgba(0,0,0,0.5);'>🔴 LIVE • %{text}</div>"
                    "<div style='font-size:13px;color:#fce4ec;line-height:1.8;'>"
                    "⚡ <b style='color:#f8bbd0;'>Live Trips:</b> <span style='color:#ffeb3b;font-weight:600;'>%{customdata[0]}</span><br>"
                    "🚨 <b style='color:#f8bbd0;'>Live Late Rate:</b> <span style='color:#ff5252;font-weight:600;'>%{customdata[1]}%</span><br>"
                    "🏢 <b style='color:#f8bbd0;'>Borough:</b> <span style='color:#80deea;'>%{customdata[2]}</span>"
                    "</div>"
                    "<div style='margin-top:10px;padding-top:8px;border-top:1px solid rgba(255,161,255,0.5);"
                    " font-size:11px;color:#f8bbd0;font-style:italic;'>🔴 Real-time monitoring</div>"
                    "</div><extra></extra>"
                )

                live_customdata = live_3d_data[["live_trips_fmt", "live_late_pct", "borough"]].to_numpy()

                import plotly.graph_objects as go

                live_fig = go.Figure()
                live_fig.add_trace(
                    go.Scattermapbox(
                        lat=live_3d_data["lat"],
                        lon=live_3d_data["lon"],
                        mode="markers",
                        text=live_3d_data["zoneName"],
                        customdata=live_customdata,
                        marker=dict(
                            size=live_3d_data["size"],
                            sizemode="diameter",
                            color=live_3d_data["live_late_rate"],
                            # Enhanced gradient for live view
                            colorscale=[
                                [0, "#00e676"],      # Bright green
                                [0.2, "#76ff03"],    # Light green
                                [0.4, "#ffeb3b"],    # Yellow
                                [0.6, "#ff9800"],    # Orange
                                [0.8, "#ff5722"],    # Deep orange
                                [1, "#d32f2f"]       # Red
                            ],
                            cmin=float(live_3d_data["live_late_rate"].min()),
                            cmax=float(live_3d_data["live_late_rate"].max()),
                            showscale=True,
                            colorbar=dict(
                                title="Live Late",
                                tickformat=".0%",
                                bgcolor="rgba(0,0,0,0.6)",
                                bordercolor="rgba(255,161,255,0.7)",
                                borderwidth=2,
                                thickness=20,
                                len=0.6,
                            ),
                            opacity=0.95,
                            symbol="marker",
                        ),
                        hovertemplate=hover_template_live,
                        hoverlabel=dict(bgcolor="rgba(40,18,74,0.95)", font=dict(color="#fff0ff", family="Inter")),
                        name="Live Zones",
                    )
                )

                live_fig.add_trace(
                    go.Scattermapbox(
                        lat=live_3d_data["lat"],
                        lon=live_3d_data["lon"],
                        mode="markers",
                        marker=dict(
                            size=live_3d_data["size"] + live_3d_data["pulse"] * 18,
                            color="rgba(255,105,180,0.25)",
                            sizemode="diameter",
                        ),
                        hoverinfo="skip",
                        showlegend=False,
                    )
                )

                if show_routes and not corridor_stats.empty:
                    live_route_lat: List[Optional[float]] = []
                    live_route_lon: List[Optional[float]] = []
                    live_route_text: List[Optional[str]] = []
                    top_routes = corridor_stats.sort_values("trips", ascending=False).head(10)

                    # Get filter info for live tooltips
                    month_label = month_labels.get(month_key, month_key)
                    hour_range_str = f"{hour_range[0]:02d}:00 - {hour_range[1]:02d}:00"

                    for _, route in top_routes.iterrows():
                        start_zone = zone_lookup.get(int(route["from"]), {}).get("properties", {})
                        end_zone = zone_lookup.get(int(route["to"]), {}).get("properties", {})
                        if start_zone.get("centroid_lat") and end_zone.get("centroid_lat"):
                            # Build professional tooltip with live badge
                            base_tooltip = _build_route_tooltip(
                                from_zone=str(route['fromZone']),
                                to_zone=str(route['toZone']),
                                from_borough=str(route['fromBorough']),
                                to_borough=str(route['toBorough']),
                                trips=float(route['trips']),
                                late_rate=float(route['lateRate']),
                                p50=float(route['p50']),
                                p90=float(route['p90']),
                                month_label=f"🔴 LIVE • {month_label}",
                                hour_range=hour_range_str,
                            )

                            live_route_lat.extend([start_zone["centroid_lat"], end_zone["centroid_lat"], None])
                            live_route_lon.extend([start_zone["centroid_lon"], end_zone["centroid_lon"], None])
                            live_route_text.extend([base_tooltip, base_tooltip, None])

                    if live_route_lat:
                        live_fig.add_trace(
                            go.Scattermapbox(
                                lat=live_route_lat,
                                lon=live_route_lon,
                                mode="lines",
                                line=dict(width=4, color="rgba(255,161,255,0.7)"),
                                opacity=0.75,
                                hoverinfo="text",
                                text=live_route_text,
                                name="🔴 Live Corridors",
                            )
                        )

                live_fig.update_layout(
                    hovermode="closest",
                    mapbox=dict(
                        style="carto-darkmatter",
                        center=dict(lat=40.7128, lon=-73.94),
                        zoom=11.4,
                        pitch=58,
                        bearing=(time_factor * 12) % 360,
                    ),
                    margin=dict(l=0, r=0, t=0, b=0),
                    legend=dict(
                        orientation="h",
                        yanchor="bottom",
                        y=0.02,
                        xanchor="center",
                        x=0.5,
                        bgcolor="rgba(24,10,45,0.65)",
                        bordercolor="rgba(255,161,255,0.45)",
                        borderwidth=1,
                    ),
                )

                live_fig.update_traces(
                    selector=dict(name="Live Zones"),
                    hovertemplate=hover_template_live,
                    hoverlabel=dict(
                        bgcolor="rgba(40,18,74,0.95)",
                        font=dict(color="#fff0ff", family="Inter", size=13),
                        bordercolor="rgba(255,161,255,0.55)",
                        align="left",
                    ),
                )

                st.plotly_chart(live_fig, use_container_width=True, config={"displayModeBar": False})

                # Live metrics dashboard
                st.markdown("#### ⚡ Live Performance Dashboard")
                live_cols = st.columns(5)

                current_avg_late = live_3d_data["live_late_rate"].mean()
                current_total_trips = live_3d_data["live_trips"].sum()
                hottest_zone = live_3d_data.loc[live_3d_data["live_late_rate"].idxmax()]

                live_cols[0].metric(
                    "🚨 Current Late Rate",
                    f"{current_avg_late:.1%}",
                    f"{(current_avg_late - weighted_late)*100:.1f}%"
                )
                live_cols[1].metric(
                    "🚕 Active Trips",
                    f"{current_total_trips:.0f}",
                    "Live monitoring"
                )
                live_cols[2].metric(
                    "🔥 Hottest Zone",
                    hottest_zone["zoneName"][:15] + "..." if len(hottest_zone["zoneName"]) > 15 else hottest_zone["zoneName"],
                    f"{hottest_zone['live_late_rate']:.1%}"
                )
                live_cols[3].metric(
                    "⏰ Time",
                    pd.Timestamp.now().strftime("%H:%M:%S"),
                    "Live updates"
                )
                live_cols[4].metric(
                    "📊 Data Points",
                    len(live_3d_data),
                    "Active zones"
                )

                # Live alerts
                if current_avg_late > weighted_late * 1.1:
                    st.error("🚨 **HIGH ALERT**: Late rates are 10%+ above normal!")
                elif current_avg_late > weighted_late * 1.05:
                    st.warning("⚠️ **CAUTION**: Late rates are elevated")
                else:
                    st.success("✅ **NORMAL**: Performance within expected range")

        else:
            st.info("No live data available. Adjust filters to see simulation.")

        # Auto-refresh with controlled timing
        if auto_refresh:
            time.sleep(refresh_interval)
            st.rerun()

    zone_count = int(filtered_full_window["PULocationID"].nunique())
    corridor_count = int(
        filtered_full_window[["PULocationID", "DOLocationID"]].drop_duplicates().shape[0]
    )
    record_count = int(filtered_for_map.shape[0])
    _render_insight_chips(zone_count, corridor_count, record_count)


    st.markdown("---")

    top_left = zone_stats.head(8)
    st.markdown("### 🔥 Hottest Pickup Zones")
    st.dataframe(
        top_left[["zoneName", "borough", "trips", "lateRate", "p50", "p90"]]
        .rename(
            columns={
                "zoneName": "Pickup Zone",
                "borough": "Borough",
                "trips": "Trips",
                "lateRate": "Late Rate",
                "p50": "P50 ETA",
                "p90": "P90 ETA",
            }
        )
        .assign(
            Trips=lambda df_: df_["Trips"].map(_format_trips),
            **{"Late Rate": lambda df_: df_["Late Rate"].map(_format_percent)},
            **{"P50 ETA": lambda df_: df_["P50 ETA"].map(_format_minutes)},
            **{"P90 ETA": lambda df_: df_["P90 ETA"].map(_format_minutes)},
        ),
        use_container_width=True,
        hide_index=True,
    )

    st.markdown("### 🚦 Top Risky Corridors")
    top_corridors = corridor_stats.sort_values(["lateRate", "trips"], ascending=[False, False]).head(10)
    if top_corridors.empty:
        st.info("No corridor meets the minimum trip threshold for the selected filters.")
    else:
        st.dataframe(
            top_corridors[
                [
                    "fromZone",
                    "toZone",
                    "fromBorough",
                    "toBorough",
                    "trips",
                    "lateRate",
                    "p90",
                ]
            ]
            .rename(
                columns={
                    "fromZone": "Pickup Zone",
                    "toZone": "Drop-off Zone",
                    "fromBorough": "Pickup Borough",
                    "toBorough": "Drop-off Borough",
                    "trips": "Trips",
                    "lateRate": "Late Rate",
                    "p90": "P90 ETA",
                }
            )
            .assign(
                Trips=lambda df_: df_["Trips"].map(_format_trips),
                **{"Late Rate": lambda df_: df_["Late Rate"].map(_format_percent)},
                **{"P90 ETA": lambda df_: df_["P90 ETA"].map(_format_minutes)},
            ),
            use_container_width=True,
            hide_index=True,
        )

    st.markdown("---")

    chart_cols = st.columns(2)

    hourly_df = (
        _filter_trips(dataset, month_key, hour_range, boroughs, weather, apply_hour_filter=False)
        .groupby("hour", as_index=False)
        .agg(trips=("trips", "sum"), lateRate=("late_rate", "mean"))
        .sort_values("hour")
    )
    if not hourly_df.empty:
        fig_hour = px.line(
            hourly_df,
            x="hour",
            y="trips",
            markers=True,
            title="Hour-by-Hour Trip Volume",
            labels={"hour": "Hour", "trips": "Trips"},
            template="plotly_dark",
        )
        fig_hour.update_layout(
            plot_bgcolor="rgba(7,26,51,0.9)", paper_bgcolor="rgba(7,26,51,0.0)",
            margin=dict(l=30, r=20, t=60, b=30),
        )
        chart_cols[0].plotly_chart(fig_hour, use_container_width=True)

    borough_df = (
        _filter_trips(dataset, month_key, hour_range, boroughs, weather, apply_hour_filter=False)
        .groupby("pickup_borough", as_index=False)
        .agg(
            trips=("trips", "sum"),
            lateRate=("late_rate", "mean"),
            p90=("p90_eta", "mean"),
        )
        .sort_values("trips", ascending=False)
    )
    if not borough_df.empty:
        fig_borough = px.bar(
            borough_df,
            x="pickup_borough",
            y="trips",
            color="lateRate",
            title="Borough Comparison • Trips & Late Rate",
            labels={"pickup_borough": "Pickup Borough", "trips": "Trips", "lateRate": "Late Rate"},
            color_continuous_scale=["#1abc9c", "#f6c744", "#e67e22", "#c0392b"],
            template="plotly_dark",
        )
        fig_borough.update_layout(
            plot_bgcolor="rgba(7,26,51,0.9)", paper_bgcolor="rgba(7,26,51,0.0)",
            margin=dict(l=30, r=20, t=60, b=30),
        )
        chart_cols[1].plotly_chart(fig_borough, use_container_width=True)

    policy_df = _prepare_policy_chart_data(
    dataset.policy_curves,
    filtered_full_window["PULocationID"].unique(),
    filtered_full_window,
)
    if not policy_df.empty:
        fig_policy = px.area(
            policy_df.sort_values("percentile"),
            x="percentile",
            y="late_rate",
            title="Promise Curve • Average Late Rate by Percentile",
            labels={"percentile": "Promised Percentile", "late_rate": "Late Rate"},
            template="plotly_dark",
        )
        fig_policy.update_layout(
            plot_bgcolor="rgba(7,26,51,0.9)", paper_bgcolor="rgba(7,26,51,0.0)",
            margin=dict(l=30, r=20, t=60, b=30),
        )
        st.plotly_chart(fig_policy, use_container_width=True)
    else:
        st.info("Policy curve data is not available for the selected filters.")

st.markdown(
    """
    ---
    **How to run this app locally**

    ```bash
    streamlit run src/streamlit_app.py
    ```

    Optional environment overrides before launching:

    ```bash
    export PROMISE_DB_SERVER=localhost
    export PROMISE_DB_DATABASE=NYC_Promise_System
    export PROMISE_DB_USERNAME=SA
    export PROMISE_DB_PASSWORD=yourStrong(!)Password
    export PROMISE_DB_PORT=1433
    ```
    """
)
