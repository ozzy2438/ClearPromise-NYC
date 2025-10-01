from __future__ import annotations

import json
import math
import os
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd
import plotly.express as px
import pydeck as pdk
import streamlit as st
from sqlalchemy import create_engine, text

try:
    from shapely import wkb, wkt
    from shapely.geometry import shape, mapping
except ImportError:  # pragma: no cover
    wkb = None  # type: ignore
    wkt = None  # type: ignore
    shape = None  # type: ignore
    mapping = None  # type: ignore

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
    """Create a lazily cached SQLAlchemy engine using environment overrides."""
    server = os.getenv("PROMISE_DB_SERVER", "localhost")
    database = os.getenv("PROMISE_DB_DATABASE", "NYC_Promise_System")
    username = os.getenv("PROMISE_DB_USERNAME", "SA")
    password = os.getenv("PROMISE_DB_PASSWORD", "Allah248012!")
    port = int(os.getenv("PROMISE_DB_PORT", "1433"))

    connection_string = (
        f"mssql+pyodbc://{username}:{password}@{server}:{port}/{database}?"
        "driver=ODBC+Driver+17+for+SQL+Server&TrustServerCertificate=yes"
    )
    return create_engine(connection_string)


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

        if (lat is None or lon is None) and geom is not None:
            centroid = geom.centroid
            lat, lon = centroid.y, centroid.x
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
    engine = get_engine()
    with engine.connect() as conn:
        zones_df = pd.read_sql(text("SELECT * FROM dim_zone"), conn)
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

    zone_features = _load_zone_features(zones_df)
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


dataset = load_dataset()

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
        _render_legend(scale)

    with tab_3d:
        st.markdown("### 🌃 3D NYC Promise Cityscape")

        # Enhanced 3D view controls
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            view_mode = st.selectbox("View Style", ["Skyline", "Satellite", "Navigation", "Dark Mode"])
        with col2:
            animation_speed = st.slider("Animation Speed", 0.5, 3.0, 1.5, 0.1)
        with col3:
            building_height = st.slider("Building Scale", 10, 100, 25, 5)
        with col4:
            show_routes = st.checkbox("Show Live Routes", value=True)

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
            zone_stats_3d["color"] = zone_stats_3d["lateRate"].apply(lambda v: _color_rgb(v, scale))
            zone_stats_3d["elevation"] = zone_stats_3d["trips"]
            zone_stats_3d["latePct"] = (zone_stats_3d["lateRate"] * 100).round(1)
            # Add formatted columns for tooltip display
            zone_stats_3d["tripsFormatted"] = zone_stats_3d["trips"].apply(lambda x: f"{int(x):,}")
            zone_stats_3d["p90Formatted"] = zone_stats_3d["p90"].apply(lambda x: f"{x:.1f}")

            # Dynamic view based on selection with enhanced styling
            map_styles = {
                "Skyline": "https://basemaps.cartocdn.com/gl/dark-matter-gl-style/style.json",
                "Satellite": "https://basemaps.cartocdn.com/gl/voyager-gl-style/style.json",
                "Navigation": "https://basemaps.cartocdn.com/gl/positron-gl-style/style.json",
                "Dark Mode": "https://basemaps.cartocdn.com/gl/dark-matter-nolabels-gl-style/style.json"
            }

            # Enhanced map styles for better 3D visualization
            if view_mode == "Navigation":
                # Use a style with better street detail and labels
                selected_style = "https://basemaps.cartocdn.com/gl/voyager-gl-style/style.json"
            else:
                selected_style = map_styles.get(view_mode, map_styles["Skyline"])

            # Enhanced layers with multiple visualizations
            layers = []

            # Add 3D buildings layer for Navigation mode (realistic city buildings)
            if view_mode == "Navigation":
                # Create synthetic 3D building data for NYC
                building_data = []
                for _, row in zone_stats_3d.head(50).iterrows():  # Top 50 zones
                    # Create multiple buildings per zone for realistic city feel
                    for offset in [(0, 0), (0.002, 0), (0, 0.002), (0.002, 0.002)]:
                        building_data.append({
                            "position": [row["lon"] + offset[1], row["lat"] + offset[0]],
                            "height": np.random.randint(50, 300),  # Random building heights
                            "color": [200, 200, 200, 80]
                        })

                if building_data:
                    layers.append(
                        pdk.Layer(
                            "ColumnLayer",
                            data=building_data,
                            get_position="position",
                            get_elevation="height",
                            elevation_scale=1,
                            radius=80,
                            get_fill_color="color",
                            pickable=False,
                            opacity=0.15,
                        )
                    )

            layers.extend([
                # Main column layer for trips (data visualization)
                pdk.Layer(
                    "ColumnLayer",
                    data=zone_stats_3d,
                    get_position=["lon", "lat"],
                    get_elevation="elevation",
                    elevation_scale=building_height,
                    radius=200,
                    get_fill_color="color",
                    pickable=True,
                    auto_highlight=True,
                    opacity=0.8,
                ),
                # Hexagon layer for density visualization
                pdk.Layer(
                    "HexagonLayer",
                    data=zone_stats_3d,
                    get_position=["lon", "lat"],
                    get_weight="trips",
                    radius=300,
                    elevation_scale=building_height * 0.3,
                    pickable=True,
                    opacity=0.3,
                    coverage=0.8,
                ),
            ])

            # Add animated route flows if enabled
            if show_routes and not corridor_stats.empty:
                # Create route flow data
                route_data = []
                top_routes = corridor_stats.sort_values("trips", ascending=False).head(20)
                for _, route in top_routes.iterrows():
                    start_zone = zone_lookup.get(int(route["from"]), {}).get("properties", {})
                    end_zone = zone_lookup.get(int(route["to"]), {}).get("properties", {})
                    if start_zone.get("centroid_lat") and end_zone.get("centroid_lat"):
                        route_data.append({
                            "start": [start_zone["centroid_lon"], start_zone["centroid_lat"]],
                            "end": [end_zone["centroid_lon"], end_zone["centroid_lat"]],
                            "trips": route["trips"],
                            "late_rate": route["lateRate"]
                        })

                if route_data:
                    layers.append(
                        pdk.Layer(
                            "ArcLayer",
                            data=route_data,
                            get_source_position="start",
                            get_target_position="end",
                            get_width="trips",
                            width_scale=0.01,
                            get_source_color=[255, 140, 0, 160],
                            get_target_color=[255, 0, 128, 160],
                            pickable=True,
                            auto_highlight=True,
                        )
                    )

            # Enhanced view state with better camera positioning
            initial_view = pdk.ViewState(
                latitude=40.7128,
                longitude=-73.94,
                zoom=11.5,
                pitch=60,
                bearing=15,
                min_zoom=10,
                max_zoom=16
            )

            deck = pdk.Deck(
                map_style=selected_style,
                initial_view_state=initial_view,
                layers=layers,
                tooltip={
                    "html": """
                    <div style='background: linear-gradient(135deg, #1e3c72, #2a5298); padding: 12px; border-radius: 8px; border: 1px solid #4a90e2;'>
                        <h4 style='margin: 0 0 8px 0; color: #fff; font-size: 16px;'>{zoneName}</h4>
                        <div style='color: #e0e0e0; font-size: 13px;'>
                            <div>🚕 Trips: <strong>{tripsFormatted}</strong></div>
                            <div>⏱️ Late Rate: <strong>{latePct}%</strong></div>
                            <div>📊 P90 ETA: <strong>{p90Formatted} min</strong></div>
                            <div>📍 Borough: <strong>{borough}</strong></div>
                        </div>
                    </div>
                    """,
                    "style": {"backgroundColor": "transparent", "color": "white"},
                },
            )

            st.pydeck_chart(deck, use_container_width=True)

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
                live_3d_data["color"] = live_3d_data["live_late_rate"].apply(lambda v: _color_rgb(v, scale))
                live_3d_data["elevation"] = live_3d_data["live_trips"] * live_3d_data["pulse"]
                # Format columns for tooltip
                live_3d_data["live_trips_fmt"] = live_3d_data["live_trips"].apply(lambda x: f"{int(x):,}")
                live_3d_data["live_late_pct"] = (live_3d_data["live_late_rate"] * 100).round(1)

                # Pulsing effect layers
                live_layers = [
                    # Main pulsing columns
                    pdk.Layer(
                        "ColumnLayer",
                        data=live_3d_data,
                        get_position=["lon", "lat"],
                        get_elevation="elevation",
                        elevation_scale=30,
                        radius=150,
                        get_fill_color="color",
                        pickable=True,
                        opacity=0.9,
                    ),
                    # Scatterplot for hotspots
                    pdk.Layer(
                        "ScatterplotLayer",
                        data=live_3d_data,
                        get_position=["lon", "lat"],
                        get_radius="pulse",
                        radius_scale=200,
                        get_fill_color=[255, 165, 0, 150],
                        opacity=0.6,
                    ),
                ]

                # Dynamic smooth camera movement - like flying through NYC
                bearing = (time_factor * 15) % 360  # Faster rotation
                pitch = 55 + 20 * np.sin(time_factor * 0.3)  # More dramatic pitch changes

                # Add gentle camera position changes for exploration feel
                lat_offset = 0.01 * np.sin(time_factor * 0.4)
                lon_offset = 0.01 * np.cos(time_factor * 0.5)
                zoom_factor = 11.5 + 1.5 * np.sin(time_factor * 0.2)  # Breathing zoom effect

                live_view = pdk.ViewState(
                    latitude=40.7128 + lat_offset,
                    longitude=-73.94 + lon_offset,
                    zoom=zoom_factor,
                    pitch=pitch,
                    bearing=bearing,
                )

                live_deck = pdk.Deck(
                    map_style="https://basemaps.cartocdn.com/gl/dark-matter-gl-style/style.json",
                    initial_view_state=live_view,
                    layers=live_layers,
                    tooltip={
                        "html": """
                        <div style='background: linear-gradient(45deg, #ff6b6b, #4ecdc4); padding: 15px; border-radius: 10px; color: white;'>
                            <h3 style='margin: 0; font-size: 18px;'>🔴 LIVE: {zoneName}</h3>
                            <div style='margin-top: 8px; font-size: 14px;'>
                                <div>⚡ Live Trips: <strong>{live_trips_fmt}</strong></div>
                                <div>🚨 Live Late Rate: <strong>{live_late_pct}%</strong></div>
                                <div>📍 Zone ID: <strong>{zoneId}</strong></div>
                                <div>🏢 Borough: <strong>{borough}</strong></div>
                            </div>
                        </div>
                        """,
                    },
                )

                st.pydeck_chart(live_deck, use_container_width=True)

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
