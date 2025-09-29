# app.py — PromiseWise NYC: Interactive Dashboard (Streamlit)

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from io import BytesIO

# ---------------------------
# Page setup (16:9 wide)
# ---------------------------
st.set_page_config(
    page_title="PromiseWise NYC",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ---------------------------
# Styles
# ---------------------------
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(90deg, #2c3e50 0%, #34495e 100%);
        color: white; padding: 1rem 1.2rem; border-radius: 12px;
        text-align: center; margin-bottom: 1.2rem;
    }
    .kpi-card {
        background: white; padding: 1.0rem 1.2rem; border-radius: 12px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.08);
        border-left: 5px solid #F7C948; height: 100%;
    }
    .metric-value { font-size: 2.2rem; font-weight: 700; color: #2c3e50; }
    .metric-label { font-size: 0.95rem; color: #7f8c8d; margin-top: 0.35rem; }
</style>
""", unsafe_allow_html=True)

# ---------------------------
# Data loader (CSV → fallback synthetic)
# Expected columns (semantic): 
# month, pickup_hour, pickup_borough, pickup_zone, dropoff_zone,
# weather_wet (bool), is_weekend (bool), eta_p50, eta_p90, delay_flag (0/1 optional)
# ---------------------------
@st.cache_data
def load_or_generate():
    # 1) Try real serving table (replace name if needed)
    csv_candidates = [
        "nyc_delivery_serving_table_sample.csv",
        "serving_table.csv",
        "model_ready_serving.csv"
    ]
    for path in csv_candidates:
        try:
            df = pd.read_csv(path)
            # best-effort semantic mapping (renames if your headers differ)
            cols = {c.lower().strip(): c for c in df.columns}
            def pick(*keys, default=None):
                for k in keys:
                    if k in cols: return cols[k]
                return default
            # Create minimal expected columns if possible
            rename_map = {}
            if pick("month"): rename_map[cols["month"]] = "month"
            if pick("pickup_hour","hour"): rename_map[cols.get("pickup_hour","hour")] = "pickup_hour"
            if pick("pickup_borough","pu_borough","borough"): rename_map[cols.get("pickup_borough",cols.get("pu_borough","borough"))] = "pickup_borough"
            if pick("pickup_zone","pu_zone"): rename_map[cols.get("pickup_zone",cols.get("pu_zone"))] = "pickup_zone"
            if pick("dropoff_zone","do_zone"): rename_map[cols.get("dropoff_zone",cols.get("do_zone"))] = "dropoff_zone"
            if pick("weather_wet","is_wet","wet"): rename_map[cols.get("weather_wet",cols.get("is_wet","wet"))] = "weather_wet"
            if pick("is_weekend","weekend"): rename_map[cols.get("is_weekend","weekend")] = "is_weekend"
            if pick("eta_p50","p50","pred_eta_p50"): rename_map[cols.get("eta_p50",cols.get("p50","pred_eta_p50"))] = "eta_p50"
            if pick("eta_p90","p90","pred_eta_p90"): rename_map[cols.get("eta_p90",cols.get("p90","pred_eta_p90"))] = "eta_p90"
            if pick("delay_flag","delay","is_delay", "late_flag"):
                rename_map[cols.get("delay_flag", cols.get("delay", cols.get("is_delay","late_flag")))] = "delay_flag"

            df = df.rename(columns=rename_map)

            # Minimal sanity: fill missing optional fields if absent
            if "month" not in df.columns:
                # derive month from any datetime if exists
                dt_col = next((c for c in df.columns if "pickup" in c.lower() and "time" in c.lower()), None)
                df["month"] = pd.to_datetime(df[dt_col]).dt.strftime("%b") if dt_col else "Jul"
            for need in ["pickup_hour","pickup_borough","pickup_zone","dropoff_zone","eta_p50"]:
                if need not in df.columns:
                    raise ValueError(f"Required column missing: {need}")

            if "eta_p90" not in df.columns:
                df["eta_p90"] = df["eta_p50"] * 1.6  # safe default
            if "weather_wet" not in df.columns:
                df["weather_wet"] = False
            if "is_weekend" not in df.columns:
                df["is_weekend"] = False
            if "delay_flag" not in df.columns:
                # if you don't have labels, create a light proxy (10%)
                df["delay_flag"] = (np.random.rand(len(df)) < 0.10).astype(int)

            # Normalize dtypes
            df["pickup_hour"] = pd.to_numeric(df["pickup_hour"], errors="coerce").fillna(0).astype(int).clip(0,23)
            df["weather_wet"] = df["weather_wet"].astype(bool)
            df["is_weekend"] = df["is_weekend"].astype(bool)
            return df, f"Loaded: {path}"
        except Exception:
            continue

    # 2) Synthetic fallback (demo)
    np.random.seed(42)
    n = 10000
    months = np.random.choice(["May","Jun","Jul"], n, p=[0.33,0.33,0.34])
    hour_probs = np.array([0.02,0.01,0.01,0.01,0.02,0.03,0.04,0.05,0.06,0.05,0.04,0.05,
                           0.06,0.05,0.04,0.05,0.06,0.08,0.09,0.08,0.06,0.05,0.04,0.03])
    hour_probs /= hour_probs.sum()
    hours = np.random.choice(range(24), n, p=hour_probs)
    boroughs = np.random.choice(["Manhattan","Queens","Brooklyn","Bronx","Staten Island"], n,
                                p=[0.65,0.15,0.12,0.06,0.02])
    wet = np.random.choice([True, False], n, p=[0.2, 0.8])
    weekend = np.random.choice([True, False], n, p=[0.25, 0.75])

    base = np.random.gamma(2, 7, n)  # ~14 min
    b_adj = np.select(
        [boroughs=="Queens", boroughs=="Brooklyn", boroughs=="Bronx"],
        [15, 8, 12], default=0
    )
    h_adj = np.where((hours>=17)&(hours<=19), np.random.uniform(3,5,n), 0)
    w_adj = np.where(wet, 0.5, 0)
    we_adj = np.where(weekend, -1.5, 0)
    p50 = np.maximum(base + b_adj + h_adj + w_adj + we_adj, 2)
    p90 = p50 * 1.6
    delay = (np.random.rand(n) < 0.10).astype(int)

    manhattan = ['Midtown Center','Times Square','Upper East Side','SoHo','Financial District']
    queens = ['JFK Airport','LaGuardia Airport','Astoria','Flushing']
    brooklyn = ['Williamsburg','Park Slope','DUMBO','Brooklyn Heights']
    pu = []; do = []
    for b in boroughs:
        if b == "Manhattan":
            pu.append(np.random.choice(manhattan)); do.append(np.random.choice(manhattan))
        elif b == "Queens":
            pu.append(np.random.choice(queens)); do.append(np.random.choice(manhattan+queens))
        else:
            pu.append(f"{b} Zone"); do.append(np.random.choice(manhattan))

    df = pd.DataFrame({
        "month": months, "pickup_hour": hours, "pickup_borough": boroughs,
        "pickup_zone": pu, "dropoff_zone": do, "weather_wet": wet,
        "is_weekend": weekend, "eta_p50": p50, "eta_p90": p90, "delay_flag": delay
    })
    return df, "Synthetic demo data (fallback)"

data, source_msg = load_or_generate()

# ---------------------------
# Header
# ---------------------------
st.markdown("""
<div class="main-header">
  <h1>🚕 PromiseWise NYC — Weather-aware ETA & Delay Risk</h1>
  <p>Interactive promise strategy using public NYC trip signals</p>
</div>
""", unsafe_allow_html=True)
st.caption(f"Data source: {source_msg}")

# ---------------------------
# Sidebar filters
# ---------------------------
st.sidebar.header("📊 Controls")

month = st.sidebar.selectbox("Month", ["May","Jun","Jul"], index=2)
hour_range = st.sidebar.slider("Hour Range", 0, 23, (0,23))
boroughs = ["Manhattan","Queens","Brooklyn","Bronx","Staten Island"]
sel_boroughs = st.sidebar.multiselect("Pickup Borough", boroughs, default=boroughs)
show_dry = st.sidebar.checkbox("Dry Weather", value=True)
show_wet = st.sidebar.checkbox("Wet Weather", value=True)
promise_pct = st.sidebar.slider("Promise Percentile (P)", 50, 95, 90)

# Apply filters
df_f = data.copy()
df_f = df_f[df_f["month"] == month]
df_f = df_f[(df_f["pickup_hour"] >= hour_range[0]) & (df_f["pickup_hour"] <= hour_range[1])]
df_f = df_f[df_f["pickup_borough"].isin(sel_boroughs)]
if show_dry and not show_wet:
    df_f = df_f[~df_f["weather_wet"]]
elif show_wet and not show_dry:
    df_f = df_f[df_f["weather_wet"]]
# else both True → no extra filter

if len(df_f) == 0:
    st.warning("No data for current filters. Adjust selections.")
    st.stop()

# ---------------------------
# KPI calculations
# ---------------------------
# Promise time interpolation: between P50 and P90 (linear for simplicity)
alpha = (promise_pct - 50) / (90 - 50) if promise_pct <= 90 else 1.0 + (promise_pct-90)/10.0*0.5
# cap alpha to 0..1.5 (so P95 ≈ P90 + 50% of (P90-P50))
alpha = float(np.clip(alpha, 0, 1.5))
promise_time_series = df_f["eta_p50"] + alpha * (df_f["eta_p90"] - df_f["eta_p50"])
promise_time = float(np.median(promise_time_series))

median_eta = float(np.median(df_f["eta_p50"]))
median_buffer = max(promise_time - median_eta, 0.0)

# Late-promise rate estimate:
# If you have true labels "delay_flag" (1=exceeds P90), use them to anchor:
#   - at P90: late ≈ mean(delay_flag)
#   - at P50: late ≈ 50%
# interpolate between P50 and P90; beyond P90, taper towards 5%.
p90_anchor = float(df_f["delay_flag"].mean()) * 100.0 if "delay_flag" in df_f.columns else 10.0
if promise_pct <= 90:
    late_rate = 50.0 + (p90_anchor - 50.0) * ((promise_pct - 50) / 40.0)
else:
    # between 90 and 95: glide from p90_anchor to ~5%
    late_rate = p90_anchor + (5.0 - p90_anchor) * ((promise_pct - 90) / 5.0)
late_rate = float(np.clip(late_rate, 0, 100))

# Optional coverage proxy if labels exist
coverage_note = None
if "delay_flag" in df_f.columns and promise_pct >= 90:
    coverage_note = f"Approx. late at P90 ≈ {p90_anchor:.1f}% (label-anchored)"

# ---------------------------
# KPI row
# ---------------------------
k1,k2,k3 = st.columns(3)
with k1:
    st.markdown('<div class="kpi-card"><div class="metric-value">{:.1f} min</div><div class="metric-label">Median ETA</div></div>'.format(median_eta), unsafe_allow_html=True)
with k2:
    st.markdown('<div class="kpi-card"><div class="metric-value">{:.1f}%</div><div class="metric-label">Late-Promise Rate (est.)</div></div>'.format(late_rate), unsafe_allow_html=True)
with k3:
    st.markdown('<div class="kpi-card"><div class="metric-value">{:.1f} min</div><div class="metric-label">Median Buffer vs P50</div></div>'.format(median_buffer), unsafe_allow_html=True)
if coverage_note:
    st.caption(coverage_note)

# ---------------------------
# Charts — row 1
# ---------------------------
c1,c2 = st.columns(2)

with c1:
    st.subheader("📈 Promise Strategy Trade-off")
    # Build curve
    xs = np.array([50,60,70,80,90,95])
    ys = []
    for p in xs:
        if p <= 90:
            y = 50.0 + (p90_anchor - 50.0) * ((p - 50) / 40.0)
        else:
            y = p90_anchor + (5.0 - p90_anchor) * ((p - 90) / 5.0)
        ys.append(float(np.clip(y,0,100)))
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=xs, y=ys, mode="lines+markers",
                             line=dict(color="#3498db", width=3),
                             marker=dict(size=8), name="Late-Promise Rate"))
    fig.add_vline(x=promise_pct, line_dash="dash", line_color="#F7C948")
    fig.update_layout(height=420, xaxis_title="Promised Percentile (P)",
                      yaxis_title="Late-Promise Rate (%)", showlegend=False)
    st.plotly_chart(fig, use_container_width=True)

with c2:
    st.subheader("⏰ Median ETA by Hour (Weekday vs Weekend)")
    g = df_f.groupby(["pickup_hour","is_weekend"])["eta_p50"].median().reset_index()
    fig2 = go.Figure()
    wd = g[g["is_weekend"]==False]
    we = g[g["is_weekend"]==True]
    if len(wd): fig2.add_trace(go.Scatter(x=wd["pickup_hour"], y=wd["eta_p50"], mode="lines+markers",
                                          name="Weekday", line=dict(color="#3498db", width=2)))
    if len(we): fig2.add_trace(go.Scatter(x=we["pickup_hour"], y=we["eta_p50"], mode="lines+markers",
                                          name="Weekend", line=dict(color="#e74c3c", width=2)))
    fig2.update_layout(height=420, xaxis_title="Hour of Day", yaxis_title="Median ETA (min)")
    st.plotly_chart(fig2, use_container_width=True)

# ---------------------------
# Charts — row 2
# ---------------------------
c3,c4 = st.columns(2)

with c3:
    st.subheader("🗺️ Average ETA by Borough × Hour")
    heat = df_f.groupby(["pickup_borough","pickup_hour"])["eta_p50"].mean().reset_index()
    pivot = heat.pivot(index="pickup_borough", columns="pickup_hour", values="eta_p50")
    fig3 = go.Figure(data=go.Heatmap(
        z=pivot.values, x=pivot.columns, y=pivot.index,
        colorscale="RdYlBu_r",
        text=np.round(pivot.values,1), texttemplate="%{text}", textfont={"size":10},
        hovertemplate="Borough: %{y}<br>Hour: %{x}<br>Avg ETA: %{z:.1f} min<extra></extra>"
    ))
    fig3.update_layout(height=420, xaxis_title="Hour", yaxis_title="Pickup Borough")
    st.plotly_chart(fig3, use_container_width=True)

with c4:
    st.subheader("🛣️ Top-10 Riskiest Corridors")
    routes = df_f.groupby(["pickup_zone","dropoff_zone"]).agg(avg_eta=("eta_p50","mean"),
                                                             trip_count=("eta_p50","count")).reset_index()
    top = routes[routes["trip_count"]>=2].nlargest(10,"avg_eta").copy()
    if len(top):
        top["route"] = top["pickup_zone"] + " → " + top["dropoff_zone"]
        fig4 = go.Figure(go.Bar(
            y=top["route"][::-1], x=top["avg_eta"][::-1], orientation="h",
            marker_color="#e74c3c",
            text=np.round(top["avg_eta"][::-1],1), textposition="inside"
        ))
        fig4.update_layout(height=420, xaxis_title="Average ETA (min)", margin=dict(l=200))
        st.plotly_chart(fig4, use_container_width=True)
    else:
        st.info("Not enough routes for current filters.")

# ---------------------------
# Downloads
# ---------------------------
st.markdown("---")
st.subheader("⬇️ Export")

csv_bytes = df_f.to_csv(index=False).encode("utf-8")
st.download_button("Download filtered CSV", data=csv_bytes, file_name="promisewise_filtered.csv", mime="text/csv")

# Attempt to export the trade-off chart as PNG (requires kaleido)
try:
    import plotly.io as pio
    png_bytes = pio.to_image(fig, format="png", width=1280, height=720, scale=2)
    st.download_button("Download Trade-off PNG", data=png_bytes, file_name="tradeoff.png", mime="image/png")
except Exception:
    st.caption("Tip: install `kaleido` to enable PNG export of charts (`pip install -U kaleido`).")