import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import seaborn as sns
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# Page configuration for 16:9 aspect ratio
st.set_page_config(
    page_title="PromiseWise NYC",
    page_icon="🚚",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling and 16:9 layout
st.markdown("""
<style>
    .main {
        padding-top: 1rem;
    }
    .block-container {
        padding-top: 1rem;
        padding-bottom: 0rem;
        max-width: 100%;
    }
    .metric-card {
        background-color: #f8f9fa;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #F7C948;
    }
    .taxi-yellow {
        color: #F7C948;
        font-weight: bold;
    }
    .success-indicator {
        color: #28a745;
        font-size: 1.2rem;
    }
    .warning-indicator {
        color: #dc3545;
        font-size: 1.2rem;
    }
    h1 {
        color: #2c3e50;
        text-align: center;
        margin-bottom: 0.5rem;
    }
    .subtitle {
        text-align: center;
        color: #7f8c8d;
        margin-bottom: 2rem;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_data
def load_data():
    """Load and preprocess all data files"""
    try:
        # Load trip data
        trips = pd.read_csv('data/sample/yellow_tripdata_sample_10k.csv')
        
        # Load zone lookup
        zones = pd.read_csv('data/sample/taxi_zone_lookup.csv')
        
        # Load weather data
        weather = pd.read_csv('data/sample/nyc_weather_sample.csv')
        
        # Load holidays
        holidays = pd.read_csv('data/sample/us_public_holidays_2025.csv')
        
        return trips, zones, weather, holidays
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return None, None, None, None

@st.cache_data
def preprocess_data(trips, zones, weather, holidays):
    """Preprocess and combine all data"""
    
    # Convert datetime columns
    trips['pickup_datetime'] = pd.to_datetime(trips['tpep_pickup_datetime'])
    trips['dropoff_datetime'] = pd.to_datetime(trips['tpep_dropoff_datetime'])
    
    # Calculate trip duration in minutes
    trips['duration_minutes'] = (trips['dropoff_datetime'] - trips['pickup_datetime']).dt.total_seconds() / 60
    
    # Filter out invalid trips
    trips = trips[
        (trips['duration_minutes'] > 0) & 
        (trips['duration_minutes'] <= 120) &  # Less than 2 hours
        (trips['trip_distance'] > 0) &
        (trips['trip_distance'] <= 50)  # Reasonable distance
    ].copy()
    
    # Extract temporal features
    trips['hour'] = trips['pickup_datetime'].dt.hour
    trips['day_of_week'] = trips['pickup_datetime'].dt.dayofweek
    trips['is_weekend'] = trips['day_of_week'].isin([5, 6])
    trips['month'] = trips['pickup_datetime'].dt.month
    trips['date'] = trips['pickup_datetime'].dt.date
    
    # Add month names
    month_names = {5: 'May', 6: 'Jun', 7: 'Jul'}
    trips['month_name'] = trips['month'].map(month_names)
    
    # Merge with zone data
    zones_pickup = zones.rename(columns={'LocationID': 'PULocationID', 'Borough': 'pickup_borough', 'Zone': 'pickup_zone'})
    zones_dropoff = zones.rename(columns={'LocationID': 'DOLocationID', 'Borough': 'dropoff_borough', 'Zone': 'dropoff_zone'})
    
    trips = trips.merge(zones_pickup[['PULocationID', 'pickup_borough', 'pickup_zone']], on='PULocationID', how='left')
    trips = trips.merge(zones_dropoff[['DOLocationID', 'dropoff_borough', 'dropoff_zone']], on='DOLocationID', how='left')
    
    # Process weather data
    weather['date'] = pd.to_datetime(weather['time']).dt.date
    weather['is_wet'] = weather['prcp'] > 0.1  # More than 0.1mm precipitation
    weather['weather_condition'] = weather['is_wet'].map({True: 'Wet', False: 'Dry'})
    
    # Merge with weather
    trips = trips.merge(weather[['date', 'tavg', 'prcp', 'is_wet', 'weather_condition']], on='date', how='left')
    
    # Fill missing weather data
    trips['is_wet'] = trips['is_wet'].fillna(False)
    trips['weather_condition'] = trips['weather_condition'].fillna('Dry')
    trips['tavg'] = trips['tavg'].fillna(trips['tavg'].median())
    
    # Process holidays
    holidays['date'] = pd.to_datetime(holidays['date']).dt.date
    trips = trips.merge(holidays[['date']].assign(is_holiday=True), on='date', how='left')
    trips['is_holiday'] = trips['is_holiday'].fillna(False)
    
    # Calculate ETA estimates and percentiles
    trips['route'] = trips['pickup_zone'].astype(str) + ' → ' + trips['dropoff_zone'].astype(str)
    
    # Simulate ETA predictions based on historical data
    # Group by similar conditions and calculate percentiles
    for idx, row in trips.iterrows():
        similar_trips = trips[
            (trips['pickup_borough'] == row['pickup_borough']) &
            (trips['dropoff_borough'] == row['dropoff_borough']) &
            (trips['hour'].between(row['hour']-1, row['hour']+1)) &
            (trips['is_weekend'] == row['is_weekend'])
        ]['duration_minutes']
        
        if len(similar_trips) > 10:
            trips.loc[idx, 'eta_p50'] = similar_trips.quantile(0.5)
            trips.loc[idx, 'eta_p90'] = similar_trips.quantile(0.9)
        else:
            # Fallback to overall percentiles
            trips.loc[idx, 'eta_p50'] = trips['duration_minutes'].quantile(0.5)
            trips.loc[idx, 'eta_p90'] = trips['duration_minutes'].quantile(0.9)
    
    return trips

def calculate_promise_metrics(df, promise_percentile=90):
    """Calculate promise-related metrics"""
    if len(df) == 0:
        return {
            'median_eta': 0,
            'late_rate': 0,
            'p90_coverage': 0,
            'total_trips': 0
        }
    
    # Calculate promise time based on percentile
    promise_time = df['duration_minutes'].quantile(promise_percentile/100)
    
    # Calculate metrics
    median_eta = df['duration_minutes'].median()
    late_trips = (df['duration_minutes'] > promise_time).sum()
    late_rate = (late_trips / len(df)) * 100
    
    # P90 coverage (how many trips finish within P90 prediction)
    p90_coverage = ((df['duration_minutes'] <= df['eta_p90']).sum() / len(df)) * 100
    
    return {
        'median_eta': median_eta,
        'late_rate': late_rate,
        'p90_coverage': p90_coverage,
        'total_trips': len(df),
        'promise_time': promise_time
    }

def create_trade_off_curve(df):
    """Create promise percentile vs late rate trade-off curve"""
    percentiles = range(50, 96, 5)
    late_rates = []
    
    for p in percentiles:
        promise_time = df['duration_minutes'].quantile(p/100)
        late_rate = ((df['duration_minutes'] > promise_time).sum() / len(df)) * 100
        late_rates.append(late_rate)
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=percentiles,
        y=late_rates,
        mode='lines+markers',
        name='Late Promise Rate',
        line=dict(color='#F7C948', width=3),
        marker=dict(size=8)
    ))
    
    fig.update_layout(
        title="Promise Strategy Trade-off Curve",
        xaxis_title="Promise Percentile (%)",
        yaxis_title="Late Promise Rate (%)",
        height=400,
        showlegend=False
    )
    
    return fig, percentiles, late_rates

def create_hourly_profile(df):
    """Create hourly ETA profile for weekdays vs weekends"""
    hourly_stats = df.groupby(['hour', 'is_weekend'])['duration_minutes'].median().reset_index()
    
    fig = go.Figure()
    
    # Weekday line
    weekday_data = hourly_stats[hourly_stats['is_weekend'] == False]
    fig.add_trace(go.Scatter(
        x=weekday_data['hour'],
        y=weekday_data['duration_minutes'],
        mode='lines+markers',
        name='Weekday',
        line=dict(color='#3498db', width=2),
        marker=dict(size=6)
    ))
    
    # Weekend line
    weekend_data = hourly_stats[hourly_stats['is_weekend'] == True]
    fig.add_trace(go.Scatter(
        x=weekend_data['hour'],
        y=weekend_data['duration_minutes'],
        mode='lines+markers',
        name='Weekend',
        line=dict(color='#e74c3c', width=2),
        marker=dict(size=6)
    ))
    
    fig.update_layout(
        title="Hourly ETA Profile: Weekday vs Weekend",
        xaxis_title="Hour of Day",
        yaxis_title="Median Trip Duration (min)",
        height=400,
        xaxis=dict(tickmode='linear', tick0=0, dtick=2)
    )
    
    return fig

def create_borough_heatmap(df):
    """Create borough x hour heatmap"""
    heatmap_data = df.groupby(['pickup_borough', 'hour'])['duration_minutes'].mean().reset_index()
    heatmap_pivot = heatmap_data.pivot(index='pickup_borough', columns='hour', values='duration_minutes')
    
    fig = go.Figure(data=go.Heatmap(
        z=heatmap_pivot.values,
        x=heatmap_pivot.columns,
        y=heatmap_pivot.index,
        colorscale='RdYlBu_r',
        showscale=True,
        colorbar=dict(title="Avg Duration (min)")
    ))
    
    fig.update_layout(
        title="Borough × Hour Average Duration Heatmap",
        xaxis_title="Hour of Day",
        yaxis_title="Pickup Borough",
        height=400
    )
    
    return fig

def create_corridor_top10(df, selected_borough=None, selected_hour=None):
    """Create top 10 risky corridors chart"""
    filtered_df = df.copy()
    
    # Apply filters if specified
    if selected_borough:
        filtered_df = filtered_df[filtered_df['pickup_borough'] == selected_borough]
    if selected_hour is not None:
        filtered_df = filtered_df[filtered_df['hour'] == selected_hour]
    
    # Calculate corridor risk (high duration + high variability)
    corridor_stats = filtered_df.groupby('route').agg({
        'duration_minutes': ['mean', 'std', 'count']
    }).reset_index()
    
    corridor_stats.columns = ['route', 'avg_duration', 'duration_std', 'trip_count']
    corridor_stats = corridor_stats[corridor_stats['trip_count'] >= 5]  # Minimum trips
    
    # Calculate risk score
    corridor_stats['risk_score'] = corridor_stats['avg_duration'] * (1 + corridor_stats['duration_std'].fillna(0))
    corridor_stats = corridor_stats.nlargest(10, 'risk_score')
    
    fig = go.Figure(data=go.Bar(
        y=corridor_stats['route'],
        x=corridor_stats['risk_score'],
        orientation='h',
        marker_color='#F7C948',
        text=corridor_stats['trip_count'].astype(str) + ' trips',
        textposition='inside'
    ))
    
    fig.update_layout(
        title=f"Top 10 Risky Corridors" + (f" - {selected_borough}" if selected_borough else "") + (f" at {selected_hour}:00" if selected_hour is not None else ""),
        xaxis_title="Risk Score (Duration × Variability)",
        yaxis_title="Route",
        height=400,
        yaxis=dict(autorange='reversed')
    )
    
    return fig

# Main app
def main():
    # Header
    st.markdown("<h1>🚚 PromiseWise NYC</h1>", unsafe_allow_html=True)
    st.markdown('<p class="subtitle">Weather-aware ETA & Delay Risk Prediction Engine</p>', unsafe_allow_html=True)
    
    # Load data
    with st.spinner("Loading data..."):
        trips, zones, weather, holidays = load_data()
        
        if trips is None:
            st.error("Failed to load data. Please check file paths.")
            return
        
        df = preprocess_data(trips, zones, weather, holidays)
    
    # Sidebar filters
    st.sidebar.markdown("## 🎛️ Control Panel")
    
    # Month filter
    available_months = sorted(df['month_name'].dropna().unique())
    selected_month = st.sidebar.selectbox("📅 Month", available_months, index=len(available_months)-1)
    
    # Hour range filter
    hour_range = st.sidebar.slider("🕐 Hour Range", 0, 23, (0, 23))
    
    # Borough filter
    available_boroughs = sorted(df['pickup_borough'].dropna().unique())
    selected_boroughs = st.sidebar.multiselect("🏙️ Pickup Borough", available_boroughs, default=available_boroughs)
    
    # Weather filter
    weather_options = st.sidebar.radio("🌦️ Weather Condition", ["All", "Dry", "Wet"])
    
    # Promise percentile slider
    promise_percentile = st.sidebar.slider("📊 Promise Percentile", 50, 95, 90, 5)
    
    # Apply filters
    filtered_df = df[
        (df['month_name'] == selected_month) &
        (df['hour'] >= hour_range[0]) &
        (df['hour'] <= hour_range[1]) &
        (df['pickup_borough'].isin(selected_boroughs))
    ].copy()
    
    if weather_options != "All":
        filtered_df = filtered_df[filtered_df['weather_condition'] == weather_options]
    
    # Calculate metrics
    metrics = calculate_promise_metrics(filtered_df, promise_percentile)
    
    # KPI Cards
    st.markdown("## 📊 Key Performance Indicators")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown(f"""
        <div class="metric-card">
            <h3>⏱️ Median ETA</h3>
            <h2 class="taxi-yellow">{metrics['median_eta']:.1f} min</h2>
            <p>{metrics['total_trips']} trips analyzed</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        icon = "✅" if metrics['late_rate'] < 10 else "⚠️"
        st.markdown(f"""
        <div class="metric-card">
            <h3>📉 Late-Promise Rate</h3>
            <h2 class="taxi-yellow">{metrics['late_rate']:.1f}% {icon}</h2>
            <p>P{promise_percentile} strategy</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        coverage_icon = "✅" if metrics['p90_coverage'] >= 85 else "❌"
        st.markdown(f"""
        <div class="metric-card">
            <h3>🎯 P90 Coverage</h3>
            <h2 class="taxi-yellow">{metrics['p90_coverage']:.1f}% {coverage_icon}</h2>
            <p>Target: 90%</p>
        </div>
        """, unsafe_allow_html=True)
    
    # Charts Grid (2x2)
    st.markdown("## 📈 Interactive Analysis Dashboard")
    
    chart_col1, chart_col2 = st.columns(2)
    
    with chart_col1:
        # Trade-off curve
        try:
            trade_off_fig, percentiles, late_rates = create_trade_off_curve(filtered_df)
            
            # Add vertical line for current percentile
            current_late_rate = late_rates[percentiles.index(promise_percentile)] if promise_percentile in percentiles else 0
            trade_off_fig.add_vline(x=promise_percentile, line_dash="dash", line_color="red", 
                                   annotation_text=f"Current: {promise_percentile}% → {current_late_rate:.1f}% late")
            
            st.plotly_chart(trade_off_fig, use_container_width=True)
        except Exception as e:
            st.error(f"Error creating trade-off chart: {e}")
    
    with chart_col2:
        # Hourly profile
        try:
            hourly_fig = create_hourly_profile(filtered_df)
            st.plotly_chart(hourly_fig, use_container_width=True)
        except Exception as e:
            st.error(f"Error creating hourly profile: {e}")
    
    chart_col3, chart_col4 = st.columns(2)
    
    with chart_col3:
        # Borough heatmap with click handling
        try:
            heatmap_fig = create_borough_heatmap(filtered_df)
            
            # Store clicked cell info
            if 'clicked_borough' not in st.session_state:
                st.session_state.clicked_borough = None
            if 'clicked_hour' not in st.session_state:
                st.session_state.clicked_hour = None
            
            st.plotly_chart(heatmap_fig, use_container_width=True, key="heatmap")
            
            # Heatmap interaction info
            st.info("💡 Click on heatmap cells to filter the corridor chart")
            
        except Exception as e:
            st.error(f"Error creating heatmap: {e}")
    
    with chart_col4:
        # Top 10 corridors
        try:
            corridor_fig = create_corridor_top10(
                filtered_df, 
                st.session_state.get('clicked_borough'), 
                st.session_state.get('clicked_hour')
            )
            st.plotly_chart(corridor_fig, use_container_width=True)
        except Exception as e:
            st.error(f"Error creating corridor chart: {e}")
    
    # Download section
    st.markdown("## 💾 Export Options")
    
    download_col1, download_col2 = st.columns(2)
    
    with download_col1:
        if st.button("📥 Download Filtered CSV"):
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            csv_data = filtered_df.to_csv(index=False)
            st.download_button(
                label="Download CSV",
                data=csv_data,
                file_name=f"promisewise_filtered_{timestamp}.csv",
                mime="text/csv"
            )
    
    with download_col2:
        st.info("📊 Charts can be downloaded using the camera icon in each chart's toolbar")
    
    # Footer with insights
    st.markdown("---")
    st.markdown("### 🔍 Quick Insights")
    
    insights_col1, insights_col2 = st.columns(2)
    
    with insights_col1:
        if weather_options == "Wet":
            dry_metrics = calculate_promise_metrics(filtered_df[filtered_df['weather_condition'] == 'Dry'], promise_percentile)
            wet_metrics = calculate_promise_metrics(filtered_df[filtered_df['weather_condition'] == 'Wet'], promise_percentile)
            if dry_metrics['median_eta'] > 0 and wet_metrics['median_eta'] > 0:
                weather_impact = wet_metrics['median_eta'] - dry_metrics['median_eta']
                st.success(f"🌧️ Rain adds ~{weather_impact:.1f} minutes to trips")
        
        peak_hours = filtered_df.groupby('hour')['duration_minutes'].mean().nlargest(3)
        if not peak_hours.empty:
            peak_hour = peak_hours.index[0]
            st.warning(f"⏰ Peak congestion at {peak_hour}:00 (+{peak_hours.iloc[0]:.1f} min avg)")
    
    with insights_col2:
        if len(selected_boroughs) > 1:
            borough_performance = filtered_df.groupby('pickup_borough')['duration_minutes'].median().sort_values()
            fastest_borough = borough_performance.index[0]
            slowest_borough = borough_performance.index[-1]
            st.info(f"🏆 Fastest: {fastest_borough} | 🐌 Slowest: {slowest_borough}")
        
        buffer_time = metrics['promise_time'] - metrics['median_eta']
        st.success(f"🛡️ P{promise_percentile} strategy needs {buffer_time:.1f}min buffer")

if __name__ == "__main__":
    main()
