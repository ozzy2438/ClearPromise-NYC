"""
NYC Delivery Promise Engine - Feature Engineering Module

This module contains functions for transforming raw data into machine learning-ready features
for delivery promise optimization.
"""

import pandas as pd
import numpy as np
from typing import Tuple, Dict, List
import logging
from sklearn.preprocessing import LabelEncoder

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def add_temporal_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add temporal features for delivery analysis.
    
    Args:
        df: DataFrame with pickup/dropoff datetime columns
        
    Returns:
        DataFrame with added temporal features
    """
    logger.info("🕒 Adding temporal features...")
    
    # Convert datetime columns
    df['tpep_pickup_datetime'] = pd.to_datetime(df['tpep_pickup_datetime'])
    df['tpep_dropoff_datetime'] = pd.to_datetime(df['tpep_dropoff_datetime'])
    
    # Calculate trip duration
    df['trip_duration_minutes'] = (
        df['tpep_dropoff_datetime'] - df['tpep_pickup_datetime']
    ).dt.total_seconds() / 60
    
    # Temporal features
    df['pickup_date'] = df['tpep_pickup_datetime'].dt.date
    df['pickup_date_str'] = df['pickup_date'].astype(str)
    df['pickup_hour'] = df['tpep_pickup_datetime'].dt.hour
    df['pickup_dow'] = df['tpep_pickup_datetime'].dt.dayofweek  # 0=Monday
    df['is_weekend'] = df['pickup_dow'].isin([5, 6])  # Saturday, Sunday
    
    # Rush hour indicators
    df['is_morning_rush'] = df['pickup_hour'].isin([7, 8, 9])
    df['is_evening_rush'] = df['pickup_hour'].isin([16, 17, 18, 19])
    df['is_rush_hour'] = df['is_morning_rush'] | df['is_evening_rush']
    
    logger.info("✅ Temporal features added")
    return df


def add_geographic_features(df: pd.DataFrame, zones_df: pd.DataFrame) -> pd.DataFrame:
    """
    Add geographic zone and borough features.
    
    Args:
        df: DataFrame with location IDs
        zones_df: Taxi zone lookup DataFrame
        
    Returns:
        DataFrame with added geographic features
    """
    logger.info("🗺️ Adding geographic features...")
    
    # Merge pickup zone info
    df = df.merge(
        zones_df[['LocationID', 'Zone', 'Borough']], 
        left_on='PULocationID', right_on='LocationID', how='left'
    )
    df.rename(columns={'Zone': 'pickup_zone', 'Borough': 'pickup_borough'}, inplace=True)
    df.drop('LocationID', axis=1, inplace=True)
    
    # Merge dropoff zone info
    df = df.merge(
        zones_df[['LocationID', 'Zone', 'Borough']], 
        left_on='DOLocationID', right_on='LocationID', how='left'
    )
    df.rename(columns={'Zone': 'dropoff_zone', 'Borough': 'dropoff_borough'}, inplace=True)
    df.drop('LocationID', axis=1, inplace=True)
    
    # Handle missing zones
    df['pickup_zone'] = df['pickup_zone'].fillna('Unknown')
    df['pickup_borough'] = df['pickup_borough'].fillna('Unknown')
    df['dropoff_zone'] = df['dropoff_zone'].fillna('Unknown')
    df['dropoff_borough'] = df['dropoff_borough'].fillna('Unknown')
    
    # Add derived features
    df['same_borough'] = df['pickup_borough'] == df['dropoff_borough']
    df['route_pair'] = df['pickup_zone'] + ' → ' + df['dropoff_zone']
    
    # Airport indicators
    airport_zones = ['JFK Airport', 'LaGuardia Airport', 'Newark Airport']
    df['pickup_airport'] = df['pickup_zone'].isin(airport_zones)
    df['dropoff_airport'] = df['dropoff_zone'].isin(airport_zones)
    df['airport_trip'] = df['pickup_airport'] | df['dropoff_airport']
    
    logger.info("✅ Geographic features added")
    return df


def add_weather_features(df: pd.DataFrame, weather_df: pd.DataFrame) -> pd.DataFrame:
    """
    Add weather condition features.
    
    Args:
        df: DataFrame with pickup_date_str column
        weather_df: Weather data DataFrame
        
    Returns:
        DataFrame with added weather features
    """
    logger.info("🌤️ Adding weather features...")
    
    # Prepare weather data
    weather_df['date'] = pd.to_datetime(weather_df['time']).dt.date.astype(str)
    weather_cols = ['date', 'tavg', 'tmin', 'tmax', 'prcp', 'wspd']
    weather_clean = weather_df[weather_cols].copy()
    
    # Merge weather data
    df = df.merge(weather_clean, left_on='pickup_date_str', right_on='date', how='left')
    df.drop('date', axis=1, inplace=True)
    
    # Create weather condition indicators
    df['is_wet'] = df['prcp'] > 0  # Any precipitation
    df['is_hot'] = df['tavg'] > 25  # Hot days (>25°C)
    df['is_cold'] = df['tavg'] < 5   # Cold days (<5°C)
    df['is_windy'] = df['wspd'] > 15  # Windy days (>15 km/h)
    
    # Weather severity categories
    df['weather_severity'] = 'mild'
    df.loc[df['is_wet'] & df['is_windy'], 'weather_severity'] = 'severe'
    df.loc[df['is_wet'] | df['is_windy'], 'weather_severity'] = 'moderate'
    
    weather_coverage = df['tavg'].notna().mean()
    logger.info(f"✅ Weather features added ({weather_coverage:.1%} coverage)")
    return df


def add_holiday_features(df: pd.DataFrame, holidays_df: pd.DataFrame) -> pd.DataFrame:
    """
    Add holiday and special day features.
    
    Args:
        df: DataFrame with pickup_date_str column
        holidays_df: Holidays DataFrame
        
    Returns:
        DataFrame with added holiday features
    """
    logger.info("🎉 Adding holiday features...")
    
    # Prepare holidays data
    holidays_df['date'] = pd.to_datetime(holidays_df['date']).dt.date.astype(str)
    holiday_dates = set(holidays_df['date'].unique())
    
    # Add holiday indicator
    df['is_holiday'] = df['pickup_date_str'].isin(holiday_dates)
    
    # Add special day indicators
    df['is_special_day'] = df['is_holiday'] | df['is_weekend']
    
    logger.info(f"✅ Holiday features added ({df['is_holiday'].sum()} holiday trips)")
    return df


def clean_outliers(df: pd.DataFrame, duration_col: str = 'trip_duration_minutes') -> pd.DataFrame:
    """
    Remove extreme outliers from trip data.
    
    Args:
        df: DataFrame to clean
        duration_col: Name of duration column
        
    Returns:
        Cleaned DataFrame
    """
    logger.info("🧹 Cleaning outliers...")
    
    rows_before = len(df)
    
    # Define filtering criteria
    min_duration = 0.5  # 30 seconds minimum
    max_duration = df[duration_col].quantile(0.995)  # P99.5 threshold
    
    # Apply filters
    filter_mask = (
        (df[duration_col] >= min_duration) & 
        (df[duration_col] <= max_duration) &
        (df['trip_distance'] > 0) &
        (df['total_amount'] > 0)
    )
    
    df_clean = df[filter_mask].copy()
    rows_after = len(df_clean)
    rows_dropped = rows_before - rows_after
    
    logger.info(f"✅ Removed {rows_dropped:,} outliers ({rows_dropped/rows_before:.2%})")
    logger.info(f"📊 Duration range: {df_clean[duration_col].min():.1f} to {df_clean[duration_col].max():.1f} minutes")
    
    return df_clean


def encode_categorical_features(
    df_train: pd.DataFrame, 
    df_test: pd.DataFrame, 
    categorical_features: List[str]
) -> Tuple[pd.DataFrame, pd.DataFrame, Dict]:
    """
    Encode categorical features for machine learning.
    
    Args:
        df_train: Training DataFrame
        df_test: Test DataFrame
        categorical_features: List of categorical column names
        
    Returns:
        Tuple of (encoded_train_df, encoded_test_df, label_encoders_dict)
    """
    logger.info("🔢 Encoding categorical features...")
    
    le_dict = {}
    
    for feature in categorical_features:
        le = LabelEncoder()
        
        # Fit on combined data to ensure consistent encoding
        combined_values = pd.concat([df_train[feature], df_test[feature]]).unique()
        le.fit(combined_values)
        
        # Transform both datasets
        df_train[f'{feature}_encoded'] = le.transform(df_train[feature])
        df_test[f'{feature}_encoded'] = le.transform(df_test[feature])
        
        le_dict[feature] = le
        logger.info(f"✅ Encoded {feature}: {len(le.classes_)} categories")
    
    return df_train, df_test, le_dict


def create_ml_features(
    df_train: pd.DataFrame,
    df_test: pd.DataFrame,
    weather_df: pd.DataFrame,
    zones_df: pd.DataFrame,
    holidays_df: pd.DataFrame
) -> Tuple[pd.DataFrame, pd.DataFrame, Dict]:
    """
    Complete feature engineering pipeline for ML models.
    
    Args:
        df_train: Training data
        df_test: Test data  
        weather_df: Weather data
        zones_df: Taxi zones data
        holidays_df: Holidays data
        
    Returns:
        Tuple of (processed_train_df, processed_test_df, encoders_dict)
    """
    logger.info("🔧 Starting complete feature engineering pipeline...")
    
    # Process both datasets
    for name, df in [("Training", df_train), ("Test", df_test)]:
        logger.info(f"\n📊 Processing {name} dataset: {len(df):,} records")
        
        # Add features
        df = add_temporal_features(df)
        df = add_geographic_features(df, zones_df)
        df = add_weather_features(df, weather_df)
        df = add_holiday_features(df, holidays_df)
        df = clean_outliers(df)
        
        # Update reference
        if name == "Training":
            df_train = df
        else:
            df_test = df
    
    # Encode categorical features
    categorical_features = ['pickup_borough', 'dropoff_borough', 'weather_severity']
    df_train, df_test, le_dict = encode_categorical_features(
        df_train, df_test, categorical_features
    )
    
    logger.info(f"\n🎯 Feature engineering complete!")
    logger.info(f"Training: {len(df_train):,} records")
    logger.info(f"Test: {len(df_test):,} records")
    
    return df_train, df_test, le_dict


if __name__ == "__main__":
    # Example usage
    from data_download import load_taxi_data, load_supporting_data
    
    # Load sample data
    data_dir = "../DATA"
    df = load_taxi_data(2025, [7], data_dir)
    weather, zones, holidays = load_supporting_data(data_dir)
    
    # Test feature engineering
    df_processed = add_temporal_features(df.head(1000))
    df_processed = add_geographic_features(df_processed, zones)
    df_processed = add_weather_features(df_processed, weather)
    
    print(f"✅ Feature engineering test completed: {len(df_processed)} records")
    print(f"📊 Features added: {df_processed.columns.tolist()}")
