"""
NYC Delivery Promise Engine - Data Download Module

This module handles downloading and loading of NYC taxi trip data, weather data,
and supporting datasets for delivery promise optimization analysis.
"""

import pandas as pd
import requests
import os
from typing import List, Tuple
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def download_nyc_taxi_data(year: int, month: int, data_dir: str = "DATA") -> str:
    """
    Download NYC taxi trip data for a specific month.
    
    Args:
        year: Year of data (e.g., 2025)
        month: Month of data (1-12)
        data_dir: Directory to save data
        
    Returns:
        Path to downloaded file
    """
    # NYC Taxi & Limousine Commission data URL
    base_url = "https://d37ci6vzurychx.cloudfront.net/trip-data"
    filename = f"yellow_tripdata_{year}-{month:02d}.csv"
    url = f"{base_url}/{filename}"
    filepath = os.path.join(data_dir, filename)
    
    if os.path.exists(filepath):
        logger.info(f"✅ File already exists: {filepath}")
        return filepath
    
    logger.info(f"📥 Downloading {filename}...")
    response = requests.get(url)
    response.raise_for_status()
    
    os.makedirs(data_dir, exist_ok=True)
    with open(filepath, 'wb') as f:
        f.write(response.content)
    
    logger.info(f"✅ Downloaded: {filepath}")
    return filepath


def load_taxi_data(year: int, months: List[int], data_dir: str = "DATA") -> pd.DataFrame:
    """
    Load and combine multiple months of taxi data.
    
    Args:
        year: Year of data
        months: List of months to load
        data_dir: Directory containing data files
        
    Returns:
        Combined DataFrame
    """
    dataframes = []
    
    for month in months:
        filename = f"yellow_tripdata_{year}-{month:02d}.csv"
        filepath = os.path.join(data_dir, filename)
        
        if not os.path.exists(filepath):
            logger.warning(f"⚠️ File not found: {filepath}")
            continue
            
        logger.info(f"📖 Loading {filename}...")
        df = pd.read_csv(filepath)
        df['source_month'] = month
        dataframes.append(df)
    
    if not dataframes:
        raise ValueError("No data files found!")
    
    combined_df = pd.concat(dataframes, ignore_index=True)
    logger.info(f"✅ Loaded {len(combined_df):,} total records from {len(dataframes)} months")
    
    return combined_df


def load_supporting_data(data_dir: str = "DATA") -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Load weather, taxi zones, and holiday data.
    
    Args:
        data_dir: Directory containing data files
        
    Returns:
        Tuple of (weather_df, zones_df, holidays_df)
    """
    # Load taxi zone lookup
    zones_path = os.path.join(data_dir, "taxi_zone_lookup.csv")
    zones_df = pd.read_csv(zones_path)
    logger.info(f"✅ Loaded {len(zones_df)} taxi zones")
    
    # Load weather data  
    weather_path = os.path.join(data_dir, "nyc_weather_3months.csv")
    weather_df = pd.read_csv(weather_path)
    logger.info(f"✅ Loaded {len(weather_df)} weather records")
    
    # Load holidays
    holidays_path = os.path.join(data_dir, "us_public_holidays_2025.csv")
    holidays_df = pd.read_csv(holidays_path)
    logger.info(f"✅ Loaded {len(holidays_df)} holidays")
    
    return weather_df, zones_df, holidays_df


def validate_data_schema(df: pd.DataFrame) -> bool:
    """
    Validate that required columns exist in taxi data.
    
    Args:
        df: DataFrame to validate
        
    Returns:
        True if schema is valid
    """
    required_columns = [
        'tpep_pickup_datetime',
        'tpep_dropoff_datetime', 
        'PULocationID',
        'DOLocationID',
        'trip_distance',
        'total_amount'
    ]
    
    missing_columns = [col for col in required_columns if col not in df.columns]
    
    if missing_columns:
        logger.error(f"❌ Missing required columns: {missing_columns}")
        return False
    
    logger.info("✅ Data schema validation passed")
    return True


if __name__ == "__main__":
    # Example usage
    data_dir = "../DATA"
    
    # Load sample data for testing
    try:
        # Load July 2025 data for analysis
        df = load_taxi_data(2025, [7], data_dir)
        
        # Validate schema
        if validate_data_schema(df):
            print(f"✅ Successfully loaded and validated {len(df):,} records")
        
        # Load supporting data
        weather, zones, holidays = load_supporting_data(data_dir)
        print(f"✅ Supporting data loaded: {len(weather)} weather, {len(zones)} zones, {len(holidays)} holidays")
        
    except Exception as e:
        logger.error(f"❌ Error: {e}")
