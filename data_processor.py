"""Core data processing module for fitness tracker data."""

import pandas as pd
import numpy as np
import os
import json
import sqlite3
import traceback
import warnings
from datetime import datetime
import config
from utils import convert_duration_to_minutes

# Ignore specific FutureWarning about concat behavior
warnings.filterwarnings('ignore', category=FutureWarning, 
                        message='The behavior of DataFrame concatenation with empty or all-NA entries is deprecated')


def load_miifit_data():
    """Load and process MiiFit data from either SQLite database or CSV file."""
    # Check for SQLite database
    if os.path.exists(config.MIIFIT_PATH) and config.MIIFIT_PATH.endswith('.db'):
        return load_miifit_from_db(config.MIIFIT_PATH)
    
    # Check for CSV file
    csv_path = config.MIIFIT_PATH.replace('.db', '.csv')
    if os.path.exists(csv_path) or os.path.exists(config.MIIFIT_PATH):
        actual_path = csv_path if os.path.exists(csv_path) else config.MIIFIT_PATH
        return load_miifit_from_csv(actual_path)
    
    print(f"No MiiFit data found at: {config.MIIFIT_PATH} or {csv_path}")
    return create_empty_df()

def load_miifit_from_db(db_path):
    """Load MiiFit data from SQLite database."""
    try:
        print(f"Attempting to load MiiFit data from database: {db_path}")
        
        # Connect to SQLite database and execute query
        conn = sqlite3.connect(db_path)
        query = "SELECT _id, DATE, TYPE, TRACKID, ENDTIME, CAL, AVGHR, MAX_HR from TRACKRECORD"
        df = pd.read_sql(query, conn, index_col="_id")
        
        # Process data
        # Remove HR outliers
        df = df[(df["AVGHR"] > 50) & (df["MAX_HR"] > 50)]
        
        # Create duration column from timestamps
        df['TRACKID'] = pd.to_datetime(df['TRACKID'], unit='s')
        df['ENDTIME'] = pd.to_datetime(df['ENDTIME'], unit='s')
        df['duration_minutes'] = ((df['ENDTIME'] - df['TRACKID']).dt.total_seconds() / 60).round()
        
        # Remove duration outliers
        df = df[df["duration_minutes"] > 10]
        
        # Replace type with sport 
        new_type = {16: "Free", 10: "IndCyc", 9: "OutCyc", 12: "Elliptical", 60: "Yoga", 14: "Swim"}
        df['TYPE'] = df['TYPE'].replace(new_type)
        
        # Rename and standardize
        df = df.rename(columns=config.MIIFIT_COLUMNS)
        df['source'] = 'MiiFit'
        df['location'] = None
        df['date'] = pd.to_datetime(df['date'], errors='coerce').dt.date
        
        # Standardize activity types
        df['activity_type'] = df['activity_type'].map(
            lambda x: config.ACTIVITY_TYPE_MAPPING.get(x, x) if isinstance(x, str) else x
        )
        
        # Select and clean columns
        standard_columns = ['date', 'activity_type', 'avg_hr', 'max_hr', 'calories', 'duration', 'source', 'location']
        for col in standard_columns:
            if col not in df.columns:
                df[col] = None
        
        df = df[standard_columns].dropna(subset=['date'])
        print(f"Processed MiiFit data: {len(df)} rows")
        return df
    except Exception as e:
        print(f"Error loading MiiFit data from database: {e}")
        traceback.print_exc()
        return create_empty_df()

def load_miifit_from_csv(csv_path):
    """Load MiiFit data from CSV file."""
    try:
        print(f"Attempting to load MiiFit data from CSV: {csv_path}")
        df = pd.read_csv(csv_path)
        
        # Map columns to standard names
        column_mapping = {col: std_col for col in df.columns 
                         for std_col in config.MIIFIT_COLUMNS 
                         if col.lower() == std_col.lower()}
        
        if column_mapping:
            df = df.rename(columns=column_mapping)
        
        if any(col in df.columns for col in config.MIIFIT_COLUMNS):
            valid_columns = {k: v for k, v in config.MIIFIT_COLUMNS.items() if k in df.columns}
            df = df.rename(columns=valid_columns)
        
        # Standardize data
        df['source'] = 'MiiFit'
        if 'location' not in df.columns:
            df['location'] = None
        
        if 'date' in df.columns:
            df['date'] = pd.to_datetime(df['date'], errors='coerce').dt.date
        
        if 'activity_type' in df.columns:
            df['activity_type'] = df['activity_type'].map(
                lambda x: config.ACTIVITY_TYPE_MAPPING.get(x, x) if isinstance(x, str) else x
            )
        
        # Select relevant columns
        standard_columns = ['date', 'activity_type', 'avg_hr', 'max_hr', 'calories', 'duration', 'source', 'location']
        for col in standard_columns:
            if col not in df.columns:
                df[col] = None
        
        df = df[standard_columns].dropna(subset=['date'])
        print(f"Processed MiiFit CSV data: {len(df)} rows")
        return df
    except Exception as e:
        print(f"Error loading MiiFit data from CSV: {e}")
        traceback.print_exc()
        return create_empty_df()

def load_garmin_data():
    """Load and process Garmin data from either JSON file or CSV file."""
    # Check for JSON file
    if os.path.exists(config.GARMIN_PATH) and config.GARMIN_PATH.endswith('.json'):
        return load_garmin_from_json(config.GARMIN_PATH)
    
    # Check for CSV file
    csv_path = config.GARMIN_PATH.replace('.json', '.csv')
    if os.path.exists(csv_path) or (os.path.exists(config.GARMIN_PATH) and config.GARMIN_PATH.endswith('.csv')):
        actual_path = csv_path if os.path.exists(csv_path) else config.GARMIN_PATH
        return load_garmin_from_csv(actual_path)
    
    print(f"No Garmin data found at: {config.GARMIN_PATH} or {csv_path}")
    return create_empty_df()

def load_garmin_from_json(json_path):
    """Load Garmin data from JSON file."""
    try:
        print(f"Attempting to load Garmin data from JSON: {json_path}")
        with open(json_path, 'r') as file:
            json_data = json.load(file)
        
        # Extract activities
        activities = json_data[0]['summarizedActivitiesExport'] if (
            isinstance(json_data, list) and 
            len(json_data) > 0 and 
            'summarizedActivitiesExport' in json_data[0]
        ) else []
        
        if not activities:
            print("No activities found in Garmin JSON data")
            return create_empty_df()
        
        # Extract necessary fields
        extracted = []
        for act in activities:
            try:
                # Skip activities without date
                start_time_local = act.get('startTimeLocal')
                if not start_time_local:
                    continue
                
                activity_date = datetime.fromtimestamp(start_time_local / 1000).date()
                duration_ms = act.get('duration', 0)
                duration_minutes = duration_ms / 60000 if duration_ms else None
                
                extracted.append({
                    "date": activity_date,
                    "activity_type": act.get('activityType', 'Unknown'),
                    "avg_hr": act.get('avgHr'),
                    "max_hr": act.get('maxHr'),
                    "calories": act.get('calories'),
                    "duration": duration_minutes,
                    "source": "Garmin",
                    "location": None
                })
            except Exception as e:
                print(f"Error processing Garmin activity: {e}")
                continue
        
        # Create DataFrame and standardize
        df = pd.DataFrame(extracted)
        if 'activity_type' in df.columns and not df.empty:
            df['activity_type'] = df['activity_type'].map(
                lambda x: config.ACTIVITY_TYPE_MAPPING.get(x, x) if isinstance(x, str) else x
            )
        
        print(f"Processed Garmin JSON data: {len(df)} rows")
        return df
    except Exception as e:
        print(f"Error loading Garmin data from JSON: {e}")
        traceback.print_exc()
        return create_empty_df()

def load_garmin_from_csv(csv_path):
    """Load Garmin data from CSV file."""
    try:
        print(f"Attempting to load Garmin data from CSV: {csv_path}")
        df = pd.read_csv(csv_path)
        
        # Map columns to standard names
        column_mapping = {col: std_col for col in df.columns 
                         for std_col in config.GARMIN_COLUMNS 
                         if col.lower() == std_col.lower()}
        
        if column_mapping:
            df = df.rename(columns=column_mapping)
        
        if any(col in df.columns for col in config.GARMIN_COLUMNS):
            valid_columns = {k: v for k, v in config.GARMIN_COLUMNS.items() if k in df.columns}
            df = df.rename(columns=valid_columns)
        
        # Standardize data
        df['source'] = 'Garmin'
        if 'location' not in df.columns:
            df['location'] = None
        
        if 'date' in df.columns:
            df['date'] = pd.to_datetime(df['date'], errors='coerce').dt.date
        
        if 'duration_string' in df.columns:
            df['duration'] = df['duration_string'].apply(convert_duration_to_minutes)
        
        if 'calories' in df.columns:
            df['calories'] = pd.to_numeric(df['calories'], errors='coerce')
        
        if 'activity_type' in df.columns:
            df['activity_type'] = df['activity_type'].map(
                lambda x: config.ACTIVITY_TYPE_MAPPING.get(x, x) if isinstance(x, str) else x
            )
        
        # Select relevant columns
        standard_columns = ['date', 'activity_type', 'avg_hr', 'max_hr', 'calories', 'duration', 'source', 'location']
        for col in standard_columns:
            if col not in df.columns:
                df[col] = None
        
        df = df[standard_columns].dropna(subset=['date'])
        print(f"Processed Garmin CSV data: {len(df)} rows")
        return df
    except Exception as e:
        print(f"Error loading Garmin data from CSV: {e}")
        traceback.print_exc()
        return create_empty_df()

def load_polar_f11_data():
    """Load and process Polar F11 data from CSV file."""
    try:
        # Check if the configured path exists
        if not os.path.exists(config.POLAR_F11_PATH):
            print(f"Polar F11 data file not found at: {config.POLAR_F11_PATH}")
            return try_alternate_paths('PolarF11')
        
        print(f"Found Polar F11 data at: {config.POLAR_F11_PATH}")
        df = pd.read_csv(config.POLAR_F11_PATH)
        
        # Standardize data
        df = df.rename(columns=config.POLAR_F11_COLUMNS)
        df['source'] = 'PolarF11'
        
        # Process dates
        if 'date' in df.columns:
            clean_dates = df['date'].astype(str).str.replace(r'\.0$', '', regex=True)
            df['date'] = pd.to_datetime(clean_dates, errors='coerce').dt.date
        
        # Process duration
        if 'duration_string' in df.columns:
            df['duration'] = df['duration_string'].apply(convert_duration_to_minutes)
        
        # Standardize activities
        if 'activity_type' in df.columns:
            df['activity_type'] = df['activity_type'].map(
                lambda x: config.ACTIVITY_TYPE_MAPPING.get(x, x) if isinstance(x, str) else x
            )
        
        # Convert numeric columns
        for col in ['avg_hr', 'max_hr', 'calories']:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
        
        # Ensure all standard columns exist
        standard_columns = ['date', 'activity_type', 'avg_hr', 'max_hr', 'calories', 'duration', 'source', 'location']
        for col in standard_columns:
            if col not in df.columns:
                df[col] = None
        
        df = df[standard_columns].dropna(subset=['date'])
        print(f"Processed Polar F11 data: {len(df)} rows")
        return df
    except Exception as e:
        print(f"Error loading Polar F11 data: {e}")
        traceback.print_exc()
        return create_empty_df()

def load_charge_hr_data():
    """Load and process Charge HR data from CSV file."""
    try:
        # Check if the configured path exists
        if not os.path.exists(config.CHARGE_HR_PATH):
            print(f"Charge HR data file not found at: {config.CHARGE_HR_PATH}")
            return try_alternate_paths('ChargeHR')
        
        print(f"Found Charge HR data at: {config.CHARGE_HR_PATH}")
        df = pd.read_csv(config.CHARGE_HR_PATH)
        
        # Standardize data
        df = df.rename(columns=config.CHARGE_HR_COLUMNS)
        df['source'] = 'ChargeHR'
        
        # Process dates
        if 'date' in df.columns:
            # Add year if available
            if 'year' in df.columns and not df['date'].astype(str).str.contains(r'\d{4}').any():
                df['date'] = df['date'].astype(str) + " " + df['year'].astype(str)
            
            # Try multiple date formats
            df['date'] = pd.to_datetime(df['date'], errors='coerce')
            df['date'] = df['date'].dt.date
        
        # Process duration
        if 'duration_string' in df.columns:
            df['duration'] = df['duration_string'].apply(convert_duration_to_minutes)
        
        # Standardize activities
        if 'activity_type' in df.columns:
            df['activity_type'] = df['activity_type'].map(
                lambda x: config.ACTIVITY_TYPE_MAPPING.get(x, x) if isinstance(x, str) else x
            )
        
        # Estimate heart rate from calories
        if 'calories' in df.columns and ('avg_hr' not in df.columns or df['avg_hr'].isna().all()):
            df['avg_hr'] = df['calories'] / 4
        
        # Ensure all standard columns exist
        standard_columns = ['date', 'activity_type', 'avg_hr', 'max_hr', 'calories', 'duration', 'source', 'location']
        for col in standard_columns:
            if col not in df.columns:
                df[col] = None
        
        df = df[standard_columns].dropna(subset=['date'])
        print(f"Processed Charge HR data: {len(df)} rows")
        return df
    except Exception as e:
        print(f"Error loading Charge HR data: {e}")
        traceback.print_exc()
        return create_empty_df()

def try_alternate_paths(source_type):
    """Try to load data from alternative paths."""
    potential_paths = {
        'PolarF11': [
            'PolarF11Data.csv', 'polar_f11_data.csv', 'PolarF11.csv',
            'polar_f11.csv', 'polar_data.csv', 'PolarData.csv'
        ],
        'ChargeHR': [
            'ChargeHRDataScrape.csv', 'charge_hr_data_scrape.csv', 'ChargeHR.csv',
            'charge_hr.csv', 'fitbit_data.csv', 'FitbitData.csv'
        ]
    }
    
    # Try each path
    for path in potential_paths.get(source_type, []):
        if os.path.exists(path):
            print(f"Found {source_type} data at alternate path: {path}")
            try:
                df = pd.read_csv(path)
                df['source'] = source_type
                
                # Add empty columns if needed
                standard_columns = ['date', 'activity_type', 'avg_hr', 'max_hr', 'calories', 'duration', 'source', 'location']
                for col in standard_columns:
                    if col not in df.columns:
                        df[col] = None
                
                return df
            except Exception as e:
                print(f"Error loading {source_type} data from alternate path {path}: {e}")
    
    print(f"No valid {source_type} data file found at any location")
    return create_empty_df()

def create_empty_df():
    """Create an empty DataFrame with the standard columns."""
    return pd.DataFrame(columns=['date', 'activity_type', 'avg_hr', 'max_hr', 'calories', 'duration', 'source', 'location'])

def combine_data():
    """Combine data from all sources."""
    print("\n===== LOADING DATA FROM ALL SOURCES =====")
    
    # Load data from all sources including PolarF11 and ChargeHR
    dataframes = {
        'MiiFit': load_miifit_data(),
        'Garmin': load_garmin_data(),
        'PolarF11': load_polar_f11_data(),
        'ChargeHR': load_charge_hr_data()
    }
    
    for source, df in dataframes.items():
        print(f"{source} data: {len(df)} rows")
    
    print("======================================")
    
    # Combine all data sources
    non_empty_dfs = [df for df in dataframes.values() if not df.empty]
    
    if not non_empty_dfs:
        print("Warning: All data sources are empty or could not be loaded.")
        return create_empty_df()
    
    # Updated code to suppress warning:
    if non_empty_dfs:
        # Filter out completely empty DataFrames before concatenation
        valid_dfs = [df for df in non_empty_dfs if not df.empty]
        combined_df = pd.concat(valid_dfs, ignore_index=True) if valid_dfs else pd.DataFrame()
    else:
        combined_df = pd.DataFrame()
    # Additional data cleaning and standardization
    if not combined_df.empty:
        # Sort by date
        combined_df = combined_df.sort_values('date')
        
        # Convert all columns to the correct data types
        for col in ['avg_hr', 'max_hr', 'calories', 'duration']:
            if col in combined_df.columns:
                combined_df[col] = pd.to_numeric(combined_df[col], errors='coerce')
    
    print(f"Combined data: {len(combined_df)} rows")
    return combined_df

def filter_data(df, start_date=None, end_date=None, sources=None, activity_types=None):
    """Filter data based on selected criteria."""
    if df.empty:
        return df
    
    filtered_df = df.copy()
    
    # Apply filters
    if start_date is not None and 'date' in filtered_df.columns:
        filtered_df = filtered_df[filtered_df['date'] >= start_date]
    
    if end_date is not None and 'date' in filtered_df.columns:
        filtered_df = filtered_df[filtered_df['date'] <= end_date]
    
    if sources and 'source' in filtered_df.columns:
        filtered_df = filtered_df[filtered_df['source'].isin(sources)]
    
    if activity_types and 'activity_type' in filtered_df.columns and activity_types:
        filtered_df = filtered_df[filtered_df['activity_type'].isin(activity_types)]
    
    return filtered_df

def aggregate_data_by_date(df, metrics=None):
    """Aggregate data by date for the selected metrics."""
    if df.empty or not metrics or 'date' not in df.columns:
        return pd.DataFrame()
    
    # Validate metrics exist in dataframe
    valid_metrics = [metric for metric in metrics if metric in df.columns]
    
    if not valid_metrics:
        return pd.DataFrame()
    
    # Group by date and source, calculate the mean of each metric
    agg_dict = {metric: 'mean' for metric in valid_metrics}
    aggregated = df.groupby(['date', 'source']).agg(agg_dict).reset_index()
    
    return aggregated

def calculate_monthly_activity_counts(df):
    """Calculate monthly activity counts by type."""
    if df.empty or 'date' not in df.columns or 'activity_type' not in df.columns:
        return pd.DataFrame()
    
    # Create a copy and ensure date is datetime
    monthly_df = df.copy()
    if not pd.api.types.is_datetime64_any_dtype(monthly_df['date']):
        monthly_df['date'] = pd.to_datetime(monthly_df['date'])
    
    # Extract year and month
    monthly_df['year'] = monthly_df['date'].dt.year
    monthly_df['month'] = monthly_df['date'].dt.month
    
    # Group and count
    monthly_counts = monthly_df.groupby(['year', 'month', 'activity_type']).size().reset_index(name='count')
    monthly_counts['date'] = pd.to_datetime(monthly_counts[['year', 'month']].assign(day=1))
    
    return monthly_counts

def get_summary_stats(df):
    """Calculate summary statistics for numeric columns."""
    if df.empty:
        return pd.DataFrame()
    
    numeric_cols = ['avg_hr', 'max_hr', 'calories', 'duration']
    valid_cols = [col for col in numeric_cols if col in df.columns]
    
    if not valid_cols:
        return pd.DataFrame()
    
    stats = df[valid_cols].agg(['mean', 'std', 'min', 'max']).round(2)
    stats.index = ['Average', 'Std Dev', 'Minimum', 'Maximum']
    
    return stats
