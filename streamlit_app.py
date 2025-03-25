"""
Fitness Dashboard - Streamlit App for Cloud Deployment

A Streamlit dashboard for visualizing and analyzing fitness data
from multiple sources (MiiFit, Garmin, PolarF11, and ChargeHR).
Designed for deployment on Cloudlit and other cloud platforms.
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import os
import json
import numpy as np
from datetime import datetime, timedelta

# Define configuration variables directly in this file for cloud deployment
DEBUG = True
PLOT_HEIGHT = 600
PLOT_WIDTH = 800
DEFAULT_DAYS_BACK = 30
DATE_FORMAT = '%Y-%m-%d'

# Column mappings for data standardization
MIIFIT_COLUMNS = {
    'DATE': 'date',
    'TYPE': 'activity_type',
    'AVGHR': 'avg_hr',
    'MAX_HR': 'max_hr',
    'CAL': 'calories',
    'duration_minutes': 'duration'
}

GARMIN_COLUMNS = {
    'Date': 'date',
    'Activity Type': 'activity_type',
    'Avg HR': 'avg_hr',
    'Max HR': 'max_hr',
    'Calories': 'calories',
    'Total Time': 'duration_string'  # Will be converted to minutes
}

POLAR_F11_COLUMNS = {
    'sport': 'activity_type',
    'time': 'date',
    'calories': 'calories',
    'Duration': 'duration_string',
    'average': 'avg_hr',
    'maximum': 'max_hr',
    'Location': 'location',
    'Cal/HR': 'cal_hr_ratio'
}

CHARGE_HR_COLUMNS = {
    'Date': 'date',
    'Activity': 'activity_type',
    'Cals': 'calories',
    'Duration': 'duration_string',
    'Location': 'location',
    'Year': 'year'
}

# Activity type mapping for standardization
ACTIVITY_TYPE_MAPPING = {
    # MiiFit mappings
    'Free': 'Free Workout',
    'IndCyc': 'Indoor Cycling',
    'OutCyc': 'Outdoor Cycling',
    'Elliptical': 'Elliptical',
    'Yoga': 'Yoga',
    'Swim': 'Swimming',
    
    # Garmin mappings
    'Cycling': 'Outdoor Cycling',
    'CYCLING': 'Outdoor Cycling',
    'CYCLING_INDOOR': 'Indoor Cycling',
    'Indoor Cycling': 'Indoor Cycling',
    'Running': 'Running',
    'RUNNING': 'Running',
    'Swimming': 'Swimming',
    'SWIMMING': 'Swimming',
    'POOL_SWIMMING': 'Swimming',
    'Walking': 'Walking',
    'WALKING': 'Walking',
    'Yoga': 'Yoga',
    'YOGA': 'Yoga',
    'Strength Training': 'Strength Training',
    'STRENGTH_TRAINING': 'Strength Training',
    'Elliptical': 'Elliptical',
    'ELLIPTICAL': 'Elliptical',
    'CARDIO': 'Cardio',
    'Cardio': 'Cardio',
    'TREADMILL_RUNNING': 'Running',
    'OPEN_WATER_SWIMMING': 'Swimming',
    'FITNESS_EQUIPMENT': 'Gym Workout',
    
    # Polar F11 and Charge HR mappings
    'Tennis': 'Tennis',
    'Biking': 'Outdoor Cycling',
    'treadmill': 'Running',
    'Gym': 'Gym Workout',
    'Spin': 'Indoor Cycling',
    'Mountain Biking': 'Outdoor Cycling'
}

# Color schemes
COLOR_SCHEME_SOURCES = {
    'MiiFit': '#1f77b4',
    'Garmin': '#ff7f0e',
    'PolarF11': '#2ca02c',
    'ChargeHR': '#d62728'
}

COLOR_SCHEME = COLOR_SCHEME_SOURCES  # For backward compatibility

COLOR_SCHEME_ACTIVITIES = {
    'Free Workout': '#1f77b4',
    'Indoor Cycling': '#ff7f0e',
    'Outdoor Cycling': '#2ca02c',
    'Elliptical': '#d62728',
    'Yoga': '#9467bd',
    'Swimming': '#8c564b',
    'Running': '#e377c2',
    'Walking': '#7f7f7f',
    'Strength Training': '#bcbd22',
    'Cardio': '#17becf',
    'Gym Workout': '#9467bd',
    'Tennis': '#e7ba52'
}

# Utility functions
def format_duration(minutes):
    """Format duration in minutes to HH:MM:SS format."""
    if pd.isna(minutes):
        return "N/A"
    
    hours = int(minutes // 60)
    mins = int(minutes % 60)
    secs = int((minutes * 60) % 60)
    
    return f"{hours:02d}:{mins:02d}:{secs:02d}"

def convert_duration_to_minutes(duration_str):
    """Convert duration string (HH:MM:SS) to minutes."""
    try:
        if pd.isna(duration_str):
            return None
        time = datetime.strptime(str(duration_str), '%H:%M:%S')
        return time.hour * 60 + time.minute + time.second / 60
    except Exception:
        try:
            # Try to convert directly from milliseconds to minutes
            return float(duration_str) / 60000
        except:
            return None

def create_empty_df():
    """Create an empty DataFrame with the standard columns."""
    return pd.DataFrame(columns=['date', 'activity_type', 'avg_hr', 'max_hr', 'calories', 'duration', 'source', 'location'])

def get_sample_miifit_data(num_records=50):
    """Generate sample MiiFit data."""
    np.random.seed(42)
    
    # Generate dates (last 180 days)
    end_date = datetime.now().date()
    start_date = end_date - timedelta(days=180)
    dates = [start_date + timedelta(days=np.random.randint(0, 180)) for _ in range(num_records)]
    dates.sort()
    
    # Generate data
    data = []
    for i in range(num_records):
        # Randomly select activity type
        activity_type = np.random.choice(['Free', 'IndCyc', 'OutCyc', 'Elliptical', 'Yoga', 'Swim'])
        
        # Generate duration (20-90 minutes)
        duration = np.random.randint(20, 91)
        
        # Heart rate values
        avg_hr = np.random.randint(90, 160)
        max_hr = avg_hr + np.random.randint(10, 40)
        
        # Calories burned (roughly based on duration and intensity)
        calories = int(duration * avg_hr * 0.05)
        
        data.append({
            "date": dates[i],
            "activity_type": activity_type,
            "avg_hr": avg_hr,
            "max_hr": max_hr,
            "calories": calories,
            "duration": duration,
            "source": "MiiFit",
            "location": None
        })
    
    return pd.DataFrame(data)

def get_sample_garmin_data(num_records=50):
    """Generate sample Garmin data."""
    np.random.seed(24)
    
    # Generate dates (last 180 days)
    end_date = datetime.now().date()
    start_date = end_date - timedelta(days=180)
    dates = [start_date + timedelta(days=np.random.randint(0, 180)) for _ in range(num_records)]
    dates.sort()
    
    # Activity types
    activity_types = ["Running", "Walking", "Cycling", "Swimming", "Yoga", "Strength Training", "Indoor Cycling", "Elliptical"]
    
    # Generate data
    data = []
    for i in range(num_records):
        activity_type = np.random.choice(activity_types)
        
        # Calculate duration (20-90 minutes)
        duration_minutes = np.random.randint(20, 91)
        
        # Heart rate values
        avg_hr = np.random.randint(90, 160)
        max_hr = avg_hr + np.random.randint(10, 40)
        
        # Calories burned (roughly based on duration and intensity)
        calories = int(duration_minutes * avg_hr * 0.05)
        
        data.append({
            "date": dates[i],
            "activity_type": activity_type,
            "avg_hr": avg_hr,
            "max_hr": max_hr,
            "calories": calories,
            "duration": duration_minutes,
            "source": "Garmin",
            "location": None
        })
    
    return pd.DataFrame(data)

# Main data processing functions
def load_data():
    """Load data from uploaded files or generate sample data if none provided."""
    uploaded_files = st.session_state.get('uploaded_files', {})
    
    if not uploaded_files:
        # No files uploaded, offer to generate sample data
        if 'use_sample_data' not in st.session_state:
            st.session_state.use_sample_data = False
            
        use_sample = st.sidebar.checkbox("Use sample data", st.session_state.use_sample_data)
        
        if use_sample:
            st.session_state.use_sample_data = True
            # Generate sample data
            miifit_df = get_sample_miifit_data(50)
            garmin_df = get_sample_garmin_data(50)
            
            # Apply activity type mapping
            for df in [miifit_df, garmin_df]:
                df['activity_type'] = df['activity_type'].map(
                    lambda x: ACTIVITY_TYPE_MAPPING.get(x, x) if isinstance(x, str) else x
                )
            
            combined_df = pd.concat([miifit_df, garmin_df], ignore_index=True)
            combined_df = combined_df.sort_values('date')
            
            return combined_df
        else:
            st.session_state.use_sample_data = False
            return create_empty_df()
    else:
        # Process uploaded files
        dataframes = []
        
        # Process MiiFit data if uploaded
        if 'miifit' in uploaded_files:
            try:
                miifit_data = uploaded_files['miifit']
                miifit_df = pd.read_csv(miifit_data)
                
                # Rename columns to standard format
                miifit_df = miifit_df.rename(columns=MIIFIT_COLUMNS)
                miifit_df['source'] = 'MiiFit'
                
                # Convert date and standardize activity types
                miifit_df['date'] = pd.to_datetime(miifit_df['date']).dt.date
                miifit_df['activity_type'] = miifit_df['activity_type'].map(
                    lambda x: ACTIVITY_TYPE_MAPPING.get(x, x) if isinstance(x, str) else x
                )
                
                dataframes.append(miifit_df)
            except Exception as e:
                st.error(f"Error processing MiiFit data: {e}")
        
        # Process Garmin data if uploaded
        if 'garmin' in uploaded_files:
            try:
                garmin_data = uploaded_files['garmin']
                garmin_df = pd.read_csv(garmin_data)
                
                # Rename columns to standard format
                garmin_df = garmin_df.rename(columns=GARMIN_COLUMNS)
                garmin_df['source'] = 'Garmin'
                
                # Convert date and standardize activity types
                garmin_df['date'] = pd.to_datetime(garmin_df['date']).dt.date
                
                # Convert duration string to minutes if needed
                if 'duration_string' in garmin_df.columns:
                    garmin_df['duration'] = garmin_df['duration_string'].apply(convert_duration_to_minutes)
                
                garmin_df['activity_type'] = garmin_df['activity_type'].map(
                    lambda x: ACTIVITY_TYPE_MAPPING.get(x, x) if isinstance(x, str) else x
                )
                
                dataframes.append(garmin_df)
            except Exception as e:
                st.error(f"Error processing Garmin data: {e}")
        
        # Process PolarF11 data if uploaded
        if 'polarf11' in uploaded_files:
            try:
                polar_data = uploaded_files['polarf11']
                polar_df = pd.read_csv(polar_data)
                
                # Rename columns to standard format
                polar_df = polar_df.rename(columns=POLAR_F11_COLUMNS)
                polar_df['source'] = 'PolarF11'
                
                # Convert date and standardize activity types
                polar_df['date'] = pd.to_datetime(polar_df['date']).dt.date
                
                # Convert duration string to minutes if needed
                if 'duration_string' in polar_df.columns:
                    polar_df['duration'] = polar_df['duration_string'].apply(convert_duration_to_minutes)
                
                polar_df['activity_type'] = polar_df['activity_type'].map(
                    lambda x: ACTIVITY_TYPE_MAPPING.get(x, x) if isinstance(x, str) else x
                )
                
                dataframes.append(polar_df)
            except Exception as e:
                st.error(f"Error processing PolarF11 data: {e}")
        
        # Process ChargeHR data if uploaded
        if 'chargehr' in uploaded_files:
            try:
                charge_data = uploaded_files['chargehr']
                charge_df = pd.read_csv(charge_data)
                
                # Rename columns to standard format
                charge_df = charge_df.rename(columns=CHARGE_HR_COLUMNS)
                charge_df['source'] = 'ChargeHR'
                
                # Convert date and standardize activity types
                charge_df['date'] = pd.to_datetime(charge_df['date']).dt.date
                
                # Convert duration string to minutes if needed
                if 'duration_string' in charge_df.columns:
                    charge_df['duration'] = charge_df['duration_string'].apply(convert_duration_to_minutes)
                
                charge_df['activity_type'] = charge_df['activity_type'].map(
                    lambda x: ACTIVITY_TYPE_MAPPING.get(x, x) if isinstance(x, str) else x
                )
                
                # Estimate heart rate from calories if needed
                if 'calories' in charge_df.columns and ('avg_hr' not in charge_df.columns or charge_df['avg_hr'].isna().all()):
                    charge_df['avg_hr'] = charge_df['calories'] / 4
                
                dataframes.append(charge_df)
            except Exception as e:
                st.error(f"Error processing ChargeHR data: {e}")
        
        if not dataframes:
            return create_empty_df()
        
        # Combine all dataframes
        standard_columns = ['date', 'activity_type', 'avg_hr', 'max_hr', 'calories', 'duration', 'source', 'location']
        
        # Ensure all standard columns exist in each dataframe
        for df in dataframes:
            for col in standard_columns:
                if col not in df.columns:
                    df[col] = None
            
            # Keep only standard columns
            df = df[standard_columns]
        
        combined_df = pd.concat(dataframes, ignore_index=True)
        combined_df = combined_df.sort_values('date')
        
        # Convert numeric columns
        for col in ['avg_hr', 'max_hr', 'calories', 'duration']:
            combined_df[col] = pd.to_numeric(combined_df[col], errors='coerce')
        
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

def aggregate_by_date(df, metrics=None):
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

# Visualization functions
def plot_time_series(df, metrics, title="Metrics Over Time"):
    """Create a time series line chart for selected metrics."""
    if df.empty or not metrics:
        st.warning("No data available for the selected filters.")
        return
    
    fig = go.Figure()
    
    for metric in metrics:
        for source in df['source'].unique():
            source_data = df[df['source'] == source]
            if not source_data.empty:
                fig.add_trace(
                    go.Scatter(
                        x=source_data['date'],
                        y=source_data[metric],
                        mode='lines+markers',
                        name=f"{source} - {metric}",
                        line=dict(width=2, shape='spline')
                    )
                )
    
    fig.update_layout(
        title=title,
        xaxis_title="Date",
        yaxis_title="Value",
        height=PLOT_HEIGHT,
        legend=dict(orientation="h", yanchor="bottom", y=-0.3, xanchor="center", x=0.5),
        margin=dict(l=20, r=20, t=40, b=20),
        hovermode="x unified"
    )
    
    st.plotly_chart(fig, use_container_width=True)

def plot_activity_distribution(df, title="Activity Type Distribution"):
    """Create a bar chart showing activity type distribution by source."""
    if df.empty:
        st.warning("No data available for the selected filters.")
        return
    
    activity_counts = df.groupby(['source', 'activity_type']).size().reset_index(name='count')
    
    fig = px.bar(
        activity_counts,
        x='activity_type',
        y='count',
        color='source',
        barmode='group',
        title=title
    )
    
    fig.update_layout(
        xaxis_title="Activity Type",
        yaxis_title="Count",
        height=PLOT_HEIGHT,
        margin=dict(l=20, r=20, t=40, b=20),
        xaxis={'categoryorder':'total descending'}
    )
    
    st.plotly_chart(fig, use_container_width=True)

def plot_scatter_matrix(df, dimensions, color_by="source", title="Relationships Between Metrics"):
    """Create a scatter matrix plot for exploring relationships between metrics."""
    if df.empty or not dimensions:
        st.warning("No data available for the selected filters.")
        return
    
    # Validate dimensions
    valid_dimensions = [dim for dim in dimensions if dim in df.columns]
    
    if not valid_dimensions:
        st.warning("No valid dimensions selected for scatter matrix.")
        return
    
    fig = px.scatter_matrix(
        df,
        dimensions=valid_dimensions,
        color=color_by,
        title=title,
        opacity=0.7
    )
    
    fig.update_layout(
        height=PLOT_HEIGHT,
        margin=dict(l=20, r=20, t=40, b=20)
    )
    
    # Update traces
    fig.update_traces(diagonal_visible=False, showupperhalf=False)
    
    st.plotly_chart(fig, use_container_width=True)

def plot_duration_distribution(df, title="Activity Duration Distribution"):
    """Create a box plot showing activity duration distribution."""
    if df.empty:
        st.warning("No data available for the selected filters.")
        return
    
    groupby = st.radio(
        "Group duration by:",
        options=["Source", "Activity Type", "Both"],
        horizontal=True
    )
    
    x_column = 'activity_type' if groupby in ["Activity Type", "Both"] else 'source'
    color_column = 'source' if groupby in ["Source", "Both"] else 'activity_type'
    
    fig = px.box(
        df,
        x=x_column,
        y='duration',
        color=color_column,
        points="all",
        title=title
    )
    
    fig.update_layout(
        xaxis_title=x_column.replace('_', ' ').title(),
        yaxis_title="Duration (minutes)",
        height=PLOT_HEIGHT,
        margin=dict(l=20, r=20, t=40, b=20),
        xaxis={'categoryorder':'mean descending'}
    )
    
    st.plotly_chart(fig, use_container_width=True)

# UI Components
def upload_component():
    """File upload component for cloud deployment."""
    st.sidebar.header("Data Upload")
    
    # Initialize uploaded files in session state if not already there
    if 'uploaded_files' not in st.session_state:
        st.session_state.uploaded_files = {}
    
    # File upload widgets
    miifit_file = st.sidebar.file_uploader("Upload MiiFit Data (CSV)", type=['csv'])
    garmin_file = st.sidebar.file_uploader("Upload Garmin Data (CSV)", type=['csv'])
    polar_file = st.sidebar.file_uploader("Upload PolarF11 Data (CSV)", type=['csv'])
    charge_file = st.sidebar.file_uploader("Upload ChargeHR Data (CSV)", type=['csv'])
    
    # Store uploaded files in session state
    if miifit_file is not None:
        st.session_state.uploaded_files['miifit'] = miifit_file
    
    if garmin_file is not None:
        st.session_state.uploaded_files['garmin'] = garmin_file
    
    if polar_file is not None:
        st.session_state.uploaded_files['polarf11'] = polar_file
    
    if charge_file is not None:
        st.session_state.uploaded_files['chargehr'] = charge_file
    
    # Clear uploaded files button
    if st.sidebar.button("Clear Uploaded Files"):
        st.session_state.uploaded_files = {}
        st.experimental_rerun()

def sidebar_controls(df):
    """Create the sidebar controls matching the original dashboard."""
    st.sidebar.header("Controls")
    
    # Date range selector
    min_date = df['date'].min() if not df.empty and 'date' in df.columns else datetime(2020, 1, 1).date()
    max_date = df['date'].max() if not df.empty and 'date' in df.columns else datetime.now().date()
    default_start = max_date - timedelta(days=30) if not pd.isna(max_date) else datetime.now().date() - timedelta(days=30)
    
    start_date = st.sidebar.date_input(
        "Start Date",
        default_start,
        min_value=min_date if not pd.isna(min_date) else datetime(2020, 1, 1).date(),
        max_value=max_date if not pd.isna(max_date) else datetime.now().date()
    )
    
    end_date = st.sidebar.date_input(
        "End Date",
        max_date if not pd.isna(max_date) else datetime.now().date(),
        min_value=start_date,
        max_value=max_date if not pd.isna(max_date) else datetime.now().date()
    )
    
    # Source selector - simplified to match original dashboard
    available_sources = sorted(df['source'].unique()) if not df.empty else ['MiiFit', 'Garmin', 'PolarF11', 'ChargeHR']
    selected_sources = st.sidebar.multiselect(
        "Select Data Sources",
        options=available_sources,
        default=available_sources[:2] if len(available_sources) >= 2 else available_sources
    )
    
    # Optional activity type selector
    st.sidebar.markdown("---")
    show_activity_filter = st.sidebar.checkbox("Filter by Activity Type", False)
    
    selected_activities = []
    if show_activity_filter and not df.empty:
        available_activities = sorted(df['activity_type'].unique()) 
        selected_activities = st.sidebar.multiselect(
            "Select Activity Types",
            options=available_activities,
            default=[]
        )
    
    # Return the filters
    return {
        'start_date': start_date,
        'end_date': end_date,
        'sources': selected_sources,
        'activity_types': selected_activities
    }

def show_quick_overview_tab(filtered_df):
    """Display a simplified overview dashboard like the original."""
    st.title("Heart Rate Monitor Dashboard")
    
    # Summary statistics (matching the original dashboard)
    st.header("Summary Statistics")
    stats = get_summary_stats(filtered_df)
    st.dataframe(stats, use_container_width=True)
    
    # Heart rate line chart (matching the original dashboard)
    st.header("Heart Rate Over Time")
    hr_metrics = st.multiselect(
        "Select HR Metrics",
        ['avg_hr', 'max_hr'],
        default=['avg_hr', 'max_hr']
    )
    
    if hr_metrics and not filtered_df.empty:
        hr_data = aggregate_by_date(filtered_df, hr_metrics)
        plot_time_series(hr_data, hr_metrics, "Heart Rate Over Time")
    
    # Activity duration distribution
    st.header("Activity Duration")
    plot_duration_distribution(filtered_df)
    
    # Scatter matrix for metric relationships
    st.header("Metric Relationships")
    dimensions = ['avg_hr', 'max_hr', 'calories', 'duration']
    valid_dimensions = [dim for dim in dimensions if dim in filtered_df.columns]
    
    if valid_dimensions:
        plot_scatter_matrix(filtered_df, valid_dimensions)
    
    # Activity breakdown
    st.header("Activity Type Distribution")
    plot_activity_distribution(filtered_df)

def show_heart_rate_tab(filtered_df):
    """Display Heart Rate Analysis tab content."""
    st.header("Heart Rate Analysis")
    
    # Heart rate metrics selection
    hr_metrics = st.multiselect(
        "Select Heart Rate Metrics",
        options=['avg_hr', 'max_hr'],
        default=['avg_hr', 'max_hr'],
        format_func=lambda x: "Average HR" if x == 'avg_hr' else "Maximum HR"
    )
    
    if hr_metrics:
        # Time series data aggregated by date
        hr_data = aggregate_by_date(filtered_df, hr_metrics)
        plot_time_series(hr_data, hr_metrics, "Heart Rate Over Time")
    
    # Heart rate by activity type
    st.subheader("Heart Rate by Activity Type")
    
    # Calculate average heart rates by activity type
    if not filtered_df.empty and 'activity_type' in filtered_df.columns:
        hr_by_activity = filtered_df.groupby(['source', 'activity_type'])[['avg_hr', 'max_hr']].mean().reset_index()
        
        if not hr_by_activity.empty:
            fig = px.bar(
                hr_by_activity,
                x='activity_type',
                y=['avg_hr', 'max_hr'],
                color='source',
                barmode='group',
                title="Average Heart Rates by Activity Type"
            )
            
            fig.update_layout(
                xaxis_title="Activity Type",
                yaxis_title="Heart Rate (bpm)",
                legend_title="Source",
                height=PLOT_HEIGHT
            )
            
            st.plotly_chart(fig, use_container_width=True)

def show_raw_data_tab(df):
    """Display raw data tab content."""
    st.header("Raw Data")
    
    # Show filtered data
    st.dataframe(df, use_container_width=True)
    
    # Allow CSV download
    if not df.empty:
        csv = df.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="Download Data as CSV",
            data=csv,
            file_name="fitness_data.csv",
            mime="text/csv"
        )

# Main app
def main():
    """Main function to run the dashboard."""
    st.set_page_config(
        page_title="Fitness Dashboard",
        page_icon="🏃",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Apply custom CSS
    st.markdown("""
    <style>
        .block-container {padding-top: 1rem; padding-bottom: 1rem;}
        .stTabs [data-baseweb="tab-panel"] {padding-top: 0.5rem;}
    </style>
    """, unsafe_allow_html=True)
    
    # Add title
    st.title("Fitness Dashboard")
    st.write("A dashboard for visualizing and analyzing fitness tracking data from multiple sources.")
    
    # File upload component in sidebar
    upload_component()
    
    # Load data
    with st.spinner("Loading data..."):
        df = load_data()
    
    # Display message if no data
    if df.empty and not st.session_state.get('use_sample_data', False):
        st.info("👈 Please upload your fitness data files using the sidebar or check the 'Use sample data' option.")
        st.markdown("""
        ## Supported Data Files
        
        This dashboard supports data from the following fitness trackers:
        - **MiiFit**: CSV export with activity data
        - **Garmin**: CSV export with activity data
        - **PolarF11**: CSV export with activity data
        - **ChargeHR**: CSV export with activity data
        
        ## Sample Data
        
        You can check "Use sample data" in the sidebar to see the dashboard with example data.
        """)
        return
    
    # Create sidebar controls
    filters = sidebar_controls(df)
    
    # Filter data based on selection
    filtered_df = filter_data(
        df,
        filters['start_date'],
        filters['end_date'],
        filters['sources'],
        filters['activity_types']
    )
    
    # Handle no data after filtering
    if filtered_df.empty:
        st.warning("No data available for the selected filters. Please adjust your selection or upload different data files.")
        return
    
    # Dashboard tabs
    tab_names = [
        "Quick Overview",
        "Heart Rate Analysis",
        "Activity Distribution", 
        "Duration Analysis",
        "Metric Relationships",
        "Raw Data"
    ]
    
    # Create tabs
    tabs = st.tabs(tab_names)
    
    # Tab content
    with tabs[0]:  # Quick Overview
        show_quick_overview_tab(filtered_df)
        
    with tabs[1]:  # Heart Rate Analysis
        show_heart_rate_tab(filtered_df)
    
    with tabs[2]:  # Activity Distribution
        st.header("Activity Distribution")
        plot_activity_distribution(filtered_df)
        
        # Activity trends over time
        st.subheader("Activity Trends Over Time")
        
        if not filtered_df.empty and 'date' in filtered_df.columns and 'activity_type' in filtered_df.columns:
            # Create month column
            trend_df = filtered_df.copy()
            trend_df['month'] = pd.to_datetime(trend_df['date']).dt.to_period('M')
            
            # Group by month and activity type
            monthly_activities = trend_df.groupby(['month', 'activity_type', 'source']).size().reset_index(name='count')
            monthly_activities['month'] = monthly_activities['month'].dt.to_timestamp()
            
            if not monthly_activities.empty:
                fig = px.line(
                    monthly_activities,
                    x='month',
                    y='count',
                    color='activity_type',
                    line_dash='source',
                    markers=True,
                    title="Monthly Activity Frequency"
                )
                
                fig.update_layout(
                    xaxis_title="Month",
                    yaxis_title="Number of Activities",
                    legend_title="Activity Type",
                    height=PLOT_HEIGHT
                )
                
                st.plotly_chart(fig, use_container_width=True)
        
        # Calorie breakdown by activity type
        st.subheader("Calories by Activity Type")
        
        if 'calories' in filtered_df.columns:
            calories_by_activity = filtered_df.groupby(['source', 'activity_type'])['calories'].sum().reset_index()
            
            if not calories_by_activity.empty:
                fig = px.pie(
                    calories_by_activity,
                    values='calories',
                    names='activity_type',
                    color='activity_type',
                    facet_col='source',
                    title="Total Calories by Activity Type"
                )
                
                fig.update_layout(height=PLOT_HEIGHT)
                st.plotly_chart(fig, use_container_width=True)
    
    with tabs[3]:  # Duration Analysis
        st.header("Duration Analysis")
        
        # Duration distribution
        st.subheader("Activity Duration Distribution")
        plot_duration_distribution(filtered_df)
        
        # Duration over time
        st.subheader("Activity Duration Over Time")
        
        if not filtered_df.empty and 'date' in filtered_df.columns and 'duration' in filtered_df.columns:
            duration_by_date = filtered_df.groupby(['date', 'source'])['duration'].sum().reset_index()
            
            if not duration_by_date.empty:
                fig = px.line(
                    duration_by_date,
                    x='date',
                    y='duration',
                    color='source',
                    markers=True,
                    title="Total Activity Duration by Date"
                )
                
                fig.update_layout(
                    xaxis_title="Date",
                    yaxis_title="Duration (minutes)",
                    height=PLOT_HEIGHT
                )
                
                st.plotly_chart(fig, use_container_width=True)
        
        # Weekly duration totals
        st.subheader("Weekly Activity Duration")
        
        if not filtered_df.empty and 'date' in filtered_df.columns and 'duration' in filtered_df.columns:
            weekly_df = filtered_df.copy()
            weekly_df['week'] = pd.to_datetime(weekly_df['date']).dt.to_period('W')
            
            # Group by week and source
            weekly_duration = weekly_df.groupby(['week', 'source'])['duration'].sum().reset_index()
            weekly_duration['week'] = weekly_duration['week'].dt.to_timestamp()
            
            if not weekly_duration.empty:
                fig = px.bar(
                    weekly_duration,
                    x='week',
                    y='duration',
                    color='source',
                    barmode='group',
                    title="Weekly Activity Duration"
                )
                
                fig.update_layout(
                    xaxis_title="Week",
                    yaxis_title="Duration (minutes)",
                    height=PLOT_HEIGHT
                )
                
                st.plotly_chart(fig, use_container_width=True)
    
    with tabs[4]:  # Metric Relationships
        st.header("Metric Relationships")
        
        # Scatter matrix of key metrics
        st.subheader("Metrics Scatter Matrix")
        
        dimensions = st.multiselect(
            "Select Metrics to Compare",
            options=['avg_hr', 'max_hr', 'calories', 'duration'],
            default=['avg_hr', 'max_hr', 'calories', 'duration'],
            format_func=lambda x: x.replace('_', ' ').title()
        )
        
        color_by = st.radio(
            "Color by:",
            options=['source', 'activity_type'],
            format_func=lambda x: x.replace('_', ' ').title(),
            horizontal=True
        )
        
        plot_scatter_matrix(filtered_df, dimensions, color_by)
        
        # Correlation heatmap
        st.subheader("Correlation Between Metrics")
        
        if not filtered_df.empty and all(dim in filtered_df.columns for dim in dimensions):
            # Calculate correlation matrix
            corr_df = filtered_df[dimensions].corr()
            
            # Create heatmap
            fig = px.imshow(
                corr_df,
                text_auto=True,
                color_continuous_scale='RdBu_r',
                title="Correlation Matrix"
            )
            
            fig.update_layout(height=PLOT_HEIGHT)
            st.plotly_chart(fig, use_container_width=True)
    
    with tabs[5]:  # Raw Data
        show_raw_data_tab(filtered_df)

if __name__ == "__main__":
    main()
