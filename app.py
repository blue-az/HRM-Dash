"""
Fitness Dashboard - Main Application

A Streamlit dashboard for visualizing and analyzing fitness data
from multiple sources (MiiFit, Garmin, Polar F11, and Charge HR).
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import os
from datetime import datetime, timedelta

# Import local modules
import config
import data_processor as dp
import visualization as viz
import utils

# Page configuration
st.set_page_config(
    page_title="Fitness Data Dashboard",
    page_icon="🏃",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Apply custom CSS for better spacing
st.markdown("""
<style>
    .block-container {padding-top: 1rem; padding-bottom: 1rem;}
    .stTabs [data-baseweb="tab-panel"] {padding-top: 0.5rem;}
</style>
""", unsafe_allow_html=True)

# Load data with caching
@st.cache_data(ttl=600)  # Cache data for 10 minutes
def load_data():
    """Load and combine data from all sources."""
    return dp.combine_data()

def sidebar_controls(df):
    """Create the sidebar controls matching the original dashboard."""
    st.sidebar.header("Controls")
    
    # Date range selector
    min_date = df['date'].min() if not df.empty and 'date' in df.columns else datetime(2020, 1, 1).date()
    max_date = df['date'].max() if not df.empty and 'date' in df.columns else datetime.now().date()
    default_start = max_date - timedelta(days=30)
    
    start_date = st.sidebar.date_input(
        "Start Date",
        default_start,
        min_value=min_date,
        max_value=max_date
    )
    
    end_date = st.sidebar.date_input(
        "End Date",
        max_date,
        min_value=start_date,
        max_value=max_date
    )
    
    # Source selector - simplified to match original dashboard
    available_sources = sorted(df['source'].unique()) if not df.empty else ['MiiFit', 'Garmin']
    selected_sources = st.sidebar.multiselect(
        "Select Data Sources",
        options=available_sources,
        default=['MiiFit', 'Garmin'] if 'MiiFit' in available_sources and 'Garmin' in available_sources else available_sources[:2] if len(available_sources) >= 2 else available_sources
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
    numeric_cols = ['avg_hr', 'max_hr', 'calories', 'duration']
    stats = filtered_df[numeric_cols].agg(['mean', 'std', 'min', 'max']).round(2)
    stats.index = ['Average', 'Std Dev', 'Minimum', 'Maximum']
    st.dataframe(stats, use_container_width=True)
    
    # Heart rate line chart (matching the original dashboard)
    st.header("Heart Rate Over Time")
    hr_metrics = st.multiselect(
        "Select HR Metrics",
        ['avg_hr', 'max_hr'],
        default=['avg_hr', 'max_hr']
    )
    
    if hr_metrics and not filtered_df.empty:
        fig = go.Figure()
        for metric in hr_metrics:
            for source in filtered_df['source'].unique():
                source_data = filtered_df[filtered_df['source'] == source]
                fig.add_trace(go.Scatter(
                    x=source_data['date'],
                    y=source_data[metric],
                    name=f"{source} - {metric}",
                    mode='lines+markers'
                ))
        
        fig.update_layout(
            height=config.PLOT_HEIGHT,
            width=config.PLOT_WIDTH,
            xaxis_title="Date",
            yaxis_title="Heart Rate (bpm)",
            legend=dict(orientation="h", yanchor="bottom", y=-0.3, xanchor="center", x=0.5),
        )
        st.plotly_chart(fig, use_container_width=True)
    
    # Activity duration distribution
    st.header("Activity Duration")
    fig_duration = px.box(
        filtered_df,
        x='source',
        y='duration',
        color='source',
        points='all',
        title="Activity Duration Distribution"
    )
    fig_duration.update_layout(height=config.PLOT_HEIGHT)
    st.plotly_chart(fig_duration, use_container_width=True)
    
    # Scatter matrix for metric relationships
    st.header("Metric Relationships")
    fig_scatter = px.scatter_matrix(
        filtered_df,
        dimensions=['avg_hr', 'max_hr', 'calories', 'duration'],
        color='source',
        title="Relationships Between Metrics"
    )
    fig_scatter.update_layout(height=config.PLOT_HEIGHT)
    st.plotly_chart(fig_scatter, use_container_width=True)
    
    # Activity breakdown
    st.header("Activity Type Distribution")
    activity_counts = filtered_df.groupby(['source', 'activity_type']).size().reset_index(name='count')
    fig_activities = px.bar(
        activity_counts,
        x='activity_type',
        y='count',
        color='source',
        barmode='group',
        title="Activity Types by Source"
    )
    fig_activities.update_layout(height=config.PLOT_HEIGHT)
    st.plotly_chart(fig_activities, use_container_width=True)

def show_overview_tab(filtered_df):
    """Display content for the detailed Overview tab."""
    st.header("Fitness Dashboard Overview")
    
    # Show data source information
    st.subheader("Data Sources Status")
    st.write("This dashboard shows information from the following data sources:")
    
    # Check which sources are loaded
    data_sources = {
        source: any(filtered_df['source'] == source) if not filtered_df.empty else False
        for source in ['MiiFit', 'Garmin', 'PolarF11', 'ChargeHR']
    }
    
    for source, present in data_sources.items():
        st.write(f"- {source}: {'✅ Loaded' if present else '❌ Not loaded'}")
    
    # Show source counts if data is available
    if not filtered_df.empty:
        st.write("Count of activities by source:")
        source_counts = filtered_df['source'].value_counts().reset_index()
        source_counts.columns = ['Source', 'Count']
        st.dataframe(source_counts)
    
    # Display summary metrics in columns
    col1, col2, col3 = st.columns(3)
    
    with col1:
        show_activity_count_metrics(filtered_df)
    
    with col2:
        show_date_range_metrics(filtered_df)
    
    with col3:
        show_average_metrics(filtered_df)
    
    # Summary statistics
    st.subheader("Summary Statistics")
    viz.display_summary_stats(filtered_df)
    
    # Show recent activities timeline
    show_recent_activities(filtered_df)

def show_activity_count_metrics(df):
    """Display activity count metrics."""
    st.subheader("Activity Count")
    total_activities = len(df)
    st.metric("Total Activities", total_activities)
    
    if 'source' in df.columns:
        source_counts = df['source'].value_counts()
        for source, count in source_counts.items():
            st.metric(f"{source} Activities", count)

def show_date_range_metrics(df):
    """Display date range metrics."""
    st.subheader("Date Range")
    if 'date' in df.columns:
        min_date = df['date'].min()
        max_date = df['date'].max()
        date_range = (max_date - min_date).days + 1 if hasattr(max_date - min_date, 'days') else "N/A"
        
        # Convert dates to strings for st.metric
        min_date_str = min_date.strftime('%Y-%m-%d') if min_date else "N/A"
        max_date_str = max_date.strftime('%Y-%m-%d') if max_date else "N/A"
        
        st.metric("Start Date", min_date_str)
        st.metric("End Date", max_date_str)
        st.metric("Days", date_range)

def show_average_metrics(df):
    """Display average fitness metrics."""
    st.subheader("Average Stats")
    
    metrics = {
        'duration': {'label': "Avg Duration", 'format': utils.format_duration},
        'calories': {'label': "Avg Calories", 'format': lambda x: f"{x:.1f}"},
        'avg_hr': {'label': "Avg Heart Rate", 'format': lambda x: f"{x:.1f} bpm"}
    }
    
    for metric, config in metrics.items():
        if metric in df.columns:
            avg_value = df[metric].mean()
            formatted_value = config['format'](avg_value)
            st.metric(config['label'], formatted_value)

def show_recent_activities(df, limit=10):
    """Display recent activities in an expandable timeline."""
    st.subheader("Activity Timeline")
    
    recent_activities = df.sort_values('date', ascending=False).head(limit)
    for _, activity in recent_activities.iterrows():
        with st.expander(f"{activity['date']} - {activity['activity_type']} ({activity['source']})"):
            cols = st.columns(4)
            
            # Display metrics in columns
            metrics = [
                ("Duration", utils.format_duration(activity['duration'])),
                ("Calories", f"{activity['calories']:.0f}" if not pd.isna(activity['calories']) else "N/A"),
                ("Avg HR", f"{activity['avg_hr']:.0f} bpm" if not pd.isna(activity['avg_hr']) else "N/A"),
                ("Max HR", f"{activity['max_hr']:.0f} bpm" if not pd.isna(activity['max_hr']) else "N/A")
            ]
            
            for i, (label, value) in enumerate(metrics):
                with cols[i]:
                    st.metric(label, value)

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
        hr_data = dp.aggregate_data_by_date(filtered_df, hr_metrics)
        viz.plot_time_series(hr_data, hr_metrics, "Heart Rate Over Time")
    
    # Heart rate zones
    st.subheader("Heart Rate Zones")
    viz.plot_heart_rate_zones(filtered_df)
    
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
                height=config.PLOT_HEIGHT
            )
            
            st.plotly_chart(fig, use_container_width=True)

def show_activity_distribution_tab(filtered_df):
    """Display Activity Distribution tab content."""
    st.header("Activity Distribution")
    
    # Activity type distribution
    st.subheader("Activity Type Distribution")
    viz.plot_activity_distribution(filtered_df)
    
    # Activity trends over time
    st.subheader("Activity Trends Over Time")
    
    # Plot monthly activity frequency
    monthly_counts = dp.calculate_monthly_activity_counts(filtered_df)
    if not monthly_counts.empty:
        fig = px.line(
            monthly_counts,
            x='date',
            y='count',
            color='activity_type',
            markers=True,
            title="Monthly Activity Frequency"
        )
        
        fig.update_layout(
            xaxis_title="Month",
            yaxis_title="Number of Activities",
            height=config.PLOT_HEIGHT
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    # Calorie breakdown by activity type
    st.subheader("Calories by Activity Type")
    
    if not filtered_df.empty and 'calories' in filtered_df.columns:
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
            
            fig.update_layout(height=config.PLOT_HEIGHT)
            st.plotly_chart(fig, use_container_width=True)

def show_duration_analysis_tab(filtered_df):
    """Display Duration Analysis tab content."""
    st.header("Duration Analysis")
    
    # Duration distribution
    st.subheader("Activity Duration Distribution")
    viz.plot_duration_distribution(filtered_df)
    
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
                height=config.PLOT_HEIGHT
            )
            
            st.plotly_chart(fig, use_container_width=True)

def show_metric_relationships_tab(filtered_df):
    """Display Metric Relationships tab content."""
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
    
    viz.plot_scatter_matrix(filtered_df, dimensions, color_by)
    
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
        
        fig.update_layout(height=config.PLOT_HEIGHT)
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

def main():
    """Main function to run the dashboard."""
    # Load data
    with st.spinner("Loading data..."):
        df = load_data()
    
    # Create sidebar controls - using the simplified version to match the original dashboard
    filters = sidebar_controls(df)
    
    # Filter data based on selection
    filtered_df = dp.filter_data(
        df,
        filters['start_date'],
        filters['end_date'],
        filters['sources'],
        filters['activity_types']
    )
    
    # Handle no data case
    if filtered_df.empty:
        st.warning("No data available for the selected filters. Please adjust your selection or check your data files.")
        
        # Display troubleshooting info
        with st.expander("Troubleshooting"):
            st.error(f"""
            ### Troubleshooting Data Loading Issues
            
            1. **Check File Paths**:
               - MiiFit: `{config.MIIFIT_PATH}`
               - Garmin: `{config.GARMIN_PATH}`
            
            2. **Check Data Format**:
               - MiiFit data should contain columns for date, heart rate, and activity type
               - Garmin data should contain similar information
            
            3. **Generate Sample Data**:
               If needed, you can generate sample data for testing.
            """)
        
        return
    
    # Dashboard tabs - adding a Quick Overview tab to match the original dashboard
    tab_names = [
        "Quick Overview",  # New tab matching the original dashboard
        "Detailed Overview", 
        "Heart Rate Analysis", 
        "Activity Distribution", 
        "Duration Analysis",
        "Metric Relationships",
        "Raw Data"
    ]
    
    # Create tabs
    tabs = st.tabs(tab_names)
    
    # Tab content
    with tabs[0]:  # Quick Overview - matches the original dashboard layout
        show_quick_overview_tab(filtered_df)
        
    with tabs[1]:  # Detailed Overview - our existing implementation
        show_overview_tab(filtered_df)
    
    with tabs[2]:  # Heart Rate Analysis
        show_heart_rate_tab(filtered_df)
    
    with tabs[3]:  # Activity Distribution
        show_activity_distribution_tab(filtered_df)
    
    with tabs[4]:  # Duration Analysis
        show_duration_analysis_tab(filtered_df)
    
    with tabs[5]:  # Metric Relationships
        show_metric_relationships_tab(filtered_df)
    
    with tabs[6]:  # Raw Data
        show_raw_data_tab(filtered_df)

# Run the application
if __name__ == "__main__":
    main()
