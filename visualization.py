"""Visualization modules for fitness data dashboard."""

import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import config

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
        height=config.PLOT_HEIGHT,
        legend=dict(orientation="h", yanchor="bottom", y=-0.3, xanchor="center", x=0.5),
        margin=dict(l=20, r=20, t=40, b=20),
        hovermode="x unified"
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
        height=config.PLOT_HEIGHT,
        margin=dict(l=20, r=20, t=40, b=20)
    )
    
    # Update traces
    fig.update_traces(diagonal_visible=False, showupperhalf=False)
    
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
        height=config.PLOT_HEIGHT,
        margin=dict(l=20, r=20, t=40, b=20),
        xaxis={'categoryorder':'total descending'}
    )
    
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
        height=config.PLOT_HEIGHT,
        margin=dict(l=20, r=20, t=40, b=20),
        xaxis={'categoryorder':'mean descending'}
    )
    
    st.plotly_chart(fig, use_container_width=True)

def plot_heart_rate_zones(df, title="Heart Rate Zone Distribution"):
    """Create heart rate zone distribution visualization."""
    if df.empty:
        st.warning("No data available for the selected filters.")
        return
    
    # Define heart rate zones based on max HR
    df_copy = df.copy()
    
    # Define zones (percentages of max HR)
    zone_ranges = {
        'Zone 1 (Very Light)': (0.5, 0.6),
        'Zone 2 (Light)': (0.6, 0.7),
        'Zone 3 (Moderate)': (0.7, 0.8),
        'Zone 4 (Hard)': (0.8, 0.9),
        'Zone 5 (Maximum)': (0.9, 1.0)
    }
    
    # Calculate time spent in each zone
    zone_times = []
    
    for source in df_copy['source'].unique():
        source_df = df_copy[df_copy['source'] == source]
        
        for activity_type in source_df['activity_type'].unique():
            activity_df = source_df[source_df['activity_type'] == activity_type]
            
            avg_hr = activity_df['avg_hr'].mean()
            max_hr = activity_df['max_hr'].mean()
            total_duration = activity_df['duration'].sum()
            
            if pd.isna(avg_hr) or pd.isna(max_hr) or pd.isna(total_duration):
                continue
                
            # Estimate zone based on average HR as percentage of max HR
            hr_percentage = avg_hr / max_hr if max_hr > 0 else 0
            
            for zone_name, (min_pct, max_pct) in zone_ranges.items():
                if min_pct <= hr_percentage < max_pct:
                    zone_times.append({
                        'source': source,
                        'activity_type': activity_type,
                        'zone': zone_name,
                        'duration': total_duration
                    })
                    break
    
    if not zone_times:
        st.warning("Not enough heart rate data to calculate zones.")
        return
        
    zone_df = pd.DataFrame(zone_times)
    
    fig = px.sunburst(
        zone_df,
        path=['source', 'zone', 'activity_type'],
        values='duration',
        title=title
    )
    
    fig.update_layout(
        height=config.PLOT_HEIGHT,
        margin=dict(l=20, r=20, t=40, b=20)
    )
    
    st.plotly_chart(fig, use_container_width=True)

def plot_multi_source_comparison(df, title="Data Source Comparison"):
    """Create visualizations comparing data across different sources."""
    if df.empty:
        st.warning("No data available for the selected filters.")
        return
    
    # Get available sources
    sources = df['source'].unique()
    
    if len(sources) <= 1:
        st.info("At least two data sources are needed for comparison. Please select more data sources in the filters.")
        return
    
    st.subheader("Data Source Comparison")
    
    # Create comparison plots for different metrics
    metrics = [
        {'column': 'avg_hr', 'title': 'Average Heart Rate by Activity Type', 'label': 'Average Heart Rate (bpm)'},
        {'column': 'calories', 'title': 'Average Calories by Activity Type', 'label': 'Average Calories'},
        {'column': 'duration', 'title': 'Average Duration by Activity Type', 'label': 'Average Duration (minutes)'}
    ]
    
    for metric in metrics:
        if metric['column'] in df.columns:
            metric_by_activity = df.groupby(['source', 'activity_type'])[metric['column']].mean().reset_index()
            
            if not metric_by_activity.empty and not metric_by_activity[metric['column']].isna().all():
                fig = px.bar(
                    metric_by_activity,
                    x='activity_type',
                    y=metric['column'],
                    color='source',
                    barmode='group',
                    title=metric['title']
                )
                
                fig.update_layout(
                    xaxis_title="Activity Type",
                    yaxis_title=metric['label'],
                    height=config.PLOT_HEIGHT
                )
                
                st.plotly_chart(fig, use_container_width=True)
    
    # Activity count comparison
    activity_counts = df.groupby(['source', 'activity_type']).size().reset_index(name='count')
    
    fig = px.bar(
        activity_counts,
        x='activity_type',
        y='count',
        color='source',
        barmode='group',
        title="Activity Count by Type and Source"
    )
    
    fig.update_layout(
        xaxis_title="Activity Type",
        yaxis_title="Number of Activities",
        height=config.PLOT_HEIGHT
    )
    
    st.plotly_chart(fig, use_container_width=True)

def plot_bike_analysis(df, title="Bike Route Analysis"):
    """Create visualizations for cycling routes and speeds."""
    # Filter to only biking activities
    bike_df = df[df['activity_type'].isin(['Outdoor Cycling', 'Indoor Cycling', 'Biking'])]
    
    if bike_df.empty:
        st.warning("No biking activities found in the selected data.")
        return
    
    # Show mph data if available    
    if 'mph' in bike_df.columns and not bike_df['mph'].isna().all():
        if 'location' in bike_df.columns and not bike_df['location'].isna().all():
            # Group by location and calculate stats
            route_stats = bike_df.groupby('location').agg({
                'mph': ['mean', 'max', 'min', 'std', 'count'],
                'duration': ['mean'],
                'avg_hr': ['mean'],
                'calories': ['mean', 'sum']
            }).reset_index()
            
            # Flatten the multi-index
            route_stats.columns = ['_'.join(col).strip('_') for col in route_stats.columns.values]
            
            # Create route comparison chart
            fig_routes = px.bar(
                route_stats,
                x='location',
                y='mph_mean',
                error_y=route_stats['mph_std'],
                text='mph_count',
                color='avg_hr_mean',
                hover_data=['duration_mean', 'calories_mean', 'mph_max', 'mph_min'],
                title="Average Speed by Route"
            )
            
            fig_routes.update_layout(height=config.PLOT_HEIGHT)
            st.plotly_chart(fig_routes, use_container_width=True)
            
            # Display route statistics table
            st.subheader("Route Statistics")
            
            # Format table for display
            display_cols = {
                'location': 'Route',
                'mph_mean': 'Avg Speed (MPH)',
                'mph_max': 'Max Speed (MPH)',
                'duration_mean': 'Avg Duration (min)',
                'avg_hr_mean': 'Avg Heart Rate',
                'calories_mean': 'Avg Calories',
                'mph_count': 'Count'
            }
            
            display_df = route_stats[list(display_cols.keys())].rename(columns=display_cols)
            display_df = display_df.round(2)
            
            st.dataframe(display_df, use_container_width=True)
    
    # Show duration by date
    st.subheader("Cycling Duration Over Time")
    
    if 'date' in bike_df.columns and 'duration' in bike_df.columns:
        fig_duration = px.line(
            bike_df.sort_values('date'),
            x='date',
            y='duration',
            color='source',
            markers=True,
            title="Cycling Duration Over Time"
        )
        
        fig_duration.update_layout(
            xaxis_title="Date",
            yaxis_title="Duration (minutes)",
            height=config.PLOT_HEIGHT
        )
        
        st.plotly_chart(fig_duration, use_container_width=True)
    
    # Show heart rate vs. duration
    st.subheader("Heart Rate vs. Duration for Cycling")
    
    if all(col in bike_df.columns for col in ['avg_hr', 'duration']):
        hr_duration_df = bike_df.dropna(subset=['avg_hr', 'duration'])
        
        if not hr_duration_df.empty:
            fig_hr = px.scatter(
                hr_duration_df,
                x='duration',
                y='avg_hr',
                color='source',
                size='calories' if 'calories' in hr_duration_df.columns else None,
                hover_data=['date', 'activity_type'],
                trendline='ols',
                title="Heart Rate vs. Duration"
            )
            
            fig_hr.update_layout(
                xaxis_title="Duration (minutes)",
                yaxis_title="Average Heart Rate (bpm)",
                height=config.PLOT_HEIGHT
            )
            
            st.plotly_chart(fig_hr, use_container_width=True)

def plot_tennis_analysis(df, title="Tennis Analysis"):
    """Create visualizations for tennis activities."""
    # Filter to only tennis activities
    tennis_df = df[df['activity_type'] == 'Tennis']
    
    if tennis_df.empty:
        st.warning("No tennis activities found in the selected data.")
        return
    
    # Tennis activity frequency
    st.subheader("Tennis Activity Frequency")
    
    # Group by month
    tennis_df['month'] = pd.to_datetime(tennis_df['date']).dt.to_period('M')
    monthly_counts = tennis_df.groupby(['month', 'source']).size().reset_index(name='count')
    monthly_counts['month'] = monthly_counts['month'].dt.to_timestamp()
    
    fig_freq = px.bar(
        monthly_counts,
        x='month',
        y='count',
        color='source',
        title="Tennis Activity Frequency by Month"
    )
    
    fig_freq.update_layout(
        xaxis_title="Month",
        yaxis_title="Number of Sessions",
        height=config.PLOT_HEIGHT
    )
    
    st.plotly_chart(fig_freq, use_container_width=True)
    
    # Tennis metrics over time
    metrics = [
        {'column': 'avg_hr', 'title': 'Average Heart Rate During Tennis', 'label': 'Average Heart Rate (bpm)'},
        {'column': 'duration', 'title': 'Tennis Session Duration Over Time', 'label': 'Duration (minutes)'},
        {'column': 'calories', 'title': 'Calories Burned During Tennis', 'label': 'Calories'}
    ]
    
    for metric in metrics:
        if metric['column'] in tennis_df.columns and not tennis_df[metric['column']].isna().all():
            fig = px.line(
                tennis_df.sort_values('date'),
                x='date',
                y=metric['column'],
                color='source',
                markers=True,
                title=metric['title']
            )
            
            fig.update_layout(
                xaxis_title="Date",
                yaxis_title=metric['label'],
                height=config.PLOT_HEIGHT
            )
            
            st.plotly_chart(fig, use_container_width=True)
    
    # Tennis statistics by location
    if 'location' in tennis_df.columns and not tennis_df['location'].isna().all():
        st.subheader("Tennis Statistics by Location")
        
        location_stats = tennis_df.groupby('location').agg({
            'avg_hr': ['mean', 'std'],
            'duration': ['mean', 'std'],
            'calories': ['mean', 'sum'],
            'date': 'count'
        }).reset_index()
        
        # Flatten the multi-index
        location_stats.columns = ['_'.join(col).strip('_') for col in location_stats.columns.values]
        
        # Rename count column
        location_stats.rename(columns={'date_count': 'count'}, inplace=True)
        
        # Display table
        st.dataframe(location_stats.round(2), use_container_width=True)

def display_summary_stats(df):
    """Display summary statistics for the filtered data."""
    if df.empty:
        st.warning("No data available for the selected filters.")
        return
    
    stats = df.groupby('source')[['avg_hr', 'max_hr', 'calories', 'duration']].agg(['mean', 'max']).round(1)
    
    # Reformat the stats for better display
    stats.columns = [f"{col[0]} ({col[1]})" for col in stats.columns]
    
    st.dataframe(stats, use_container_width=True)
    
    # Show total activities count and date range
    st.metric("Total Activities", len(df))
    
    if 'date' in df.columns and not df['date'].empty:
        st.text(f"Date Range: {df['date'].min()} to {df['date'].max()}")
