# Fitness Dashboard

A streamlit-based dashboard for visualizing and analyzing fitness tracking data from multiple sources.

## Overview

This application provides a comprehensive visualization dashboard for fitness data, supporting multiple data sources:
- MiiFit
- Garmin
- PolarF11
- ChargeHR

The dashboard offers interactive filtering, various visualization types, and data analysis features to help track fitness metrics over time.

## Features

- Interactive date range selection
- Data source filtering
- Activity type filtering
- Multiple dashboard views:
  - Quick Overview (simplified view)
  - Detailed Overview
  - Heart Rate Analysis
  - Activity Distribution
  - Duration Analysis
  - Metric Relationships
  - Bike Analysis
  - Tennis Analysis 
  - Source Comparison
  - Raw Data Viewer

## Quick Overview Dashboard
The Quick Overview tab provides a simplified dashboard similar to the original heart rate monitor dashboard, showing:
- Summary statistics
- Heart rate trends over time
- Activity duration distribution
- Metric relationships
- Activity type distribution

## Detailed Views
The additional tabs provide more in-depth analysis:
- **Heart Rate Analysis**: Detailed heart rate metrics and zones
- **Activity Distribution**: Activity type breakdowns and trends
- **Duration Analysis**: Workout duration patterns
- **Metric Relationships**: Correlations between different metrics
- **Bike Analysis**: Specific cycling metrics and trends
- **Tennis Analysis**: Tennis activity analysis
- **Source Comparison**: Compare data across different tracking devices
- **Raw Data**: Access to the underlying dataset

## Setup

### Prerequisites
- Python 3.7+
- Fitness tracking data files (CSV format)

### Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/fitness-dashboard.git
cd fitness-dashboard

# Install dependencies
pip install -r requirements.txt
```

### Data Sources

Configure your data sources in `config.py`:

```python
# Update these paths with your actual data files
MIIFIT_PATH = "/path/to/your/DataSnippet.csv"
GARMIN_PATH = "/path/to/your/Activities.csv"
POLAR_F11_PATH = "/path/to/your/PolarF11Data.csv"
CHARGE_HR_PATH = "/path/to/your/ChargeHRDataScrape.csv"
```

## Data Format Requirements

### MiiFit Data
Required columns:
- DATE: Activity date
- TYPE: Activity type
- AVGHR: Average heart rate
- MAX_HR: Maximum heart rate
- CAL: Calories burned
- duration_minutes: Activity duration

### Garmin Data
Required columns:
- Date: Activity date
- Activity Type: Type of activity
- Avg HR: Average heart rate
- Max HR: Maximum heart rate
- Calories: Calories burned
- Total Time: Activity duration (HH:MM:SS format)

### PolarF11 Data
Required columns:
- sport: Activity type
- time: Date of activity
- calories: Calories burned
- Duration: Activity duration (HH:MM:SS format)
- average: Average heart rate
- maximum: Maximum heart rate

### ChargeHR Data
Required columns:
- Date: Activity date
- Activity: Type of activity
- Cals: Calories burned
- Duration: Activity duration (HH:MM:SS format)

## Running the Dashboard

```bash
streamlit run app.py
```

The dashboard will be available at http://localhost:8501

## Project Structure

- `app.py`: Main Streamlit application
- `data_processor.py`: Data loading and processing
- `utils.py`: Utility functions
- `config.py`: Configuration settings
- `visualization.py`: Visualization functions

## Customization

The dashboard can be customized through the `config.py` file:
- Data file paths
- Column mappings
- Activity type mappings
- Plot dimensions
- Color schemes

## Troubleshooting

If you encounter issues with data loading:
- Check that your data files exist at the specified paths
- Verify that CSV files have the expected column names
- Try using the "Generate Sample Data" option in the dashboard
- Check the dashboard.log file for detailed error information

## License

This project is licensed under the MIT License - see the LICENSE file for details.

-
