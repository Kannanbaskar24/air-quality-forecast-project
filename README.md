# AirAware: Smart Air Quality Prediction System

## Project Overview
Air pollution has become a critical concern in urban areas, directly impacting human health, productivity, and environmental quality. Exposure to pollutants such as PM2.5, PM10, NO2, SO2, and O3 can lead to respiratory diseases, cardiovascular problems, and reduced quality of life.  

**AirAware** is a data-driven, intelligent forecasting system designed to predict air quality using historical pollution data and provide actionable insights for authorities and the public. By leveraging advanced time series models, the system enables proactive interventions, awareness, and better urban planning for healthier environments.

---

## Motivation
Urban air pollution is often unpredictable and varies by region, time of day, and season. Traditional monitoring provides only real-time or historical insights, which limits proactive measures. AirAware addresses this gap by:

- Predicting future AQI levels with high accuracy.
- Highlighting high-risk days in advance.
- Supporting data-driven decisions for citizens, city planners, and environmental agencies.

---

## Key Features
- **Time Series Forecasting:** Predict AQI and key pollutant levels (PM2.5, PM10, NO2, etc.) using ARIMA, Prophet, LSTM, and XGBoost models.
- **Interactive Dashboard:** Streamlit-based dashboard for visualizing historical data, forecasts, and trends.
- **Alerts & Notifications:** Automatic identification of days exceeding safe air quality limits.
- **Admin Panel:** Upload new datasets and retrain models for continuous updates.
- **Data Insights:** Seasonal and regional analysis to identify patterns and pollutant contributions.

---

## Modules

### 1. Data Collection & Preprocessing
- Aggregate datasets from sources like CPCB, OpenAQ, or Kaggle.
- Handle missing values, normalize pollutant levels, and remove outliers.
- Engineer additional features (day of the week, month, season, temperature, humidity).

### 2. Forecasting Model
- Train and compare multiple models: ARIMA, Prophet, LSTM, XGBoost.
- Evaluate using metrics like RMSE (Root Mean Squared Error) and MAE (Mean Absolute Error).
- Save the best-performing model for inference in production.

### 3. Alerting & Trend Analysis
- Convert predicted pollutant levels into AQI categories: Good, Moderate, Unhealthy, etc.
- Highlight high-risk days exceeding safety thresholds.
- Provide seasonal, regional, and long-term pollutant trend analysis.

### 4. Web Interface & Admin Panel
- **Streamlit Dashboard:** Select city/station, timeframe, and pollutant to visualize.
- **AQI Gauge & Line Plots:** Interactive visualizations of current and forecasted air quality.
- **Alerts:** Display warning banners for unsafe air conditions.
- **Admin Upload:** Upload new datasets and trigger retraining of models for updated predictions.

---

## Workflow Description
The AirAware system follows this workflow:

1. **Data Collection:** Historical air quality datasets are collected from open sources.
2. **Data Preprocessing:** Clean and normalize data, handle missing timestamps, and generate feature sets.
3. **Model Training:** Train multiple time series models and select the best-performing one based on accuracy metrics.
4. **Prediction & Alerting:** Forecast AQI and pollutants, generate alerts for high-risk days.
5. **Visualization:** Display forecasts and trends through an interactive dashboard.
6. **Admin Functionality:** Update datasets and retrain models to maintain forecasting accuracy.

---

## System Architecture
The system architecture includes four main layers:

1. **Data Layer:** Stores historical air quality datasets, preprocessing scripts, and feature-engineered tables.
2. **Model Layer:** Contains forecasting models (ARIMA, Prophet, LSTM, XGBoost) trained on historical data.
3. **Logic Layer:** Handles AQI calculations, alert generation, and threshold evaluation.
4. **Presentation Layer:** Streamlit dashboard for users to visualize trends, forecasts, and receive alerts.

---

## Installation

1. Clone the repository:
```bash
git clone <your-repo-url>
cd air-quality-forecast

2.Create a virtual environment and install dependencies:
python -m venv myenv
myenv\Scripts\activate
pip install -r requirements.txt

Usage
Run the Streamlit dashboard:
    streamlit run dashboard/app.py
Explore forecasts, AQI trends, and alerts via the interactive dashboard.

Future Enhancements
Real-time data integration from air quality sensors.
Mobile-friendly web interface and push notifications.
Multi-city forecasting with regional comparison.
Integration with weather data for more accurate predictions.

Contributors:
  Kannan Baskar
