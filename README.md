# AirAware: Smart Air Quality Prediction System

## Project Overview
Air pollution is a critical concern in urban areas, directly affecting human health, productivity, and environmental quality. Exposure to pollutants such as PM2.5, PM10, NO2, SO2, and O3 can lead to respiratory diseases, cardiovascular problems, and reduced quality of life.

**AirAware** is a data-driven intelligent forecasting system designed to predict air quality using historical pollution data and provide actionable insights for authorities and the public. By leveraging advanced time series models, the system enables proactive interventions, awareness, and better urban planning for healthier environments.

---

## Motivation
Urban air pollution is often unpredictable and varies by region, time of day, and season. Traditional monitoring provides only real-time or historical insights, limiting proactive measures. AirAware addresses this gap by:

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
- Handle missing timestamps, normalize pollutant levels, and remove outliers.
- Feature engineering: day of the week, month, season, temperature, humidity.

### 2. Forecasting Model
- Train multiple models: ARIMA, Prophet, LSTM, XGBoost.
- Evaluate with RMSE and MAE.
- Save the best-performing model for production use.

### 3. Alerting & Trend Analysis
- Convert predicted pollutant levels into AQI categories: Good, Moderate, Unhealthy, etc.
- Highlight high-risk days exceeding safety thresholds.
- Analyze seasonal and regional pollutant trends.

### 4. Web Interface & Admin Panel
- **Streamlit Dashboard:** Select city/station, timeframe, and pollutant.
- **AQI Gauge & Line Plots:** Interactive visualizations of current and forecasted air quality.
- **Alerts:** Warning banners for unsafe air conditions.
- **Admin Upload:** Upload new datasets and trigger retraining for updated predictions.

---

## Workflow
1. **Data Collection:** Historical air quality datasets collected from open sources.
2. **Data Preprocessing:** Clean, normalize, handle missing timestamps, generate features.
3. **Model Training:** Train and compare multiple time series models.
4. **Prediction & Alerting:** Forecast AQI, generate alerts for high-risk days.
5. **Visualization:** Display forecasts and trends via dashboard.
6. **Admin Functionality:** Update datasets and retrain models.

---

## System Architecture
Four main layers:

1. **Data Layer:** Stores historical datasets, preprocessing scripts, and feature-engineered tables.
2. **Model Layer:** Contains forecasting models trained on historical data.
3. **Logic Layer:** Handles AQI calculations, alert generation, and threshold evaluation.
4. **Presentation Layer:** Streamlit dashboard for visualization and alerts.

---

## Installation

1. Clone the repository:
git clone <your-repo-url>
cd AirAware

2. Create and activate a virtual environment:
python -m venv myenv

Windows
myenv\Scripts\activate

Linux/Mac
source myenv/bin/activate

3. Install dependencies:
pip install -r requirements.txt

4. Run the Streamlit dashboard:
streamlit run app.py

5. Run the API:
uvicorn serve_api:app --reload
undefined

## License
This project is licensed under the MIT License - see the LICENSE file for details.

## Future Enhancements
Real-time data integration from air quality sensors.
Mobile-friendly interface with push notifications.
Multi-city forecasting with regional comparison.
Integration with weather data for more accurate predictions.

## Contributors
Kannan Baskar