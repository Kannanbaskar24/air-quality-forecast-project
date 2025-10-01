# AirAware/dashboard/app.py

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import os
import sys

# -----------------------------
# Ensure project root is in path
# -----------------------------
sys.path.append(os.path.abspath(".."))
from src.inference import forecast_future  # Module 2 retrain function

# -----------------------------
# Dashboard Title
# -----------------------------
st.set_page_config(page_title="AirAware Dashboard", layout="wide")
st.title("🌬️ AirAware: Air Quality Forecast Dashboard")

# -----------------------------
# Load Forecast & Alerts
# -----------------------------
forecast_file = "../data/processed/forecast_7days_full.csv"
alerts_file = "../data/processed/high_risk_alerts.csv"

forecast = pd.read_csv(forecast_file, index_col=0, parse_dates=True)
alerts = pd.read_csv(alerts_file, index_col=0, parse_dates=True)

# -----------------------------
# Sidebar - Filters
# -----------------------------
pollutants = ["PM2.5","PM10","NO2","O3","CO","SO2","BC","NO","NOX","Overall_AQI_Level"]
selected_pollutant = st.sidebar.selectbox("Select pollutant / AQI", pollutants)

start_date = st.sidebar.date_input("Start date", forecast.index.min())
end_date = st.sidebar.date_input("End date", forecast.index.max())

filtered = forecast.loc[start_date:end_date]

# -----------------------------
# Line Chart - Selected Pollutant
# -----------------------------
st.subheader(f"📈 {selected_pollutant} Forecast")
fig = px.line(filtered, x=filtered.index, y=selected_pollutant,
              labels={'x':'Date', selected_pollutant:'Concentration'},
              title=f"{selected_pollutant} Forecast")
st.plotly_chart(fig, use_container_width=True)

# -----------------------------
# Multi-Pollutant Line Chart
# -----------------------------
st.subheader("📊 Multi-Pollutant Forecast")
multi_pollutants = [p for p in pollutants if p not in ["Overall_AQI_Level","Overall_AQI_Category"]]
fig_multi = px.line(filtered, x=filtered.index, y=multi_pollutants,
                    labels={'x':'Date', 'value':'Concentration', 'variable':'Pollutant'},
                    title="Forecast for All Pollutants")
st.plotly_chart(fig_multi, use_container_width=True)

# -----------------------------
# Overall AQI Gauge (Latest Forecast Day)
# -----------------------------
latest_aqi = filtered["Overall_AQI_Level"].iloc[-1]
latest_category = filtered["Overall_AQI_Category"].iloc[-1]

st.subheader("🟢 Overall AQI (Latest Forecast Day)")
fig_gauge = go.Figure(go.Indicator(
    mode = "gauge+number+delta",
    value = latest_aqi,
    title = {'text': f"Overall AQI: {latest_category}"},
    gauge = {'axis': {'range': [0,6]},
             'bar': {'color': "red" if latest_aqi>=4 else "green"},
             'steps' : [
                 {'range': [0,1], 'color':'green'},
                 {'range': [1,2], 'color':'lime'},
                 {'range': [2,3], 'color':'yellow'},
                 {'range': [3,4], 'color':'orange'},
                 {'range': [4,5], 'color':'red'},
                 {'range': [5,6], 'color':'darkred'}]}))
st.plotly_chart(fig_gauge, use_container_width=True)

# -----------------------------
# High-Risk Alerts Table
# -----------------------------
st.subheader("⚠️ High-Risk AQI Days")
def highlight_alerts(row):
    if row["Overall_AQI_Level"] >= 4:
        return ["background-color: red; color: white"]*len(row)
    else:
        return [""]*len(row)

st.dataframe(alerts.style.apply(highlight_alerts, axis=1))

# -----------------------------
# Admin Panel - Upload New Data
# -----------------------------
st.sidebar.subheader("Admin Panel")

uploaded_file = st.sidebar.file_uploader("Upload new air quality CSV", type=["csv"])
if uploaded_file:
    st.sidebar.success("File uploaded successfully!")
    df_new = pd.read_csv(uploaded_file, index_col=0, parse_dates=True)
    st.write("Preview of uploaded CSV:")
    st.dataframe(df_new.head())

    # Button to retrain forecast using uploaded data
    if st.sidebar.button("🔄 Retrain Forecast"):
        st.sidebar.info("Retraining forecasts for next 7 days...")
        new_forecast_dict = {}
        forecast_days = 7
        for pollutant in multi_pollutants:
            series = df_new[pollutant].dropna().asfreq("D")
            new_forecast_dict[pollutant] = forecast_future(pollutant, series, days=forecast_days)
        new_forecast_df = pd.DataFrame(new_forecast_dict)
        new_forecast_df.index.name = "Date"
        
        # Recompute AQI
        from copy import deepcopy
        def categorize_pm25(val):
            if val <= 12: return "Good"
            elif val <= 35.4: return "Moderate"
            elif val <= 55.4: return "Unhealthy for Sensitive Groups"
            elif val <= 150.4: return "Unhealthy"
            elif val <= 250.4: return "Very Unhealthy"
            else: return "Hazardous"
        def categorize_pm10(val):
            if val <= 54: return "Good"
            elif val <= 154: return "Moderate"
            elif val <= 254: return "Unhealthy for Sensitive Groups"
            elif val <= 354: return "Unhealthy"
            elif val <= 424: return "Very Unhealthy"
            else: return "Hazardous"
        def categorize_no2(val):
            if val <= 53: return "Good"
            elif val <= 100: return "Moderate"
            elif val <= 360: return "Unhealthy for Sensitive Groups"
            elif val <= 649: return "Unhealthy"
            elif val <= 1249: return "Very Unhealthy"
            else: return "Hazardous"
        def categorize_o3(val):
            if val <= 54: return "Good"
            elif val <= 70: return "Moderate"
            elif val <= 85: return "Unhealthy for Sensitive Groups"
            elif val <= 105: return "Unhealthy"
            elif val <= 200: return "Very Unhealthy"
            else: return "Hazardous"
        def categorize_co(val):
            if val <= 4.4: return "Good"
            elif val <= 9.4: return "Moderate"
            elif val <= 12.4: return "Unhealthy for Sensitive Groups"
            elif val <= 15.4: return "Unhealthy"
            elif val <= 30.4: return "Very Unhealthy"
            else: return "Hazardous"
        def categorize_so2(val):
            if val <= 35: return "Good"
            elif val <= 75: return "Moderate"
            elif val <= 185: return "Unhealthy for Sensitive Groups"
            elif val <= 304: return "Unhealthy"
            elif val <= 604: return "Very Unhealthy"
            else: return "Hazardous"
        
        aqi_categories = {}
        for pollutant in multi_pollutants:
            if pollutant == "PM2.5":
                aqi_categories[pollutant] = new_forecast_df[pollutant].apply(categorize_pm25)
            elif pollutant == "PM10":
                aqi_categories[pollutant] = new_forecast_df[pollutant].apply(categorize_pm10)
            elif pollutant == "NO2":
                aqi_categories[pollutant] = new_forecast_df[pollutant].apply(categorize_no2)
            elif pollutant == "O3":
                aqi_categories[pollutant] = new_forecast_df[pollutant].apply(categorize_o3)
            elif pollutant == "CO":
                aqi_categories[pollutant] = new_forecast_df[pollutant].apply(categorize_co)
            elif pollutant == "SO2":
                aqi_categories[pollutant] = new_forecast_df[pollutant].apply(categorize_so2)
            else:
                aqi_categories[pollutant] = pd.Series(["Unknown"]*len(new_forecast_df), index=new_forecast_df.index)

        category_mapping = {
            "Good": 1, "Moderate": 2, "Unhealthy for Sensitive Groups": 3,
            "Unhealthy": 4, "Very Unhealthy": 5, "Hazardous": 6,
            "Unknown": 0
        }
        reverse_mapping = {v:k for k,v in category_mapping.items()}

        aqi_numeric = pd.DataFrame({p: aqi_categories[p].map(category_mapping) for p in aqi_categories})
        new_forecast_df["Overall_AQI_Level"] = aqi_numeric.max(axis=1)
        new_forecast_df["Overall_AQI_Category"] = new_forecast_df["Overall_AQI_Level"].map(reverse_mapping)

        # Save updated forecasts
        new_forecast_df.to_csv("../data/processed/forecast_7days_full.csv")
        st.success("✅ Forecasts retrained and saved successfully!")
