# ===============================================================
# app.py - AirAware Dashboard (with Admin Retrain Button)
# ===============================================================

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import os, sys, traceback

st.set_page_config(page_title="AirAware - Air Quality Forecast", layout="wide")

# ===================== Paths =====================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "..", "data", "processed")
forecast_csv = os.path.join(DATA_DIR, "forecast_7days_full.csv")
alerts_csv = os.path.join(DATA_DIR, "high_risk_alerts.csv")
historical_csv = os.path.join(DATA_DIR, "air_quality_cleaned.csv")

# Make src visible
sys.path.append(os.path.join(BASE_DIR, ".."))
from src.train import retrain_all_models

# ===================== AQI Category Functions =====================
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

category_mapping = {
    "Good": 1, "Moderate": 2, "Unhealthy for Sensitive Groups": 3,
    "Unhealthy": 4, "Very Unhealthy": 5, "Hazardous": 6
}
reverse_mapping = {v: k for k, v in category_mapping.items()}
pollutants = ["PM2.5","PM10","NO2","O3","CO","SO2"]

# ===================== Data Loading Function =====================
@st.cache_data
def load_and_process_data(historical_path, forecast_path):
    hist_df = pd.read_csv(historical_path, index_col=0, parse_dates=True)
    try:
        fc_df = pd.read_csv(forecast_path, index_col=0, parse_dates=True)
    except FileNotFoundError:
        fc_df = pd.DataFrame()

    # Compute AQI for historical data
    aqi_categories_hist = {}
    for p in pollutants:
        if p in hist_df.columns:
            func = globals()[f"categorize_{p.lower().replace('.','')}"]
            aqi_categories_hist[p] = hist_df[p].apply(func)
        else:
            aqi_categories_hist[p] = pd.Series(["Unknown"] * len(hist_df), index=hist_df.index)

    aqi_numeric_hist = pd.DataFrame({p: aqi_categories_hist[p].map(category_mapping) for p in pollutants})
    hist_df["Overall_AQI_Level"] = aqi_numeric_hist.max(axis=1)
    hist_df["Overall_AQI_Category"] = hist_df["Overall_AQI_Level"].map(reverse_mapping)
    for p in pollutants:
        hist_df[f"{p}_AQI_Level"] = aqi_numeric_hist[p].fillna(0)

    if not fc_df.empty:
        combined = pd.concat([hist_df, fc_df], axis=0)
        combined = combined[~combined.index.duplicated(keep="first")]
    else:
        combined = hist_df

    return hist_df, fc_df, combined

# ===================== Admin Panel =====================
st.sidebar.header("Admin Panel Login")
admin_password = st.sidebar.text_input("Enter Admin Password", type="password")
is_admin = False
if admin_password == os.environ.get("AIRAWARE_ADMIN_PASSWORD", "changeme"):
    is_admin = True
    st.sidebar.success("Admin Access Granted")
elif admin_password:
    st.sidebar.error("Incorrect Password")

if is_admin:
    st.sidebar.subheader("Upload Historical Data CSV")
    uploaded_file = st.sidebar.file_uploader("Upload CSV", type=["csv"])
    if uploaded_file:
        new_df = pd.read_csv(uploaded_file, index_col=0, parse_dates=True)
        new_df.to_csv(historical_csv, index=True)
        st.sidebar.success(f"Uploaded {len(new_df)} rows successfully.")
        st.rerun()   # ✅ updated

    # Retrain button
    st.sidebar.subheader("Model Operations")
    if st.sidebar.button("🔄 Retrain Models"):
        with st.spinner("Retraining models and generating forecasts..."):
            try:
                result = retrain_all_models(historical_csv)
                st.success("✅ Retraining finished successfully.")
                st.info(f"Forecast saved: {result['forecast_path']}")
                st.info(f"Alerts saved: {result['alerts_path']}")
                st.info(f"Metrics saved: {result['metrics_path']}")
                st.rerun()   # ✅ updated
            except Exception as e:
                st.error("Retraining failed — see logs.")
                st.text(traceback.format_exc())

# ===================== Load Data =====================
historical_df, forecast_df, combined_df = load_and_process_data(historical_csv, forecast_csv)

# ===================== Sidebar Filters =====================
st.sidebar.header("Filters")
selected_pollutant = st.sidebar.selectbox("Select Pollutant", pollutants)
min_date = combined_df.index.min().date()
max_date = combined_df.index.max().date()
selected_start = st.sidebar.date_input("Start Date", min_value=min_date, max_value=max_date, value=min_date)
selected_end = st.sidebar.date_input("End Date", min_value=min_date, max_value=max_date, value=max_date)
if selected_start > selected_end:
    st.sidebar.error("Error: Start date must be before End date.")

filtered_df = combined_df[(combined_df.index.date >= selected_start) & (combined_df.index.date <= selected_end)]

# ===================== Summary Header =====================
st.markdown("## 🌟 Air Quality Summary")
if not filtered_df.empty:
    latest = filtered_df.iloc[-1]
    high_risk_count = (filtered_df["Overall_AQI_Level"] >= 4).sum()
    critical_pollutant = filtered_df[pollutants].iloc[-1].idxmax()
    st.markdown(f"**Current Overall AQI:** {latest['Overall_AQI_Category']}")
    st.markdown(f"**High-Risk Days in Range:** {high_risk_count}")
    st.markdown(f"**Most Critical Pollutant Today:** {critical_pollutant}")

# ===================== Tabs =====================
tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs([
    "Forecast Table", "Line Plots", "AQI Heatmap", "Overall AQI Gauge",
    "Latest High-Risk", "Per-Pollutant Gauges", "📜 High-Risk Alerts History"
])

# --------------------- Tab 1: Forecast Table ---------------------
with tab1:
    st.dataframe(filtered_df[[selected_pollutant, "Overall_AQI_Category"]])

# --------------------- Tab 2: Line Plots + Seasonal Trends ---------------------
with tab2:
    st.subheader(f"{selected_pollutant} Trend")
    fig, ax = plt.subplots(figsize=(10,4))
    filtered_df[selected_pollutant].plot(marker="o", ax=ax)
    ax.set_ylabel("Concentration")
    ax.set_xlabel("Date")
    ax.set_title(f"{selected_pollutant} Trend")
    plt.grid(True)
    st.pyplot(fig)

    st.subheader("Overall AQI Trend")
    fig2, ax2 = plt.subplots(figsize=(10,4))
    filtered_df["Overall_AQI_Level"].plot(kind="line", marker="o", color="red", ax=ax2)
    ax2.set_ylabel("AQI Level")
    ax2.set_xlabel("Date")
    ax2.set_title("Overall AQI Trend")
    plt.grid(True)
    st.pyplot(fig2)

    # Monthly and Day-of-Week
    st.subheader("📊 Monthly Average AQI")
    filtered_df["month"] = filtered_df.index.month
    monthly_avg = filtered_df.groupby("month")[f"{selected_pollutant}_AQI_Level"].mean()
    fig_month, ax_month = plt.subplots(figsize=(10,4))
    ax_month.bar(monthly_avg.index, monthly_avg, color="orange")
    ax_month.set_xticks(range(1,13))
    ax_month.set_yticks(range(0,7))
    ax_month.set_yticklabels(['Unknown','Good','Moderate','USG','Unhealthy','Very Unhealthy','Hazardous'])
    ax_month.set_xlabel("Month")
    ax_month.set_ylabel("Average AQI Level")
    ax_month.set_title(f"Monthly Average AQI for {selected_pollutant}")
    st.pyplot(fig_month)

    st.subheader("📊 Average AQI by Day of Week")
    filtered_df["dayofweek"] = filtered_df.index.dayofweek
    dow_avg = filtered_df.groupby("dayofweek")[f"{selected_pollutant}_AQI_Level"].mean()
    fig_dow, ax_dow = plt.subplots(figsize=(10,4))
    ax_dow.bar(dow_avg.index, dow_avg, color="skyblue")
    ax_dow.set_xticks(range(0,7))
    ax_dow.set_yticks(range(0,7))
    ax_dow.set_yticklabels(['Unknown','Good','Moderate','USG','Unhealthy','Very Unhealthy','Hazardous'])
    ax_dow.set_xlabel("Day of Week (0=Monday)")
    ax_dow.set_ylabel("Average AQI Level")
    ax_dow.set_title(f"Average AQI by Day of Week for {selected_pollutant}")
    st.pyplot(fig_dow)

# --------------------- Tab 3: AQI Heatmap ---------------------
with tab3:
    st.subheader("AQI Levels Heatmap")
    aqi_numeric_plot = filtered_df[[f"{p}_AQI_Level" for p in pollutants]].fillna(0)
    fig3, ax3 = plt.subplots(figsize=(12,5))
    sns.heatmap(aqi_numeric_plot.T, annot=True, cmap="Reds", cbar_kws={'label':'AQI Level'}, ax=ax3)
    ax3.set_xlabel("Date")
    ax3.set_ylabel("Pollutant")
    ax3.set_title("AQI Levels (Historical + Forecast)")
    st.pyplot(fig3)

# --------------------- Tab 4: Overall AQI Gauge ---------------------
with tab4:
    st.subheader("🌡️ Overall AQI Gauge (Latest Day)")
    if not filtered_df.empty:
        latest_aqi_level = int(filtered_df["Overall_AQI_Level"].iloc[-1])
        latest_aqi_category = filtered_df["Overall_AQI_Category"].iloc[-1]
        gauge_colors = ["#d3d3d3","#00e400","#ffff00","#ff7e00","#ff0000","#99004c","#7e0023"]
        color = gauge_colors[latest_aqi_level] if latest_aqi_level >= 0 else "#d3d3d3"
        fig_gauge = go.Figure(go.Indicator(
            mode="gauge+number+delta",
            value=latest_aqi_level,
            title={'text': f"Overall AQI: {latest_aqi_category}"},
            gauge={
                'axis': {'range':[0,6],'tickvals':[0,1,2,3,4,5,6],
                         'ticktext':['Unknown','Good','Moderate','USG','Unhealthy','Very Unhealthy','Hazardous']},
                'bar': {'color': color},
                'steps':[{'range':[i,i+1],'color':c} for i,c in enumerate(gauge_colors)]
            }
        ))
        st.plotly_chart(fig_gauge, use_container_width=True)

# --------------------- Tab 5: Latest High-Risk ---------------------
with tab5:
    st.subheader("⚠️ Latest High-Risk Day")
    high_risk_df = filtered_df[filtered_df["Overall_AQI_Level"] >= 4].copy()
    if not high_risk_df.empty:
        latest_high_risk_date = high_risk_df.index.max()
        latest_aqi_category = high_risk_df.loc[latest_high_risk_date, "Overall_AQI_Category"]
        st.warning(f"Latest high-risk air quality predicted on **{latest_high_risk_date.date()}** with Overall AQI: **{latest_aqi_category}**")

        high_risk_df["Major_Pollutants"] = high_risk_df[[f"{p}_AQI_Level" for p in pollutants]].apply(
            lambda row: ', '.join([p for p, val in zip(pollutants, row) if val >= 4]), axis=1
        )
        top5 = high_risk_df.sort_index(ascending=False).head(5)
        st.subheader("Top 5 Recent High-Risk Days")
        st.dataframe(top5[["Overall_AQI_Category","Major_Pollutants"]])
    else:
        st.success("No high-risk days in the selected range.")

# --------------------- Tab 6: Per-Pollutant Gauges + Trend + Download ---------------------
with tab6:
    st.subheader("🌡️ Per-Pollutant AQI Gauges (Latest Day)")
    if not filtered_df.empty:
        latest_row = filtered_df.iloc[-1]
        gauge_colors = ["#d3d3d3","#00e400","#ffff00","#ff7e00","#ff0000","#99004c","#7e0023"]
        for pollutant in pollutants:
            aqi_level_val = latest_row.get(f"{pollutant}_AQI_Level",0)
            aqi_level = int(aqi_level_val) if pd.notna(aqi_level_val) else 0
            color = gauge_colors[aqi_level]
            fig = go.Figure(go.Indicator(
                mode="gauge+number",
                value=aqi_level,
                title={'text': f"{pollutant} AQI"},
                gauge={
                    'axis': {'range':[0,6],'tickvals':[0,1,2,3,4,5,6],
                             'ticktext':['Unknown','Good','Moderate','USG','Unhealthy','Very Unhealthy','Hazardous']},
                    'bar': {'color': color},
                    'steps':[{'range':[i,i+1],'color':c} for i,c in enumerate(gauge_colors)]
                }
            ))
            st.plotly_chart(fig, use_container_width=True)

    # --------------------- Per-Pollutant AQI Trend Line ---------------------
    st.subheader("📈 Per-Pollutant AQI Trends")
    fig_trend, ax = plt.subplots(figsize=(12,5))
    for pollutant in pollutants:
        aqi_series = filtered_df[f"{pollutant}_AQI_Level"].fillna(0)
        ax.plot(filtered_df.index, aqi_series, marker='o', label=pollutant)
    ax.set_ylabel("AQI Level")
    ax.set_xlabel("Date")
    ax.set_title("Per-Pollutant AQI Trends (Historical + Forecast)")
    ax.set_yticks(range(0,7))
    ax.set_yticklabels(['Unknown','Good','Moderate','USG','Unhealthy','Very Unhealthy','Hazardous'])
    plt.grid(True)
    plt.legend()
    st.pyplot(fig_trend)

    # --------------------- Download Per-Pollutant AQI CSV ---------------------
    st.subheader("Download Per-Pollutant AQI Numeric CSV")
    per_pollutant_aqi_df = filtered_df[[f"{p}_AQI_Level" for p in pollutants] + ["Overall_AQI_Level","Overall_AQI_Category"]]
    per_pollutant_aqi_df.index.name = "Date"
    st.download_button(
        label="Download Per-Pollutant AQI CSV",
        data=per_pollutant_aqi_df.to_csv().encode('utf-8'),
        file_name="per_pollutant_aqi.csv",
        mime="text/csv"
    )

# --------------------- General Download Buttons ---------------------
st.subheader("Download Filtered Data CSV")
st.download_button(
    label="Download Filtered Data CSV",
    data=filtered_df.to_csv().encode('utf-8'),
    file_name="air_quality_filtered.csv",
    mime="text/csv"
)

try:
    alerts_df = pd.read_csv(alerts_csv, index_col=0, parse_dates=True)
    st.download_button(
        label="Download High-Risk Alerts CSV",
        data=alerts_df.to_csv().encode('utf-8'),
        file_name="high_risk_alerts.csv",
        mime="text/csv"
    )
except FileNotFoundError:
    st.warning(f"High-risk alerts CSV not found at {alerts_csv}.")

# --------------------- Tab 7: High-Risk Alerts History ---------------------
with tab7:
    st.subheader("📜 Full High-Risk Alerts History")

    try:
        alerts_history = pd.read_csv(alerts_csv, index_col=0, parse_dates=True)

        if not alerts_history.empty:
            # Date filter
            min_date = alerts_history.index.min().date()
            max_date = alerts_history.index.max().date()
            col1, col2 = st.columns(2)
            start_date = col1.date_input("Start Date", min_value=min_date, max_value=max_date, value=min_date)
            end_date = col2.date_input("End Date", min_value=min_date, max_value=max_date, value=max_date)

            filtered_alerts = alerts_history[(alerts_history.index.date >= start_date) &
                                             (alerts_history.index.date <= end_date)]

            st.dataframe(filtered_alerts)

            # Download button
            st.download_button(
                label="Download High-Risk Alerts History",
                data=filtered_alerts.to_csv().encode("utf-8"),
                file_name="high_risk_alerts_history.csv",
                mime="text/csv"
            )
        else:
            st.info("No high-risk alerts have been recorded yet.")
    except FileNotFoundError:
        st.warning("⚠️ No high_risk_alerts.csv found. Run retraining first.")
