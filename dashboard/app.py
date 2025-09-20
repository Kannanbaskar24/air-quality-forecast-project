# =================================================================
# app.py - AirAware Dashboard with Logging & Monitoring (Full)
# =================================================================

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import os, sys, traceback
import json
from pathlib import Path

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.logger import get_logger
from src.train import retrain_all_models
from src.db import SessionLocal, Prediction, Alert

logger = get_logger()
st.set_page_config(page_title="AirAware - Air Quality Forecast", layout="wide")

# ===================== Paths =====================
BASE_DIR = Path(__file__).resolve().parents[0]
DATA_DIR = BASE_DIR / ".." / "data" / "processed"

historical_csv = DATA_DIR / "air_quality_cleaned.csv"
MODELS_DIR = BASE_DIR.parent / "models"  # points to the correct folder

# ===================== Model Card Loader =====================
def load_model_card(pollutant):
    """
    Loads the JSON model card for the given pollutant.
    Tries both formats: with dot and without dot in the filename.
    """
    # Try filename without dot (standardized)
    safe_pollutant = pollutant.replace('.', '')
    path_no_dot = MODELS_DIR / f"model_card_{safe_pollutant}.json"
    if path_no_dot.exists():
        with open(path_no_dot) as f:
            return json.load(f)
    
    # Try filename with dot
    path_with_dot = MODELS_DIR / f"model_card_{pollutant}.json"
    if path_with_dot.exists():
        with open(path_with_dot) as f:
            return json.load(f)
    
    # If neither exists, return None
    return None



# ===================== Pollutants & Mappings =====================
pollutants = ["PM2.5","PM10","NO2","O3","CO","SO2"]
category_mapping = {"Good":1,"Moderate":2,"Unhealthy for Sensitive Groups":3,
                    "Unhealthy":4,"Very Unhealthy":5,"Hazardous":6}
reverse_mapping = {v:k for k,v in category_mapping.items()}

# ===================== AQI Categorization Functions =====================
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

# ===================== Load Historical Data =====================
@st.cache_data
def load_and_process_historical_data(path):
    try:
        df = pd.read_csv(path, index_col=0, parse_dates=True)
        aqi_categories_hist = {}
        for p in pollutants:
            func = globals()[f"categorize_{p.lower().replace('.','')}"]
            aqi_categories_hist[p] = df[p].apply(func) if p in df.columns else pd.Series(["Unknown"]*len(df), index=df.index)
        aqi_numeric_hist = pd.DataFrame({p: aqi_categories_hist[p].map(category_mapping) for p in pollutants})
        df["Overall_AQI_Level"] = aqi_numeric_hist.max(axis=1)
        df["Overall_AQI_Category"] = df["Overall_AQI_Level"].map(reverse_mapping)
        for p in pollutants:
            df[f"{p}_AQI_Level"] = aqi_numeric_hist[p].fillna(0)
        logger.info(f"Loaded historical data: {path} ({len(df)} rows)")
        return df
    except Exception as e:
        logger.error(f"Error loading historical CSV: {e}")
        return pd.DataFrame()

historical_df = load_and_process_historical_data(historical_csv)

# ===================== Fetch Forecasts & Alerts =====================
def get_forecasts_from_db():
    session = SessionLocal()
    try:
        df = pd.read_sql(session.query(Prediction).statement, session.bind, parse_dates=['date'])
        if not df.empty:
            df.set_index('date', inplace=True)
            df = df.pivot_table(values='value', index=df.index, columns='pollutant')
        logger.info(f"Fetched {len(df)} forecast rows from DB")
    except Exception as e:
        logger.error(f"Error fetching forecasts from DB: {e}")
        df = pd.DataFrame()
    finally:
        session.close()
    if df.empty:
        try:
            df = pd.read_csv(DATA_DIR / "forecast_7days_full.csv", index_col=0, parse_dates=True)
            logger.info("Fetched forecast from CSV fallback")
        except FileNotFoundError:
            logger.warning("No forecast data available")
            df = pd.DataFrame()
    return df

forecast_df = get_forecasts_from_db()

def get_alerts_from_db():
    session = SessionLocal()
    try:
        df = pd.read_sql(session.query(Alert).statement, session.bind, parse_dates=['date'])
        if not df.empty:
            df.set_index('date', inplace=True)
        logger.info(f"Fetched {len(df)} alerts from DB")
    except Exception as e:
        logger.error(f"Error fetching alerts from DB: {e}")
        df = pd.DataFrame()
    finally:
        session.close()
    if df.empty:
        try:
            df = pd.read_csv(DATA_DIR / "high_risk_alerts.csv", index_col=0, parse_dates=True)
            logger.info("Fetched alerts from CSV fallback")
        except FileNotFoundError:
            logger.warning("No alerts data available")
            df = pd.DataFrame()
    return df

alerts_df = get_alerts_from_db()
alerts_history = get_alerts_from_db()

# ===================== Combine Historical + Forecast =====================
def ensure_tz_naive(df):
    if hasattr(df.index, "tz") and df.index.tz is not None:
        df.index = df.index.tz_convert(None)
    elif hasattr(df.index, "tz") and df.index.tz is None:
        df.index = df.index.tz_localize(None)
    return df

historical_df = ensure_tz_naive(historical_df)
forecast_df = ensure_tz_naive(forecast_df)
combined_df = pd.concat([historical_df, forecast_df], axis=0)
combined_df = combined_df[~combined_df.index.duplicated(keep='first')].sort_index()



# ===================== Sidebar Filters =====================
st.sidebar.header("Filters")
selected_pollutant = st.sidebar.selectbox("Select Pollutant", pollutants)

min_date = combined_df.index.min().date() if not combined_df.empty else pd.Timestamp.today().date()
max_date = combined_df.index.max().date() if not combined_df.empty else pd.Timestamp.today().date()
selected_start = st.sidebar.date_input("Start Date", min_value=min_date, max_value=max_date, value=min_date)
selected_end = st.sidebar.date_input("End Date", min_value=min_date, max_value=max_date, value=max_date)
# --- Filtered DataFrames ---
filtered_df = combined_df[(combined_df.index.date >= selected_start) & (combined_df.index.date <= selected_end)]
filtered_alerts = alerts_df[(alerts_df.index.date >= selected_start) & (alerts_df.index.date <= selected_end)]

with st.sidebar.expander("📊 Model Info", expanded=True):
    card = load_model_card(selected_pollutant)
    if card:
        st.markdown(f"**Model Name:** {card.get('model_name','-')}")
        st.markdown(f"**Pollutant:** {card.get('pollutant','-')}")
        st.markdown(f"**Type:** {card.get('type','-')}")
        st.markdown(f"**Version:** {card.get('version','-')}")
        st.markdown(f"**Training Date:** {card.get('training_date','-')}")
        st.markdown(f"**Data Source:** {card.get('data_source','-')}")
        st.markdown(f"**MAE:** {card.get('MAE','-')}  |  **RMSE:** {card.get('RMSE','-')}")
        st.markdown(f"**Intended Use:** {card.get('intended_use','-')}")
        st.markdown(f"**Limitations:** {card.get('limitations','-')}")
    else:
        st.info("No model info available for this pollutant.")


# ===================== Continue with Admin Panel and Tabs =====================
# All tabs (Tab 1 - Tab 7) and plotting code remain unchanged, now using filtered_df safely


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
    uploaded_file = st.sidebar.file_uploader("Upload CSV", type=["csv"])
    if uploaded_file:
        try:
            new_df = pd.read_csv(uploaded_file, index_col=0, parse_dates=True)
            new_df.to_csv(historical_csv, index=True)
            st.sidebar.success(f"Uploaded {len(new_df)} rows successfully.")
            logger.info(f"Admin uploaded CSV with {len(new_df)} rows")
            st.rerun()
        except Exception as e:
            logger.error(f"Error uploading CSV: {e}")

    if st.sidebar.button("🔄 Retrain Models"):
        with st.spinner("Retraining models..."):
            try:
                result = retrain_all_models(historical_csv)
                logger.info("Admin triggered retraining via Streamlit")
                st.success("✅ Retraining finished successfully.")
                st.info(f"Forecast saved: {result['forecast_path']}")
                st.info(f"Alerts saved: {result['alerts_path']}")
                st.info(f"Metrics saved: {result['metrics_path']}")
                st.rerun()
            except Exception as e:
                logger.error(f"Error during retraining via Streamlit: {e}")
                st.error("Retraining failed — see logs.")
                st.text(traceback.format_exc())

# ===================== Tabs =====================
tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs([
    "Forecast Table", "Line Plots", "AQI Heatmap", "Overall AQI Gauge",
    "Latest High-Risk", "Per-Pollutant Gauges", "High-Risk Alerts History"
])

# --------------------- Tab 1: Forecast Table ---------------------
with tab1:
    st.subheader("📅 Forecast Table with Alerts")
    if not filtered_df.empty:
        display_df = filtered_df[[selected_pollutant, "Overall_AQI_Category", "Overall_AQI_Level"]].copy()
        display_df["Overall_AQI_Level"] = pd.to_numeric(display_df["Overall_AQI_Level"], errors='coerce').fillna(0)

        def color_aqi(val):
            if val >= 5: return 'background-color: #99004c; color: white'
            elif val == 4: return 'background-color: #ff0000; color: white'
            elif val == 3: return 'background-color: #ff7e00; color: white'
            elif val == 2: return 'background-color: #ffff00; color: black'
            elif val == 1: return 'background-color: #00e400; color: black'
            else: return ''

        st.dataframe(display_df.style.applymap(color_aqi, subset=["Overall_AQI_Level"]))
    else:
        st.warning("⚠️ No data available for forecast table.")

# --------------------- Tab 2: Line Plots / Trends ---------------------
# --------------------- Tab 2: Line Plots / Trends (Daily, Weekly, Monthly) ---------------------
with tab2:
    st.subheader(f"{selected_pollutant} Trends")

    if not filtered_df.empty:
        # --- Daily Trend ---
        fig_daily, ax_daily = plt.subplots(figsize=(12,5))
        ax_daily.plot(filtered_df.index, filtered_df[selected_pollutant].fillna(method="ffill"),
                      marker='o', color='blue', label=f"{selected_pollutant} (Daily)")
        ax_daily.set_ylabel(f"{selected_pollutant} Concentration")
        ax_daily.set_xlabel("Date")
        ax_daily.set_title(f"{selected_pollutant} Daily Trend")
        ax_daily.grid(True)
        st.pyplot(fig_daily, clear_figure=True)
        logger.info(f"Displayed daily trend for {selected_pollutant}")

        # --- Weekly Trend ---
        weekly = filtered_df[selected_pollutant].resample('W').mean()
        fig_weekly, ax_weekly = plt.subplots(figsize=(12,5))
        ax_weekly.plot(weekly.index, weekly.values, marker='s', color='green', label=f"{selected_pollutant} (Weekly Avg)")
        ax_weekly.set_ylabel(f"{selected_pollutant} Concentration")
        ax_weekly.set_xlabel("Week")
        ax_weekly.set_title(f"{selected_pollutant} Weekly Average Trend")
        ax_weekly.grid(True)
        st.pyplot(fig_weekly, clear_figure=True)
        logger.info(f"Displayed weekly trend for {selected_pollutant}")

        # --- Monthly Trend ---
        monthly = filtered_df[selected_pollutant].resample('M').mean()
        fig_monthly, ax_monthly = plt.subplots(figsize=(12,5))
        ax_monthly.plot(monthly.index, monthly.values, marker='D', color='orange', label=f"{selected_pollutant} (Monthly Avg)")
        ax_monthly.set_ylabel(f"{selected_pollutant} Concentration")
        ax_monthly.set_xlabel("Month")
        ax_monthly.set_title(f"{selected_pollutant} Monthly Average Trend")
        ax_monthly.grid(True)
        st.pyplot(fig_monthly, clear_figure=True)
        logger.info(f"Displayed monthly trend for {selected_pollutant}")

        # --- Overall AQI Trend (Daily) ---
        st.subheader("Overall AQI Trend")
        fig_aqi, ax_aqi = plt.subplots(figsize=(12,5))
        ax_aqi.plot(filtered_df.index, filtered_df["Overall_AQI_Level"].fillna(0),
                    marker='x', color='red', label='Overall AQI')
        ax_aqi.set_ylabel("Overall AQI Level")
        ax_aqi.set_xlabel("Date")
        ax_aqi.set_title("Overall AQI Daily Trend")
        ax_aqi.grid(True)
        st.pyplot(fig_aqi, clear_figure=True)
        logger.info("Displayed overall AQI daily trend")
    else:
        st.warning("⚠️ No data available for trends.")
        logger.warning("Filtered DataFrame empty — cannot display trends")


# --------------------- Tab 3: AQI Heatmap ---------------------
with tab3:
    st.subheader("AQI Levels Heatmap")
    if not filtered_df.empty:
        aqi_numeric_plot = filtered_df[[f"{p}_AQI_Level" for p in pollutants]].apply(pd.to_numeric, errors='coerce').fillna(0)
        fig3, ax3 = plt.subplots(figsize=(12,5))
        sns.heatmap(aqi_numeric_plot.T, annot=True, cmap="Reds", cbar_kws={'label':'AQI Level'}, ax=ax3)
        ax3.set_xlabel("Date")
        ax3.set_ylabel("Pollutant")
        st.pyplot(fig3, clear_figure=True)
    else:
        st.warning("⚠️ No data for heatmap.")

# --------------------- Tab 4: Overall AQI Gauge ---------------------
with tab4:
    st.subheader("🌡️ Overall AQI Gauge")
    if not filtered_df.empty:
        latest_value = filtered_df["Overall_AQI_Level"].iloc[-1]
        latest_level = int(latest_value) if pd.notna(latest_value) else 0
        latest_category = filtered_df["Overall_AQI_Category"].iloc[-1]
        colors = ["#d3d3d3","#00e400","#ffff00","#ff7e00","#ff0000","#99004c","#7e0023"]
        fig = go.Figure(go.Indicator(
            mode="gauge+number+delta",
            value=latest_level,
            title={'text': f"Overall AQI: {latest_category}"},
            gauge={'axis': {'range':[0,6],'tickvals':[0,1,2,3,4,5,6],
                            'ticktext':['Unknown','Good','Moderate','USG','Unhealthy','Very Unhealthy','Hazardous']},
                   'bar': {'color': colors[latest_level]},
                   'steps':[{'range':[i,i+1],'color':c} for i,c in enumerate(colors)]}
        ))
        st.plotly_chart(fig, use_container_width=True)

# --------------------- Tab 5: Latest High-Risk Alerts ---------------------
with tab5:
    st.subheader("Latest High-Risk Alerts")
    st.dataframe(filtered_alerts)
    st.download_button(
        label="Download High-Risk Alerts",
        data=filtered_alerts.to_csv().encode("utf-8"),
        file_name="high_risk_alerts.csv",
        mime="text/csv"
    )

# --------------------- Tab 6: Per-Pollutant Gauges & Trend ---------------------
# --------------------- Tab 6: Per-Pollutant Gauges & Trend (Daily, Weekly, Monthly) ---------------------
with tab6:
    st.subheader("🌡️ Per-Pollutant AQI Gauges (Latest Day)")
    if not filtered_df.empty:
        latest_row = filtered_df.iloc[-1]
        gauge_colors = ["#d3d3d3","#00e400","#ffff00","#ff7e00","#ff0000","#99004c","#7e0023"]
        for pollutant in pollutants:
            latest_val = latest_row.get(f"{pollutant}_AQI_Level",0)
            latest_aqi = int(latest_val) if pd.notna(latest_val) else 0
            color = gauge_colors[latest_aqi]
            fig_gauge = go.Figure(go.Indicator(
                mode="gauge+number",
                value=latest_aqi,
                title={'text': f"{pollutant} AQI"},
                gauge={'axis': {'range':[0,6], 'tickvals':list(range(7)),
                                'ticktext':['Unknown','Good','Moderate','USG','Unhealthy','Very Unhealthy','Hazardous']},
                       'bar': {'color': color},
                       'steps':[{'range':[i,i+1], 'color':c} for i,c in enumerate(gauge_colors)]}
            ))
            st.plotly_chart(fig_gauge, use_container_width=True)
        logger.info("Displayed latest day AQI gauges for all pollutants")

        # --- Daily Per-Pollutant Trend ---
        st.subheader("📈 Daily AQI Trends per Pollutant")
        fig_daily, ax_daily = plt.subplots(figsize=(12,5))
        for pollutant in pollutants:
            ax_daily.plot(combined_df.index, combined_df[f"{pollutant}_AQI_Level"].fillna(0),
                          marker='o', label=pollutant)
        ax_daily.set_ylabel("AQI Level")
        ax_daily.set_xlabel("Date")
        ax_daily.set_title("Per-Pollutant Daily AQI Trends")
        ax_daily.set_yticks(range(0,7))
        ax_daily.set_yticklabels(['Unknown','Good','Moderate','USG','Unhealthy','Very Unhealthy','Hazardous'])
        ax_daily.grid(True)
        ax_daily.legend()
        st.pyplot(fig_daily, clear_figure=True)
        logger.info("Displayed daily AQI trends per pollutant")

        # --- Weekly Per-Pollutant Trend ---
        st.subheader("📈 Weekly Average AQI Trends per Pollutant")
        fig_weekly, ax_weekly = plt.subplots(figsize=(12,5))
        for pollutant in pollutants:
            weekly = combined_df[f"{pollutant}_AQI_Level"].resample('W').mean()
            ax_weekly.plot(weekly.index, weekly.values, marker='s', label=pollutant)
        ax_weekly.set_ylabel("AQI Level")
        ax_weekly.set_xlabel("Week")
        ax_weekly.set_title("Per-Pollutant Weekly AQI Trends")
        ax_weekly.set_yticks(range(0,7))
        ax_weekly.set_yticklabels(['Unknown','Good','Moderate','USG','Unhealthy','Very Unhealthy','Hazardous'])
        ax_weekly.grid(True)
        ax_weekly.legend()
        st.pyplot(fig_weekly, clear_figure=True)
        logger.info("Displayed weekly AQI trends per pollutant")

        # --- Monthly Per-Pollutant Trend ---
        st.subheader("📈 Monthly Average AQI Trends per Pollutant")
        fig_monthly, ax_monthly = plt.subplots(figsize=(12,5))
        for pollutant in pollutants:
            monthly = combined_df[f"{pollutant}_AQI_Level"].resample('M').mean()
            ax_monthly.plot(monthly.index, monthly.values, marker='D', label=pollutant)
        ax_monthly.set_ylabel("AQI Level")
        ax_monthly.set_xlabel("Month")
        ax_monthly.set_title("Per-Pollutant Monthly AQI Trends")
        ax_monthly.set_yticks(range(0,7))
        ax_monthly.set_yticklabels(['Unknown','Good','Moderate','USG','Unhealthy','Very Unhealthy','Hazardous'])
        ax_monthly.grid(True)
        ax_monthly.legend()
        st.pyplot(fig_monthly, clear_figure=True)
        logger.info("Displayed monthly AQI trends per pollutant")
    else:
        st.warning("⚠️ No data available for per-pollutant trends")
        logger.warning("Filtered DataFrame empty — cannot display per-pollutant trends")


# --------------------- Tab 7: High-Risk Alerts History ---------------------
with tab7:
    st.subheader("📜 Full High-Risk Alerts History")
    if not alerts_history.empty:
        min_date_alert = alerts_history.index.min().date()
        max_date_alert = alerts_history.index.max().date()
        col1, col2 = st.columns(2)
        start_date_alert = col1.date_input("Start Date", min_value=min_date_alert, max_value=max_date_alert, value=min_date_alert, key="tab7_alerts_start")
        end_date_alert = col2.date_input("End Date", min_value=min_date_alert, max_value=max_date_alert, value=max_date_alert, key="tab7_alerts_end")
        filtered_alerts_range = alerts_history[(alerts_history.index.date >= start_date_alert) & (alerts_history.index.date <= end_date_alert)]
        if not filtered_alerts_range.empty:
            st.dataframe(filtered_alerts_range)
            st.download_button(
                label="Download High-Risk Alerts History",
                data=filtered_alerts_range.to_csv().encode("utf-8"),
                file_name="high_risk_alerts_history.csv",
                mime="text/csv"
            )
        else:
            st.info("No alerts found in the selected date range.")
    else:
        st.info("No high-risk alerts available.")

# --------------------- Sidebar Log Viewer ---------------------
tab_logs = st.sidebar.expander("📜 Recent Logs")
try:
    log_file = os.path.join(BASE_DIR, "..", "logs", "airaware.log")
    if os.path.exists(log_file):
        with open(log_file) as f:
            logs = f.readlines()[-50:]
        tab_logs.text("".join(logs))
    else:
        tab_logs.text("No logs available yet.")
except Exception as e:
    tab_logs.text(f"Error reading logs: {e}")
    logger.error(f"Error reading log file: {e}")
