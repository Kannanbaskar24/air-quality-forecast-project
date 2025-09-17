# ===============================================================
# aqi.py - AQI Calculation Utilities
# ===============================================================

import pandas as pd

# ----------------- Categorization Functions -----------------
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

# ----------------- Mapping -----------------
category_mapping = {
    "Good": 1, "Moderate": 2, "Unhealthy for Sensitive Groups": 3,
    "Unhealthy": 4, "Very Unhealthy": 5, "Hazardous": 6
}
reverse_mapping = {v: k for k, v in category_mapping.items()}

pollutants = ["PM2.5","PM10","NO2","O3","CO","SO2"]

# ----------------- Compute AQI -----------------
def compute_overall_aqi_from_df(df: pd.DataFrame) -> pd.DataFrame:
    """
    Given a DataFrame with pollutant columns (PM2.5, PM10, NO2, O3, CO, SO2),
    compute per-pollutant AQI levels and overall AQI.
    """
    aqi_categories = {}
    for p in pollutants:
        if p in df.columns:
            func = globals()[f"categorize_{p.lower().replace('.','')}"]
            aqi_categories[p] = df[p].apply(func)
        else:
            aqi_categories[p] = pd.Series(["Unknown"] * len(df), index=df.index)

    aqi_numeric = pd.DataFrame({p: aqi_categories[p].map(category_mapping) for p in pollutants})
    df = df.copy()
    df["Overall_AQI_Level"] = aqi_numeric.max(axis=1)
    df["Overall_AQI_Category"] = df["Overall_AQI_Level"].map(reverse_mapping)
    for p in pollutants:
        df[f"{p}_AQI_Level"] = aqi_numeric[p].fillna(0)

    return df
