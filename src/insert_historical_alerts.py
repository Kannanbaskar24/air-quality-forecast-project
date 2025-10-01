import pandas as pd
import sys, os
sys.path.append(os.path.abspath("."))  # ensure 'src' is discoverable

from src.db import SessionLocal, Alert

# -------------------- Load full historical data --------------------
data = pd.read_csv("data/processed/air_quality_cleaned.csv", index_col=0, parse_dates=True)

pollutants = ["PM2.5","PM10","NO2","O3","CO","SO2"]

# -------------------- AQI Categorization Functions --------------------
def categorize_pm25(val):
    if val <= 12: return 1
    elif val <= 35.4: return 2
    elif val <= 55.4: return 3
    elif val <= 150.4: return 4
    elif val <= 250.4: return 5
    else: return 6

def categorize_pm10(val):
    if val <= 54: return 1
    elif val <= 154: return 2
    elif val <= 254: return 3
    elif val <= 354: return 4
    elif val <= 424: return 5
    else: return 6

def categorize_no2(val):
    if val <= 53: return 1
    elif val <= 100: return 2
    elif val <= 360: return 3
    elif val <= 649: return 4
    elif val <= 1249: return 5
    else: return 6

def categorize_o3(val):
    if val <= 54: return 1
    elif val <= 70: return 2
    elif val <= 85: return 3
    elif val <= 105: return 4
    elif val <= 200: return 5
    else: return 6

def categorize_co(val):
    if val <= 4.4: return 1
    elif val <= 9.4: return 2
    elif val <= 12.4: return 3
    elif val <= 15.4: return 4
    elif val <= 30.4: return 5
    else: return 6

def categorize_so2(val):
    if val <= 35: return 1
    elif val <= 75: return 2
    elif val <= 185: return 3
    elif val <= 304: return 4
    elif val <= 604: return 5
    else: return 6

category_mapping = {1:"Good",2:"Moderate",3:"Unhealthy for Sensitive Groups",
                    4:"Unhealthy",5:"Very Unhealthy",6:"Hazardous"}

# -------------------- Compute per-pollutant AQI Levels --------------------
data["PM2.5_AQI_Level"] = data["PM2.5"].apply(categorize_pm25)
data["PM10_AQI_Level"] = data["PM10"].apply(categorize_pm10)
data["NO2_AQI_Level"] = data["NO2"].apply(categorize_no2)
data["O3_AQI_Level"] = data["O3"].apply(categorize_o3)
data["CO_AQI_Level"] = data["CO"].apply(categorize_co)
data["SO2_AQI_Level"] = data["SO2"].apply(categorize_so2)

# -------------------- Overall AQI Level & Category --------------------
data["Overall_AQI_Level"] = data[[f"{p}_AQI_Level" for p in pollutants]].max(axis=1)
data["Overall_AQI_Category"] = data["Overall_AQI_Level"].map(category_mapping)

# -------------------- Identify High-Risk Days --------------------
high_risk_days = data[data["Overall_AQI_Level"] >= 4]
print(f"Found {len(high_risk_days)} high-risk days")

# -------------------- Insert into MySQL --------------------
session = SessionLocal()
try:
    for date, row in high_risk_days.iterrows():
        alert = Alert(
            date=date,
            overall_aqi_category=row["Overall_AQI_Category"],
            message=f"High AQI: {row['Overall_AQI_Category']} on {date.date()}"
        )
        session.add(alert)
    session.commit()
    print(f"✅ Inserted {len(high_risk_days)} historical high-risk alerts into MySQL")
except Exception as e:
    session.rollback()
    print("❌ Error inserting alerts:", e)
finally:
    session.close()
