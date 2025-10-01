# ===============================================================
# train.py - Model Training & Forecasting Pipeline with MySQL
# Full Logging + Drift Detection + Automatic Model Cards
# ===============================================================

import os
from pathlib import Path
import pandas as pd
import joblib
from sklearn.metrics import mean_absolute_error, mean_squared_error
from src.aqi import compute_overall_aqi_from_df
from src.db import SessionLocal, Prediction, Alert
from src.logger import get_logger
import datetime
from src.drift import check_drift  # Drift detection
import json

logger = get_logger()

# ===================== Paths =====================
BASE_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = BASE_DIR / "data" / "processed"
MODELS_DIR = BASE_DIR / "models"
REPORTS_DIR = BASE_DIR / "reports"

MODELS_DIR.mkdir(parents=True, exist_ok=True)
REPORTS_DIR.mkdir(parents=True, exist_ok=True)

POLLUTANTS = ["PM2.5", "PM10", "NO2", "O3", "CO", "SO2"]

# ===================== Persistence Forecast =====================
def persistence_forecast(series, days=7):
    last_valid = series.dropna()
    vals = [float(last_valid.iloc[-1])] * days if not last_valid.empty else [0.0] * days
    start = series.index.max() + pd.Timedelta(days=1)
    dates = pd.date_range(start=start, periods=days, freq="D")
    return pd.Series(vals, index=dates)

def train_model_for_pollutant(pollutant, series, days=7):
    forecast = persistence_forecast(series, days=days)
    last_valid = series.dropna()
    model_obj = {
        "type": "persistence",
        "last_value": float(last_valid.iloc[-1]) if not last_valid.empty else 0.0
    }
    metrics = {"model": "persistence", "MAE": None, "RMSE": None}

    if len(series.dropna()) >= days + 1:
        true = series.dropna().iloc[-days:]
        preds = series.dropna().shift(1).iloc[-days:]
        try:
            metrics["MAE"] = float(mean_absolute_error(true, preds))
            metrics["RMSE"] = float(mean_squared_error(true, preds, squared=False))
        except Exception as e:
            logger.error(f"Error computing metrics for {pollutant}: {e}")
    return model_obj, metrics, forecast

# ===================== Retrain All Models =====================
def retrain_all_models(historical_csv_path=None, forecast_days=7):
    logger.info("Starting retraining of all models...")
    historical_csv_path = historical_csv_path or DATA_DIR / "air_quality_cleaned.csv"
    historical_csv_path = Path(historical_csv_path)
    if not historical_csv_path.exists():
        logger.error(f"Historical CSV not found: {historical_csv_path}")
        raise FileNotFoundError(f"Historical CSV not found: {historical_csv_path}")

    df = pd.read_csv(historical_csv_path, index_col=0, parse_dates=True)
    forecast_dict, metrics = {}, {}
    alerts_list = []

    session = SessionLocal()
    try:
        for p in POLLUTANTS:
            if p not in df.columns:
                continue
            series = df[p].dropna().asfreq("D")

            # Drift detection
            prev_series_path = DATA_DIR / "air_quality_cleaned.csv"
            if prev_series_path.exists():
                prev_series = pd.read_csv(prev_series_path, index_col=0, parse_dates=True)[p].dropna()
                if check_drift(series, prev_series):
                    logger.warning(f"Drift detected in {p}: mean_old={prev_series.mean()}, mean_new={series.mean()}")

            model_obj, m, forecast = train_model_for_pollutant(p, series, days=forecast_days)

            joblib.dump(model_obj, MODELS_DIR / f"{p}_model.pkl")
            metrics[p] = m
            forecast_dict[p] = forecast

            forecast_start = forecast.index.min()
            forecast_end = forecast.index.max()
            session.query(Prediction).filter(
                Prediction.pollutant == p,
                Prediction.date >= forecast_start,
                Prediction.date <= forecast_end
            ).delete()

            for date, value in forecast.items():
                session.add(Prediction(date=date.to_pydatetime(), pollutant=p, value=float(value)))

            logger.info(f"Inserted forecast for {p} from {forecast_start} to {forecast_end}")

        session.commit()

        # ===================== Compute Alerts =====================
        forecast_df = pd.DataFrame(forecast_dict)
        forecast_df.index.name = "Date"
        forecast_with_aqi = compute_overall_aqi_from_df(forecast_df)

        high_risk = forecast_with_aqi[forecast_with_aqi["Overall_AQI_Level"] >= 4]
        for idx, row in high_risk.iterrows():
            exists = session.query(Alert).filter(Alert.date == idx.to_pydatetime()).first()
            if not exists:
                alert = Alert(date=idx.to_pydatetime(),
                              overall_aqi_category=row["Overall_AQI_Category"],
                              message=f"High-risk day: AQI {row['Overall_AQI_Level']}")
                session.add(alert)
                alerts_list.append(alert)
        session.commit()
        logger.info(f"High-risk alerts generated for dates: {[idx for idx in high_risk.index]}")

    except Exception as e:
        logger.error(f"Error during retraining: {e}")
        session.rollback()
        raise e
    finally:
        session.close()

    # ===================== Save CSV backups =====================
    forecast_df.to_csv(DATA_DIR / "forecast_7days_full.csv")
    high_risk.to_csv(DATA_DIR / "high_risk_alerts.csv")
    metrics_df = pd.DataFrame(metrics).T
    metrics_path = MODELS_DIR / "metrics.csv"
    metrics_df.to_csv(metrics_path)
    logger.info("Metrics saved successfully.")

    # ===================== Generate Model Cards =====================
    info = {
        "PM2.5": {
            "intended_use": "Forecast air quality for PM2.5 to help authorities take preventive actions.",
            "limitations": "Valid for stations with historical PM2.5 data. May not generalize to industrial zones with extreme pollution."
        },
        "PM10": {
            "intended_use": "Forecast air quality for PM10 to help authorities issue alerts.",
            "limitations": "Valid for urban monitoring stations. Accuracy may decrease in highly industrial regions."
        },
        "NO2": {
            "intended_use": "Forecast NO2 levels for air quality management and alerting.",
            "limitations": "Performance depends on historical NO2 trends and station coverage."
        },
        "O3": {
            "intended_use": "Forecast ozone levels to support air quality monitoring.",
            "limitations": "May not accurately predict short-term ozone spikes in industrial regions."
        },
        "CO": {
            "intended_use": "Forecast CO levels to support air quality alerts and public health decisions.",
            "limitations": "Prediction accuracy may vary in areas with irregular traffic patterns."
        },
        "SO2": {
            "intended_use": "Forecast SO2 levels for environmental monitoring and alerts.",
            "limitations": "Valid for stations with consistent SO2 readings. May not capture sudden industrial emissions."
        }
    }

    os.makedirs(MODELS_DIR, exist_ok=True)

    for pollutant, m in metrics.items():
        model_card = {
            "model_name": f"{pollutant} LSTM",
            "pollutant": pollutant,
            "type": "Persistence Model" if m["model"]=="persistence" else "LSTM Time Series Forecasting",
            "version": "v1.0",
            "training_date": str(datetime.datetime.now().date()),
            "data_source": "CPCB / OpenAQ",
            "MAE": m["MAE"],
            "RMSE": m["RMSE"],
            "intended_use": info[pollutant]["intended_use"],
            "limitations": info[pollutant]["limitations"]
        }

        filename = MODELS_DIR / f"model_card_{pollutant.replace('.', '')}.json"
        with open(filename, "w") as f:
            json.dump(model_card, f, indent=4)

    logger.info("✅ Model cards generated successfully in models/ folder!")

    return {
        "forecast_df": forecast_df,
        "alerts_df": high_risk,
        "forecast_path": str(DATA_DIR / "forecast_7days_full.csv"),
        "alerts_path": str(DATA_DIR / "high_risk_alerts.csv"),
        "metrics_path": str(metrics_path),
        "models_dir": str(MODELS_DIR)
    }

# ===================== Main =====================
if __name__ == "__main__":
    result = retrain_all_models()
    print(result["forecast_df"].head())
    print(result["alerts_df"].head())
