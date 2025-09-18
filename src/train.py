# ===============================================================
# train.py - Model Training & Forecasting Pipeline
# ===============================================================

import os
from pathlib import Path
import pandas as pd
import joblib
from sklearn.metrics import mean_absolute_error, mean_squared_error
from src.aqi import compute_overall_aqi_from_df  # make sure src/aqi.py exists

# ===================== Paths =====================
BASE_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = BASE_DIR / "data" / "processed"
MODELS_DIR = BASE_DIR / "models"
REPORTS_DIR = BASE_DIR / "reports"

MODELS_DIR.mkdir(parents=True, exist_ok=True)
REPORTS_DIR.mkdir(parents=True, exist_ok=True)

POLLUTANTS = ["PM2.5", "PM10", "NO2", "O3", "CO", "SO2"]

# ===================== Simple Persistence Baseline =====================
def persistence_forecast(series, days=7):
    """Forecast by repeating the last observed value"""
    last_valid = series.dropna()
    if last_valid.empty:
        vals = [0.0] * days
    else:
        last_value = float(last_valid.iloc[-1])
        vals = [last_value] * days
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
        except Exception:
            pass
    return model_obj, metrics, forecast

# ===================== Retrain All Models =====================
def retrain_all_models(historical_csv_path=None, forecast_days=7):
    if historical_csv_path is None:
        historical_csv_path = DATA_DIR / "air_quality_cleaned.csv"
    else:
        historical_csv_path = Path(historical_csv_path)

    if not historical_csv_path.exists():
        raise FileNotFoundError(f"Historical CSV not found: {historical_csv_path}")

    df = pd.read_csv(historical_csv_path, index_col=0, parse_dates=True)

    forecast_dict, metrics = {}, {}
    for p in POLLUTANTS:
        if p not in df.columns:
            continue
        series = df[p].dropna().asfreq("D")
        model_obj, m, forecast = train_model_for_pollutant(p, series, days=forecast_days)
        joblib.dump(model_obj, MODELS_DIR / f"{p}_model.pkl")
        metrics[p] = m
        forecast_dict[p] = forecast

    if not forecast_dict:
        raise RuntimeError("No forecasts generated (check input CSV columns).")

    # Forecast only
    forecast_df = pd.DataFrame(forecast_dict)
    forecast_df.index.name = "Date"
    forecast_with_aqi = compute_overall_aqi_from_df(forecast_df)

    # Historical with AQI
    historical_with_aqi = compute_overall_aqi_from_df(df)

    # Combine History + Forecast
    combined = pd.concat([historical_with_aqi, forecast_with_aqi])
    combined = combined[~combined.index.duplicated(keep="last")]  # remove duplicates

    # Save combined dataset
    forecast_path = DATA_DIR / "forecast_7days_full.csv"
    combined.to_csv(forecast_path)

    # ===================== Alerts (Append History) =====================
    alerts_path = DATA_DIR / "high_risk_alerts.csv"
    high_risk = forecast_with_aqi[forecast_with_aqi["Overall_AQI_Level"] >= 4]

    if not high_risk.empty:
        if alerts_path.exists():
            old_alerts = pd.read_csv(alerts_path, index_col=0, parse_dates=True)
            combined_alerts = pd.concat([old_alerts, high_risk])
            combined_alerts = combined_alerts[~combined_alerts.index.duplicated(keep="last")]
            combined_alerts.to_csv(alerts_path)
        else:
            high_risk.to_csv(alerts_path)
    else:
        if not alerts_path.exists():
            forecast_with_aqi.head(0).to_csv(alerts_path)

    # ===================== Metrics =====================
    metrics_df = pd.DataFrame(metrics).T
    metrics_path = MODELS_DIR / "metrics.csv"
    metrics_df.to_csv(metrics_path)

    return {
        "forecast_path": str(forecast_path),
        "alerts_path": str(alerts_path),
        "metrics_path": str(metrics_path),
        "models_dir": str(MODELS_DIR)
    }

if __name__ == "__main__":
    print("Retraining (baseline persistence)...")
    print(retrain_all_models())
