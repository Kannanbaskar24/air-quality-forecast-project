from tensorflow.keras.models import load_model
import joblib
import os
import pandas as pd
import numpy as np
from prophet import Prophet

def load_best_model(pollutant):
    model_dir = "../models"

    # ARIMA
    arima_path = os.path.join(model_dir, f"{pollutant}_arima.pkl")
    if os.path.exists(arima_path):
        return joblib.load(arima_path), "ARIMA"

    # Prophet
    prophet_path = os.path.join(model_dir, f"{pollutant}_prophet.pkl")
    if os.path.exists(prophet_path):
        m = Prophet()
        m = m.load(prophet_path)
        return m, "Prophet"

    # XGBoost
    xgb_path = os.path.join(model_dir, f"{pollutant}_xgb.pkl")
    scaler_path = os.path.join(model_dir, f"{pollutant}_xgb_scaler.pkl")
    if os.path.exists(xgb_path) and os.path.exists(scaler_path):
        model = joblib.load(xgb_path)
        scaler = joblib.load(scaler_path)
        return (model, scaler), "XGBoost"

    # LSTM
    lstm_path = os.path.join(model_dir, f"{pollutant}_lstm.keras")
    if os.path.exists(lstm_path):
        model = load_model(lstm_path)
        return model, "LSTM"

    # Persistence
    persistence_path = os.path.join(model_dir, f"{pollutant}_persistence.pkl")
    if os.path.exists(persistence_path):
        series = joblib.load(persistence_path)
        return series, "Persistence"

    return None, None



def forecast_future(pollutant, history, days=7):
    model, mtype = load_best_model(pollutant)
    if model is None:
        raise ValueError(f"No saved model found for {pollutant}")

    # --- ARIMA ---
    if mtype == "ARIMA":
        forecast = model.forecast(steps=days)
        return pd.Series(forecast, index=pd.date_range(history.index[-1]+pd.Timedelta("1D"), periods=days, freq="D"))

    # --- Prophet ---
    if mtype == "Prophet":
        df = pd.DataFrame({"ds": history.index, "y": history.values})
        df["ds"] = df["ds"].dt.tz_localize(None)
        model.fit(df)   # refit with historical data
        future = model.make_future_dataframe(periods=days, freq="D")
        future["ds"] = future["ds"].dt.tz_localize(None)
        forecast = model.predict(future)
        return forecast.set_index("ds")["yhat"].iloc[-days:]

    # --- XGBoost ---
    if mtype == "XGBoost":
        model, scaler = model
        def make_lags(series, n_lags=7):
            df_lag = pd.DataFrame({"y": series})
            for lag in range(1, n_lags+1):
                df_lag[f"lag_{lag}"] = df_lag["y"].shift(lag)
            return df_lag.dropna()

        last_vals = history.values[-7:].tolist()
        preds = []
        for _ in range(days):
            X_input = pd.DataFrame([last_vals[-7:]], columns=[f"lag_{i}" for i in range(1,8)])
            X_scaled = scaler.transform(X_input)
            yhat = model.predict(X_scaled)[0]
            preds.append(yhat)
            last_vals.append(yhat)

        return pd.Series(preds, index=pd.date_range(history.index[-1]+pd.Timedelta("1D"), periods=days, freq="D"))

    # --- LSTM ---
    if mtype == "LSTM":
        last_vals = history.values[-7:]
        preds = []
        for _ in range(days):
            X_input = np.array(last_vals[-7:]).reshape((1,7,1))
            yhat = model.predict(X_input, verbose=0)[0][0]
            preds.append(yhat)
            last_vals = np.append(last_vals, yhat)
        return pd.Series(preds, index=pd.date_range(history.index[-1]+pd.Timedelta("1D"), periods=days, freq="D"))

    # --- Persistence ---
    if mtype == "Persistence":
        last_vals = model.values[-1]  # last observed value
        preds = [last_vals]*days
        return pd.Series(preds, index=pd.date_range(history.index[-1]+pd.Timedelta("1D"), periods=days, freq="D"))
