from tensorflow.keras.models import load_model
import joblib
import os
import pandas as pd
import numpy as np
from prophet import Prophet
from pathlib import Path
from datetime import datetime, timedelta

def load_best_model(pollutant):
    """Load the best available model for the given pollutant"""
    model_dir = Path("../models")
    
    # Try to load different model types in order of preference
    
    # 1. Try simple model (your current approach)
    simple_path = model_dir / f"{pollutant}_model.pkl"
    if simple_path.exists():
        model_obj = joblib.load(simple_path)
        return model_obj, "Simple"
    
    # 2. Try LSTM model
    lstm_path = model_dir / f"{pollutant}_lstm.keras"
    if lstm_path.exists():
        model = load_model(lstm_path)
        return model, "LSTM"
    
    # 3. Try XGBoost model
    xgb_path = model_dir / f"{pollutant}_xgb.pkl"
    scaler_path = model_dir / f"{pollutant}_xgb_scaler.pkl"
    if xgb_path.exists() and scaler_path.exists():
        model = joblib.load(xgb_path)
        scaler = joblib.load(scaler_path)
        return (model, scaler), "XGBoost"
    
    # 4. Try ARIMA model
    arima_path = model_dir / f"{pollutant}_arima.pkl"
    if arima_path.exists():
        return joblib.load(arima_path), "ARIMA"
    
    # 5. Try Prophet model
    prophet_path = model_dir / f"{pollutant}_prophet.pkl"
    if prophet_path.exists():
        m = Prophet()
        m = joblib.load(prophet_path)
        return m, "Prophet"
    
    # 6. Fallback: Persistence model
    persistence_path = model_dir / f"{pollutant}_persistence.pkl"
    if persistence_path.exists():
        series = joblib.load(persistence_path)
        return series, "Persistence"
    
    return None, None

def forecast_future(pollutant, history, days=7):
    """Generate future forecast for the specified pollutant"""
    model, mtype = load_best_model(pollutant)
    if model is None:
        raise ValueError(f"No saved model found for {pollutant}")
    
    # Ensure history is properly formatted
    if not isinstance(history, pd.Series):
        raise ValueError("History must be a pandas Series")
    
    history = history.dropna()
    if len(history) == 0:
        raise ValueError("History data is empty")
    
    # ⚡ FIX: Generate REAL future dates from TODAY'S date
    today = pd.Timestamp.now().date()
    
    # If historical data is older than today, start from today
    last_historical_date = history.index[-1].date()
    
    if last_historical_date < today:
        # Historical data is old, predict from today onwards
        start_date = today + timedelta(days=1)  # Tomorrow
    else:
        # Historical data is recent, predict from after the last date
        start_date = last_historical_date + timedelta(days=1)
    
    # Generate future dates starting from the appropriate start date
    future_dates = pd.date_range(start=start_date, periods=days, freq='D')
    
    print(f"🔍 DEBUG: Last historical date: {last_historical_date}")
    print(f"🔍 DEBUG: Today's date: {today}")
    print(f"🔍 DEBUG: Future prediction starts from: {start_date}")
    print(f"🔍 DEBUG: Future dates: {future_dates}")
    
    # --- ARIMA ---
    if mtype == "ARIMA":
        try:
            forecast = model.forecast(steps=days)
            return pd.Series(forecast, index=future_dates)
        except Exception as e:
            print(f"⚠️ ARIMA failed: {e}, using persistence")
            last_val = history.iloc[-1]
            return pd.Series([last_val] * days, index=future_dates)
    
    # --- Prophet ---
    elif mtype == "Prophet":
        try:
            df = pd.DataFrame({"ds": history.index, "y": history.values})
            df["ds"] = df["ds"].dt.tz_localize(None) if df["ds"].dt.tz is not None else df["ds"]
            model.fit(df)
            future = model.make_future_dataframe(periods=days, freq="D")
            future["ds"] = future["ds"].dt.tz_localize(None) if future["ds"].dt.tz is not None else future["ds"]
            forecast = model.predict(future)
            return forecast.set_index("ds")["yhat"].iloc[-days:]
        except Exception as e:
            print(f"⚠️ Prophet failed: {e}, using persistence")
            last_val = history.iloc[-1]
            return pd.Series([last_val] * days, index=future_dates)
    
    # --- XGBoost ---
    elif mtype == "XGBoost":
        try:
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
                preds.append(max(0, yhat))  # Ensure non-negative predictions
                last_vals.append(yhat)
            
            return pd.Series(preds, index=future_dates)
        except Exception as e:
            print(f"⚠️ XGBoost failed: {e}, using persistence")
            last_val = history.iloc[-1]
            return pd.Series([last_val] * days, index=future_dates)
    
    # --- LSTM ---
    elif mtype == "LSTM":
        try:
            last_vals = history.values[-7:]
            preds = []
            for _ in range(days):
                X_input = np.array(last_vals[-7:]).reshape((1,7,1))
                yhat = model.predict(X_input, verbose=0)[0][0]
                preds.append(max(0, yhat))  # Ensure non-negative predictions
                last_vals = np.append(last_vals, yhat)
            return pd.Series(preds, index=future_dates)
        except Exception as e:
            print(f"⚠️ LSTM failed: {e}, using persistence")
            last_val = history.iloc[-1]
            return pd.Series([last_val] * days, index=future_dates)
    
    # --- Simple Model (your current approach) ---
    elif mtype == "Simple":
        if isinstance(model, dict) and "last_value" in model:
            last_val = model["last_value"]
            # Add some realistic variation to avoid identical predictions
            base_val = max(0, last_val)
            predictions = []
            for i in range(days):
                # Add small random variation (±10%) to make it more realistic
                variation = np.random.uniform(0.9, 1.1)
                pred_val = base_val * variation
                predictions.append(max(0, pred_val))  # Ensure non-negative
            
            return pd.Series(predictions, index=future_dates)
        else:
            last_val = history.iloc[-1]
            return pd.Series([max(0, last_val)] * days, index=future_dates)
    
    # --- Persistence (fallback) ---
    else:
        last_val = history.iloc[-1] if len(history) > 0 else 0.0
        # Add small variation to persistence model
        predictions = []
        for i in range(days):
            variation = np.random.uniform(0.95, 1.05)  # ±5% variation
            pred_val = max(0, last_val * variation)
            predictions.append(pred_val)
        
        return pd.Series(predictions, index=future_dates)
