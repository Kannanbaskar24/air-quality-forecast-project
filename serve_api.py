from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pandas as pd
import os, joblib
from tensorflow.keras.models import load_model
from src.db import SessionLocal, Prediction, Alert
from src.logger import get_logger
from datetime import datetime, timedelta
import numpy as np

logger = get_logger()
app = FastAPI(title="AirAware API")

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "processed")
MODEL_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models")
pollutants = ["PM2.5","PM10","NO2","O3","CO","SO2"]

pollutant_models = {}
for p in pollutants:
    h5_file = os.path.join(MODEL_DIR, f"{p}_lstm.h5")
    pkl_file = os.path.join(MODEL_DIR, f"{p}_model.pkl")
    try:
        if os.path.exists(h5_file):
            pollutant_models[p] = load_model(h5_file)
            logger.info(f"Loaded LSTM model for {p}")
        elif os.path.exists(pkl_file):
            pollutant_models[p] = joblib.load(pkl_file)
            logger.info(f"Loaded PKL model for {p}")
        else:
            logger.warning(f"No model found for {p}")
    except Exception as e:
        logger.error(f"Error loading model for {p}: {e}")

class ForecastRequest(BaseModel):
    values: list[float]

@app.post("/predict/{pollutant}")
def predict_pollutant(pollutant: str, req: ForecastRequest):
    logger.info(f"Received API request for {pollutant} with {len(req.values)} values")
    if pollutant not in pollutants:
        raise HTTPException(status_code=400, detail="Invalid pollutant")
    model = pollutant_models.get(pollutant)
    if not model:
        raise HTTPException(status_code=404, detail=f"No model loaded for {pollutant}")

    X = np.array(req.values).reshape(1, -1)
    try:
        forecast = model.predict(X).flatten().tolist() if hasattr(model, "predict") else model.predict(X).flatten().tolist()
    except Exception as e:
        logger.error(f"Prediction error for {pollutant}: {e}")
        raise HTTPException(status_code=500, detail=f"Prediction error: {e}")

    session = SessionLocal()
    try:
        today = datetime.now()
        for i, val in enumerate(forecast):
            pred_date = today + timedelta(days=i+1)
            session.add(Prediction(date=pred_date, pollutant=pollutant, value=float(val)))
        session.commit()
        logger.info(f"Saved forecast to DB for {pollutant} dates: {today} to {today+timedelta(days=len(forecast))}")
    except Exception as e:
        session.rollback()
        logger.error(f"DB error saving forecasts for {pollutant}: {e}")
        raise HTTPException(status_code=500, detail=f"DB error saving forecasts: {e}")
    finally:
        session.close()
    return {"pollutant": pollutant, "forecast": forecast}

@app.post("/retrain")
def retrain_models():
    from src.train import retrain_all_models
    csv_file = os.path.join(DATA_DIR, "air_quality_cleaned.csv")
    try:
        result = retrain_all_models(csv_file)
        for p in pollutants:
            h5_file = os.path.join(MODEL_DIR, f"{p}_lstm.h5")
            pkl_file = os.path.join(MODEL_DIR, f"{p}_model.pkl")
            if os.path.exists(h5_file):
                pollutant_models[p] = load_model(h5_file)
            elif os.path.exists(pkl_file):
                pollutant_models[p] = joblib.load(pkl_file)
        logger.info("Retraining completed via API")
        return {"status": "success", "details": result}
    except Exception as e:
        logger.error(f"Retraining error via API: {e}")
        raise HTTPException(status_code=500, detail=f"Retraining error: {e}")

@app.get("/alerts")
def get_high_risk_alerts():
    session = SessionLocal()
    try:
        df = pd.read_sql(session.query(Alert).statement, session.bind, parse_dates=['date'])
        if df.empty:
            return []
        df['date'] = df['date'].astype(str)
        logger.info("Fetched alerts from DB via API")
        return df.to_dict(orient="records")
    except Exception as e:
        logger.error(f"Error fetching alerts via API: {e}")
        raise HTTPException(status_code=500, detail=f"Error fetching alerts: {e}")
    finally:
        session.close()
