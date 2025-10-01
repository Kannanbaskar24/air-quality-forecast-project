import os
import datetime
from sqlalchemy import create_engine, Column, Integer, String, Float, DateTime
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from dotenv import load_dotenv
import pandas as pd
# Load environment variables from .env
load_dotenv()

Base = declarative_base()

# ==========================
# Table Definitions
# ==========================
class Measurement(Base):
    __tablename__ = "measurements"
    id = Column(Integer, primary_key=True, autoincrement=True)
    timestamp = Column(DateTime)
    station_id = Column(String(50))
    pollutant = Column(String(50))
    value = Column(Float)

class Prediction(Base):
    __tablename__ = "predictions"
    id = Column(Integer, primary_key=True, autoincrement=True)
    date = Column(DateTime, default=datetime.datetime.utcnow)
    pollutant = Column(String(50))
    value = Column(Float)

class Alert(Base):
    __tablename__ = "alerts"
    id = Column(Integer, primary_key=True, autoincrement=True)
    date = Column(DateTime)
    overall_aqi_category = Column(String(50))
    message = Column(String(255))

class ModelMeta(Base):
    __tablename__ = "models"
    id = Column(Integer, primary_key=True, autoincrement=True)
    model_name = Column(String(100))
    version = Column(String(50))
    mae = Column(Float)
    rmse = Column(Float)
    trained_at = Column(DateTime, default=datetime.datetime.utcnow)

# ==========================
# Database Connection
# ==========================
MYSQL_USER = os.getenv("MYSQL_USER", "root")
MYSQL_PASS = os.getenv("MYSQL_PASS", "1224")
MYSQL_HOST = os.getenv("MYSQL_HOST", "127.0.0.1")
MYSQL_PORT = os.getenv("MYSQL_PORT", 3306)
MYSQL_DB   = os.getenv("MYSQL_DB", "airaware")

DATABASE_URL = f"mysql+mysqlconnector://{MYSQL_USER}:{MYSQL_PASS}@{MYSQL_HOST}:{MYSQL_PORT}/{MYSQL_DB}"

engine = create_engine(DATABASE_URL, echo=True)
SessionLocal = sessionmaker(bind=engine)

# Create tables if they don't exist
def init_db():
    Base.metadata.create_all(engine)

if __name__ == "__main__":
    init_db()
    print("✅ Database initialized successfully!")
