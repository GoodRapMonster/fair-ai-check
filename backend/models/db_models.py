from sqlalchemy import Column, String, Float, DateTime, JSON, Text, Boolean
from datetime import datetime
from core.database import Base

class AnalysisRecord(Base):
    __tablename__ = "analyses"

    analysis_id = Column(String, primary_key=True, index=True)
    dataset_id = Column(String, index=True)
    timestamp = Column(DateTime, default=datetime.utcnow)
    
    # Metadata
    outcome = Column(String)
    domain = Column(String)
    
    # Results
    bias_score = Column(Float)
    risk_level = Column(String)
    
    # Full JSON payload of AnalyzeResponse for reloading
    full_result = Column(JSON)

class DriftSnapshot(Base):
    __tablename__ = "drift_history"

    id = Column(String, primary_key=True)  # analysis_id + "_" + timestamp
    analysis_id = Column(String, index=True)
    timestamp = Column(DateTime, default=datetime.utcnow)
    di_per_attr = Column(JSON)  # Dict[str, float]

class User(Base):
    __tablename__ = "users"

    username = Column(String, primary_key=True, index=True)
    hashed_password = Column(String)
    is_active = Column(Boolean, default=True)
