"""
SQLAlchemy models for ML models
"""

import json
from sqlalchemy import Column, String, Integer, DateTime, Text, ForeignKey, JSON
from sqlalchemy.sql import func
from sqlalchemy.orm import relationship

from app.db.base import Base


class ModelModel(Base):
    """ML Model database model"""
    __tablename__ = "models"
    
    id = Column(String, primary_key=True, index=True)
    name = Column(String, nullable=False)
    description = Column(Text, nullable=True)
    model_type = Column(String, nullable=False)  # cox_ph, kaplan_meier, deep_surv, etc.
    version = Column(String, nullable=False, default="1.0.0")
    created_at = Column(DateTime, default=func.now(), nullable=False)
    updated_at = Column(DateTime, nullable=True, onupdate=func.now())
    
    # Path to stored model artifacts
    path = Column(String, nullable=False)
    file_size = Column(Integer, nullable=False)
    
    # Model-specific metadata
    metadata = Column(JSON, nullable=True)
    
    # Performance metrics
    metrics = Column(JSON, nullable=True)
    
    # Relationships
    analyses = relationship("AnalysisModel", back_populates="model")
    
    def __repr__(self):
        return f"<Model {self.name} ({self.model_type})>"
    
    @property
    def metadata_dict(self):
        """Convert metadata JSON to dict"""
        if isinstance(self.metadata, str):
            return json.loads(self.metadata)
        return self.metadata or {}
    
    @property
    def metrics_dict(self):
        """Convert metrics JSON to dict"""
        if isinstance(self.metrics, str):
            return json.loads(self.metrics)
        return self.metrics or {}
