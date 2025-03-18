"""
SQLAlchemy models for visualizations
"""

import json
from sqlalchemy import Column, String, Integer, DateTime, Text, ForeignKey, JSON, Boolean
from sqlalchemy.sql import func
from sqlalchemy.orm import relationship

from app.db.base import Base


class VisualizationModel(Base):
    """Visualization database model"""
    __tablename__ = "visualizations"
    
    id = Column(String, primary_key=True, index=True)
    name = Column(String, nullable=False)
    description = Column(Text, nullable=True)
    type = Column(String, nullable=False)  # survival_curve, feature_importance, prediction, etc.
    created_at = Column(DateTime, default=func.now(), nullable=False)
    updated_at = Column(DateTime, nullable=True, onupdate=func.now())
    analysis_id = Column(String, ForeignKey("analyses.id", ondelete="CASCADE"), nullable=False)
    
    # JSON fields
    config = Column(JSON, nullable=True)
    data = Column(JSON, nullable=False)
    
    # Sharing options
    shared = Column(Boolean, default=False, nullable=False)
    share_token = Column(String, nullable=True, unique=True, index=True)
    expires_at = Column(DateTime, nullable=True)
    
    # Relationships
    analysis = relationship("AnalysisModel", back_populates="visualizations")
    
    def __repr__(self):
        return f"<Visualization {self.name}>"
    
    @property
    def config_dict(self):
        """Convert config JSON to dict"""
        if isinstance(self.config, str):
            return json.loads(self.config)
        return self.config or {}
    
    @property
    def data_dict(self):
        """Convert data JSON to dict"""
        if isinstance(self.data, str):
            return json.loads(self.data)
        return self.data or {}
