"""
SQLAlchemy models for analyses
"""

import json
from sqlalchemy import Column, String, Integer, DateTime, Text, ForeignKey, JSON
from sqlalchemy.sql import func
from sqlalchemy.orm import relationship

from app.db.base import Base


class AnalysisModel(Base):
    """Analysis database model"""
    __tablename__ = "analyses"
    
    id = Column(String, primary_key=True, index=True)
    name = Column(String, nullable=False)
    description = Column(Text, nullable=True)
    dataset_id = Column(String, ForeignKey("datasets.id", ondelete="CASCADE"), nullable=False)
    time_column = Column(String, nullable=False)
    event_column = Column(String, nullable=False)
    analysis_type = Column(String, nullable=False)
    covariates = Column(JSON, nullable=False, default=list)
    parameters = Column(JSON, nullable=False, default=dict)
    status = Column(String, nullable=False)  # pending, running, completed, failed
    created_at = Column(DateTime, default=func.now(), nullable=False)
    updated_at = Column(DateTime, nullable=True, onupdate=func.now())
    model_id = Column(String, ForeignKey("models.id", ondelete="SET NULL"), nullable=True)
    
    # Relationships
    dataset = relationship("DatasetModel", back_populates="analyses")
    model = relationship("ModelModel", back_populates="analyses")
    visualizations = relationship("VisualizationModel", back_populates="analysis", cascade="all, delete-orphan")
    
    # Store analysis results (potentially large)
    results = Column(JSON, nullable=True)
    feature_importance = Column(JSON, nullable=True)
    
    def __repr__(self):
        return f"<Analysis {self.name}>"
    
    @property
    def dataset_name(self):
        """Get the dataset name from relationship"""
        return self.dataset.name if self.dataset else None
    
    @property
    def covariates_list(self):
        """Convert covariates JSON to list"""
        if isinstance(self.covariates, str):
            return json.loads(self.covariates)
        return self.covariates or []
    
    @property
    def parameters_dict(self):
        """Convert parameters JSON to dict"""
        if isinstance(self.parameters, str):
            return json.loads(self.parameters)
        return self.parameters or {}
