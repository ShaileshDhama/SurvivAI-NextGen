"""
Report Service - Handles generation of AI-powered reports and insights for survival analysis models.
"""

import os
import json
import logging
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Any, Union

import pandas as pd
import numpy as np
from fastapi import HTTPException
from jinja2 import Environment, FileSystemLoader

from app.core.config import settings
from app.db.repositories.model_repository import ModelRepository
from app.db.repositories.dataset_repository import DatasetRepository
from app.db.repositories.analysis_repository import AnalysisRepository
from app.ml.models.base import BaseSurvivalModel
from app.ml.models.factory import ModelFactory
from app.services.ai_service import AIService
from app.schemas.report import (
    ReportType, 
    ReportConfig, 
    ReportFormat,
    SurvivalSummaryReport,
    RiskStratificationReport,
    SurvivalTrendReport,
    ReportInsight
)

logger = logging.getLogger(__name__)

class ReportService:
    """Service for generating AI-powered reports from survival analysis models."""
    
    def __init__(
        self, 
        model_repository: ModelRepository,
        dataset_repository: DatasetRepository,
        analysis_repository: AnalysisRepository,
        ai_service: AIService
    ):
        self.model_repository = model_repository
        self.dataset_repository = dataset_repository
        self.analysis_repository = analysis_repository
        self.ai_service = ai_service
        self.template_env = Environment(
            loader=FileSystemLoader(os.path.join(settings.TEMPLATES_DIR, "reports"))
        )
    
    async def generate_report(
        self, 
        model_id: str, 
        report_type: ReportType,
        config: Optional[ReportConfig] = None,
        format: ReportFormat = ReportFormat.PDF
    ) -> Dict[str, Any]:
        """
        Generate a report for the specified model based on the report type and configuration.
        
        Args:
            model_id: The ID of the survival model
            report_type: Type of report to generate
            config: Configuration options for the report
            format: Output format for the report
            
        Returns:
            Dict containing the report content and metadata
        """
        # Get the model
        model_data = await self.model_repository.get_by_id(model_id)
        if not model_data:
            raise HTTPException(status_code=404, detail=f"Model with ID {model_id} not found")
        
        # Get the related dataset
        dataset_id = model_data.get("dataset_id")
        dataset = await self.dataset_repository.get_by_id(dataset_id)
        if not dataset:
            raise HTTPException(status_code=404, detail=f"Dataset with ID {dataset_id} not found")
        
        # Load the model
        model_instance = await ModelFactory.load_model(model_id)
        
        # Generate appropriate report based on type
        if report_type == ReportType.SURVIVAL_SUMMARY:
            report_data = await self._generate_survival_summary(model_instance, model_data, dataset, config)
        elif report_type == ReportType.RISK_STRATIFICATION:
            report_data = await self._generate_risk_stratification(model_instance, model_data, dataset, config)
        elif report_type == ReportType.SURVIVAL_TREND:
            report_data = await self._generate_survival_trend(model_instance, model_data, dataset, config)
        else:
            raise HTTPException(status_code=400, detail=f"Unsupported report type: {report_type}")
        
        # Format the report for output
        output = await self._format_report(report_data, format, config)
        
        # Store the report in the database
        report_metadata = {
            "model_id": model_id,
            "report_type": report_type,
            "created_at": datetime.utcnow().isoformat(),
            "format": format,
            "config": config,
        }
        
        return {
            "metadata": report_metadata,
            "content": output
        }
    
    async def _generate_survival_summary(
        self, 
        model: BaseSurvivalModel,
        model_data: Dict[str, Any],
        dataset: Dict[str, Any],
        config: Optional[ReportConfig]
    ) -> SurvivalSummaryReport:
        """Generate a survival summary report with AI-powered insights."""
        
        # Get model predictions and metrics
        metrics = model_data.get("metrics", {})
        
        # Load the test data
        test_data = await self._load_test_data(dataset.get("id"))
        
        # Generate predictions if test data is available
        predictions = None
        if test_data is not None:
            X_test, y_test = test_data
            predictions = model.predict(X_test)
        
        # Use AI service to generate insights from model and data
        insights = await self._generate_ai_insights(
            model=model,
            model_data=model_data,
            dataset=dataset,
            predictions=predictions,
            insight_type="survival_summary"
        )
        
        # Build the survival curve data
        survival_curve_data = await self._build_survival_curve_data(model, dataset)
        
        # Build the report data structure
        report_data = SurvivalSummaryReport(
            title=f"Survival Analysis Summary Report - {model_data.get('name', 'Unnamed Model')}",
            model_name=model_data.get("name", "Unnamed Model"),
            model_type=model_data.get("type", "Unknown"),
            dataset_name=dataset.get("name", "Unnamed Dataset"),
            creation_date=datetime.utcnow().isoformat(),
            metrics=metrics,
            survival_curves=survival_curve_data,
            insights=insights,
            parameters=model_data.get("parameters", {}),
            sample_size=dataset.get("row_count", 0),
            variables=dataset.get("columns", []),
            event_distribution=await self._get_event_distribution(dataset)
        )
        
        return report_data
    
    async def _generate_risk_stratification(
        self, 
        model: BaseSurvivalModel,
        model_data: Dict[str, Any],
        dataset: Dict[str, Any],
        config: Optional[ReportConfig]
    ) -> RiskStratificationReport:
        """Generate a risk stratification report with high/low risk groups."""
        
        # Load the data
        data = await self._load_dataset(dataset.get("id"))
        
        # Set risk thresholds (from config or default)
        thresholds = config.get("risk_thresholds", [0.25, 0.75]) if config else [0.25, 0.75]
        
        # Get risk scores from model
        risk_scores = model.predict_risk(data)
        
        # Stratify patients by risk score
        low_risk_mask = risk_scores <= thresholds[0]
        high_risk_mask = risk_scores >= thresholds[1]
        medium_risk_mask = ~(low_risk_mask | high_risk_mask)
        
        risk_groups = {
            "low_risk": {
                "count": int(low_risk_mask.sum()),
                "percentage": float(low_risk_mask.sum() / len(risk_scores) * 100),
                "threshold": float(thresholds[0])
            },
            "medium_risk": {
                "count": int(medium_risk_mask.sum()),
                "percentage": float(medium_risk_mask.sum() / len(risk_scores) * 100),
                "threshold_low": float(thresholds[0]),
                "threshold_high": float(thresholds[1])
            },
            "high_risk": {
                "count": int(high_risk_mask.sum()),
                "percentage": float(high_risk_mask.sum() / len(risk_scores) * 100),
                "threshold": float(thresholds[1])
            }
        }
        
        # Generate characteristics for each risk group
        risk_characteristics = await self._analyze_risk_groups(data, risk_scores, thresholds)
        
        # Use AI to generate insights for risk stratification
        insights = await self._generate_ai_insights(
            model=model,
            model_data=model_data,
            dataset=dataset,
            risk_groups=risk_groups,
            risk_characteristics=risk_characteristics,
            insight_type="risk_stratification"
        )
        
        # Build the report data structure
        report_data = RiskStratificationReport(
            title=f"Risk Stratification Report - {model_data.get('name', 'Unnamed Model')}",
            model_name=model_data.get("name", "Unnamed Model"),
            model_type=model_data.get("type", "Unknown"),
            dataset_name=dataset.get("name", "Unnamed Dataset"),
            creation_date=datetime.utcnow().isoformat(),
            risk_groups=risk_groups,
            risk_characteristics=risk_characteristics,
            survival_by_risk_group=await self._get_survival_by_risk_group(model, data, risk_scores, thresholds),
            feature_importance=model_data.get("feature_importance", {}),
            insights=insights,
            parameters=model_data.get("parameters", {})
        )
        
        return report_data
    
    async def _generate_survival_trend(
        self, 
        model: BaseSurvivalModel,
        model_data: Dict[str, Any],
        dataset: Dict[str, Any],
        config: Optional[ReportConfig]
    ) -> SurvivalTrendReport:
        """Generate a survival trend prediction report with deep learning forecasting."""
        
        # Load historical data for trend analysis
        historical_data = await self._load_historical_data(dataset.get("id"))
        
        # Set time horizon for predictions (from config or default)
        time_horizon = config.get("time_horizon", 36) if config else 36  # Default 36 months
        
        # Generate survival trend predictions using deep learning
        trend_predictions = await self._predict_survival_trends(model, historical_data, time_horizon)
        
        # Generate confidence intervals for predictions
        confidence_intervals = await self._generate_confidence_intervals(trend_predictions)
        
        # Use AI to generate insights for survival trends
        insights = await self._generate_ai_insights(
            model=model,
            model_data=model_data,
            dataset=dataset,
            trend_predictions=trend_predictions,
            confidence_intervals=confidence_intervals,
            time_horizon=time_horizon,
            insight_type="survival_trend"
        )
        
        # Build the report data structure
        report_data = SurvivalTrendReport(
            title=f"Survival Trend Forecast - {model_data.get('name', 'Unnamed Model')}",
            model_name=model_data.get("name", "Unnamed Model"),
            model_type=model_data.get("type", "Unknown"),
            dataset_name=dataset.get("name", "Unnamed Dataset"),
            creation_date=datetime.utcnow().isoformat(),
            time_horizon=time_horizon,
            trend_predictions=trend_predictions,
            confidence_intervals=confidence_intervals,
            historical_performance=await self._get_historical_performance(historical_data),
            key_inflection_points=await self._identify_inflection_points(trend_predictions),
            insights=insights,
            factors_affecting_trends=await self._identify_trend_factors(model, historical_data, trend_predictions)
        )
        
        return report_data
    
    async def _generate_ai_insights(
        self,
        model: BaseSurvivalModel,
        model_data: Dict[str, Any],
        dataset: Dict[str, Any],
        insight_type: str,
        **kwargs
    ) -> List[ReportInsight]:
        """
        Generate AI-powered insights for the report.
        
        Args:
            model: The survival model instance
            model_data: Model metadata from repository
            dataset: Dataset metadata from repository
            insight_type: Type of insights to generate
            **kwargs: Additional data for context
            
        Returns:
            List of generated insights
        """
        # Prepare context for AI service
        context = {
            "model_data": model_data,
            "dataset": dataset,
            "insight_type": insight_type,
            **kwargs
        }
        
        # Call AI service to generate insights
        generated_insights = await self.ai_service.generate_report_insights(context)
        
        # Format insights
        insights = []
        for idx, insight in enumerate(generated_insights):
            insights.append(ReportInsight(
                id=f"insight-{idx+1}",
                title=insight.get("title", f"Insight {idx+1}"),
                description=insight.get("description", ""),
                importance=insight.get("importance", "medium"),
                category=insight.get("category", "general")
            ))
            
        return insights
    
    async def _format_report(
        self, 
        report_data: Union[SurvivalSummaryReport, RiskStratificationReport, SurvivalTrendReport],
        format: ReportFormat,
        config: Optional[ReportConfig]
    ) -> bytes:
        """Format the report in the specified output format."""
        if format == ReportFormat.PDF:
            return await self._generate_pdf_report(report_data)
        elif format == ReportFormat.CSV:
            return await self._generate_csv_report(report_data)
        elif format == ReportFormat.JSON:
            return await self._generate_json_report(report_data)
        elif format == ReportFormat.HTML:
            return await self._generate_html_report(report_data)
        else:
            raise HTTPException(status_code=400, detail=f"Unsupported report format: {format}")
    
    async def _generate_pdf_report(self, report_data: Any) -> bytes:
        """Generate a PDF report from the data."""
        # Template name based on report type
        template_name = f"{report_data.__class__.__name__.lower()}.html"
        
        # Render HTML template
        template = self.template_env.get_template(template_name)
        html_content = template.render(**report_data.dict())
        
        # Convert HTML to PDF
        # Implementation using a PDF library like weasyprint or pdfkit would go here
        # For now, we'll just return the HTML as bytes
        return html_content.encode('utf-8')
    
    async def _generate_csv_report(self, report_data: Any) -> bytes:
        """Generate a CSV report from the data."""
        # Convert report data to a flat structure suitable for CSV
        flat_data = self._flatten_report_data(report_data)
        
        # Convert to DataFrame and then to CSV
        df = pd.DataFrame(flat_data)
        return df.to_csv(index=False).encode('utf-8')
    
    async def _generate_json_report(self, report_data: Any) -> bytes:
        """Generate a JSON report from the data."""
        return json.dumps(report_data.dict(), indent=2).encode('utf-8')
    
    async def _generate_html_report(self, report_data: Any) -> bytes:
        """Generate an HTML report from the data."""
        # Template name based on report type
        template_name = f"{report_data.__class__.__name__.lower()}.html"
        
        # Render HTML template
        template = self.template_env.get_template(template_name)
        html_content = template.render(**report_data.dict())
        
        return html_content.encode('utf-8')
    
    async def email_report(
        self, 
        report_id: str, 
        email: str, 
        subject: Optional[str] = None,
        message: Optional[str] = None
    ) -> bool:
        """
        Email a generated report to the specified address.
        
        Args:
            report_id: ID of the previously generated report
            email: Email address to send the report to
            subject: Optional email subject
            message: Optional email message body
            
        Returns:
            True if email was sent successfully
        """
        # Implementation would use an email service
        # For now, we'll just log the request
        logger.info(f"Email report {report_id} to {email}")
        return True
    
    # Helper methods for data loading and analysis
    
    async def _load_test_data(self, dataset_id: str) -> Optional[Tuple[pd.DataFrame, pd.DataFrame]]:
        """Load test data for the dataset if available."""
        # Implementation would load test data split for the dataset
        return None
    
    async def _load_dataset(self, dataset_id: str) -> pd.DataFrame:
        """Load the complete dataset."""
        # Implementation would load the dataset from storage
        return pd.DataFrame()
    
    async def _load_historical_data(self, dataset_id: str) -> pd.DataFrame:
        """Load historical data for trend analysis."""
        # Implementation would load historical data related to the dataset
        return pd.DataFrame()
    
    async def _build_survival_curve_data(
        self, 
        model: BaseSurvivalModel,
        dataset: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Build survival curve data for the report."""
        # Implementation would generate survival curve data
        return {}
    
    async def _get_event_distribution(self, dataset: Dict[str, Any]) -> Dict[str, Any]:
        """Get the distribution of events in the dataset."""
        # Implementation would analyze event distribution
        return {}
    
    async def _analyze_risk_groups(
        self, 
        data: pd.DataFrame,
        risk_scores: np.ndarray,
        thresholds: List[float]
    ) -> Dict[str, Any]:
        """Analyze the characteristics of different risk groups."""
        # Implementation would analyze risk group characteristics
        return {}
    
    async def _get_survival_by_risk_group(
        self,
        model: BaseSurvivalModel,
        data: pd.DataFrame,
        risk_scores: np.ndarray,
        thresholds: List[float]
    ) -> Dict[str, Any]:
        """Get survival curves stratified by risk group."""
        # Implementation would generate survival curves by risk group
        return {}
    
    async def _predict_survival_trends(
        self,
        model: BaseSurvivalModel,
        historical_data: pd.DataFrame,
        time_horizon: int
    ) -> Dict[str, Any]:
        """Predict survival trends using deep learning models."""
        # Implementation would use deep learning to predict trends
        return {}
    
    async def _generate_confidence_intervals(
        self,
        trend_predictions: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Generate confidence intervals for trend predictions."""
        # Implementation would calculate confidence intervals
        return {}
    
    async def _get_historical_performance(
        self,
        historical_data: pd.DataFrame
    ) -> Dict[str, Any]:
        """Get historical performance metrics."""
        # Implementation would analyze historical performance
        return {}
    
    async def _identify_inflection_points(
        self,
        trend_predictions: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """Identify key inflection points in predicted trends."""
        # Implementation would identify trend inflection points
        return []
    
    async def _identify_trend_factors(
        self,
        model: BaseSurvivalModel,
        historical_data: pd.DataFrame,
        trend_predictions: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """Identify factors affecting survival trends."""
        # Implementation would analyze factors affecting trends
        return []
    
    def _flatten_report_data(self, report_data: Any) -> Dict[str, Any]:
        """Flatten complex report data structure for CSV output."""
        # Implementation would flatten nested structure
        return {}
