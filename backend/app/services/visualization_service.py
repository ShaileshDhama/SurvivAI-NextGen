"""
Visualization service - handles business logic for visualization operations
"""

import os
import uuid
import secrets
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime, timedelta
from fastapi import HTTPException

from app.models.visualization import Visualization, VisualizationCreate, ShareLink
from app.db.repositories.visualization_repository import VisualizationRepository
from app.db.repositories.analysis_repository import AnalysisRepository
from app.db.repositories.model_repository import ModelRepository
from app.core.config import settings


class VisualizationService:
    """Service for visualization operations"""
    
    def __init__(
        self, 
        repository: VisualizationRepository,
        analysis_repository: AnalysisRepository,
        model_repository: Optional[ModelRepository] = None
    ):
        """Initialize with repository dependency"""
        self.repository = repository
        self.analysis_repository = analysis_repository
        self.model_repository = model_repository
    
    async def get_visualizations(
        self, 
        analysis_id: Optional[str] = None,
        page: int = 1, 
        limit: int = 10
    ) -> Tuple[List[Visualization], int]:
        """Get paginated visualizations, optionally filtered by analysis ID"""
        return await self.repository.get_visualizations(
            analysis_id=analysis_id,
            skip=(page-1)*limit, 
            limit=limit
        )
    
    async def create_visualization(self, viz_data: VisualizationCreate) -> Visualization:
        """Create a new visualization from analysis results"""
        # Validate analysis exists
        analysis = await self.analysis_repository.get_analysis(viz_data.analysis_id)
        if not analysis:
            raise HTTPException(status_code=404, detail="Analysis not found")
        
        if analysis.status != "completed":
            raise HTTPException(status_code=400, detail="Analysis not completed")
        
        # Generate a unique ID
        viz_id = str(uuid.uuid4())
        
        # Fetch data for visualization based on type
        data = await self._get_visualization_data(viz_data.type, analysis.id, viz_data.data_settings)
        
        # Create visualization model
        visualization = Visualization(
            id=viz_id,
            name=viz_data.name,
            description=viz_data.description,
            type=viz_data.type,
            created_at=datetime.now(),
            analysis_id=viz_data.analysis_id,
            config=viz_data.config,
            data=data,
            shared=False
        )
        
        # Save to repository
        return await self.repository.create_visualization(visualization)
    
    async def _get_visualization_data(
        self, 
        viz_type: str, 
        analysis_id: str, 
        data_settings: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Fetch and format data for the specific visualization type"""
        # Get the analysis service
        from app.core.dependencies import get_analysis_service
        analysis_service = await get_analysis_service()
        
        # Get full analysis results
        results = await analysis_service.get_analysis_results(analysis_id)
        if not results:
            raise HTTPException(status_code=404, detail="Analysis results not found")
        
        # Extract data based on visualization type
        if viz_type == "survival_curve":
            return {
                "survival_curves": [curve.dict() for curve in results.survival_curves],
                "title": data_settings.get("title", "Survival Probability"),
                "x_axis": data_settings.get("x_axis", "Time"),
                "y_axis": data_settings.get("y_axis", "Survival Probability"),
                "plot_type": "kaplan_meier",  # Default to KM curves
                "confidence_intervals": data_settings.get("confidence_intervals", True),
                "risk_table": data_settings.get("risk_table", True),
                "stratification": data_settings.get("stratification", None)
            }
        elif viz_type == "feature_importance":
            if not results.feature_importance:
                raise HTTPException(status_code=404, detail="Feature importance not available for this analysis")
            
            return {
                "feature_importance": results.feature_importance.dict(),
                "title": data_settings.get("title", "Feature Importance"),
                "x_axis": data_settings.get("x_axis", "Importance"),
                "y_axis": data_settings.get("y_axis", "Features"),
                "plot_type": data_settings.get("plot_type", "bar"),  # bar, shap, or lime
                "top_n_features": data_settings.get("top_n_features", 10),
                "color_scheme": data_settings.get("color_scheme", "default")
            }
        elif viz_type == "model_summary":
            return {
                "model_summary": results.model_summary,
                "title": data_settings.get("title", "Model Summary"),
            }
        elif viz_type == "hazard_ratios":
            if not results.coefficients:
                raise HTTPException(status_code=404, detail="Hazard ratios not available for this analysis")
            
            return {
                "hazard_ratios": results.coefficients,
                "title": data_settings.get("title", "Cox Model Hazard Ratios"),
                "x_axis": data_settings.get("x_axis", "Hazard Ratio (log scale)"),
                "y_axis": data_settings.get("y_axis", "Features"),
                "confidence_intervals": data_settings.get("confidence_intervals", True),
                "sort_by": data_settings.get("sort_by", "value"),  # value, abs_value, or name
                "top_n_features": data_settings.get("top_n_features", 10)
            }
        elif viz_type == "multi_state_transitions":
            if not results.transition_matrix:
                raise HTTPException(status_code=404, detail="Transition matrix not available for this analysis")
            
            return {
                "transition_matrix": results.transition_matrix,
                "states": results.states,
                "title": data_settings.get("title", "Multi-State Transitions"),
                "plot_type": data_settings.get("plot_type", "sankey"),  # sankey, heatmap, or chord
                "time_points": data_settings.get("time_points", []),
                "color_scheme": data_settings.get("color_scheme", "default")
            }
        elif viz_type == "risk_heatmap":
            if not results.risk_scores:
                raise HTTPException(status_code=404, detail="Risk scores not available for this analysis")
            
            return {
                "risk_scores": results.risk_scores,
                "features": results.selected_features[:2] if len(results.selected_features) > 2 else results.selected_features,
                "title": data_settings.get("title", "Risk Score Heatmap"),
                "color_scheme": data_settings.get("color_scheme", "RdYlGn_r"),
                "plot_type": data_settings.get("plot_type", "heatmap"),  # heatmap or contour
                "risk_thresholds": data_settings.get("risk_thresholds", [0.25, 0.5, 0.75])
            }
        else:
            raise HTTPException(status_code=400, detail=f"Unsupported visualization type: {viz_type}")
    
    async def get_visualization(self, viz_id: str) -> Optional[Visualization]:
        """Get a visualization by ID"""
        return await self.repository.get_visualization(viz_id)
    
    async def delete_visualization(self, viz_id: str) -> bool:
        """Delete a visualization"""
        return await self.repository.delete_visualization(viz_id)
    
    async def update_visualization_config(
        self, 
        visualization_id: str, 
        config: Dict[str, Any]
    ) -> Optional[Visualization]:
        """Update visualization config"""
        # Get the visualization
        visualization = await self.repository.get_visualization(visualization_id)
        if not visualization:
            return None
        
        # Update config
        visualization.config = config
        visualization.updated_at = datetime.now()
        
        # Save to repository
        return await self.repository.update_visualization(visualization_id, {
            "config": config,
            "updated_at": datetime.now()
        })
    
    async def create_share_link(self, visualization_id: str, days_valid: int = 30) -> ShareLink:
        """Generate a shareable link for a visualization"""
        # Get the visualization
        visualization = await self.repository.get_visualization(visualization_id)
        if not visualization:
            raise HTTPException(status_code=404, detail="Visualization not found")
        
        # Generate a unique token if not already shared
        share_token = visualization.share_token
        if not share_token:
            share_token = secrets.token_urlsafe(32)
        
        # Set expiration date
        expires_at = datetime.now() + timedelta(days=days_valid)
        
        # Update the visualization
        updated_viz = await self.repository.update_visualization(visualization_id, {
            "shared": True,
            "share_token": share_token,
            "expires_at": expires_at,
            "updated_at": datetime.now()
        })
        
        # Create share URL
        share_url = f"{settings.API_V1_STR}/shared/visualization/{share_token}"
        
        return ShareLink(
            visualization_id=visualization_id,
            share_token=share_token,
            share_url=share_url,
            created_at=datetime.now(),
            expires_at=expires_at
        )

    async def create_kaplan_meier_curve(
        self,
        model_id: str,
        stratify_by: Optional[str] = None,
        time_points: Optional[List[float]] = None,
        include_confidence_intervals: bool = True,
        include_risk_table: bool = True,
        name: Optional[str] = None,
        description: Optional[str] = None
    ) -> Visualization:
        """
        Create an interactive Kaplan-Meier survival curve visualization
        
        Args:
            model_id: ID of the trained survival model
            stratify_by: Feature name to stratify the KM curves by (optional)
            time_points: Specific time points to evaluate (optional)
            include_confidence_intervals: Whether to include confidence intervals
            include_risk_table: Whether to include a risk table below the plot
            name: Name for the visualization
            description: Description for the visualization
            
        Returns:
            Created visualization object
        """
        if not self.model_repository:
            raise ValueError("Model repository is required for model-based visualizations")
            
        # Get model data
        model_data = await self.model_repository.get_model(model_id)
        if not model_data:
            raise HTTPException(status_code=404, detail="Model not found")
            
        # Get trained model and its analysis results
        analysis_id = model_data.get("analysis_id")
        if not analysis_id:
            raise HTTPException(status_code=400, detail="Model has no associated analysis")
            
        # Create visualization data
        viz_data = VisualizationCreate(
            name=name or f"Kaplan-Meier Curve - {model_data.get('name', 'Model')}",
            description=description or "Interactive Kaplan-Meier survival probability curve",
            type="survival_curve",
            analysis_id=analysis_id,
            config={
                "interactive": True,
                "show_confidence_intervals": include_confidence_intervals,
                "show_risk_table": include_risk_table,
                "time_range": time_points,
                "stratify_by": stratify_by,
                "plot_options": {
                    "grid": True,
                    "legend": True,
                    "tooltip": True,
                    "zoom": True,
                    "download": True
                }
            },
            data_settings={
                "title": f"Survival Probability - {model_data.get('name', 'Model')}",
                "confidence_intervals": include_confidence_intervals,
                "risk_table": include_risk_table,
                "stratification": stratify_by,
                "time_points": time_points
            }
        )
        
        # Create visualization using base method
        return await self.create_visualization(viz_data)
    
    async def create_cox_hazard_ratio_plot(
        self,
        model_id: str,
        top_n_features: int = 10,
        sort_by: str = "value",
        include_confidence_intervals: bool = True,
        name: Optional[str] = None,
        description: Optional[str] = None
    ) -> Visualization:
        """
        Create an interactive Cox model hazard ratio plot
        
        Args:
            model_id: ID of the trained Cox model
            top_n_features: Number of top features to display
            sort_by: How to sort features ('value', 'abs_value', or 'name')
            include_confidence_intervals: Whether to include confidence intervals
            name: Name for the visualization
            description: Description for the visualization
            
        Returns:
            Created visualization object
        """
        if not self.model_repository:
            raise ValueError("Model repository is required for model-based visualizations")
            
        # Get model data
        model_data = await self.model_repository.get_model(model_id)
        if not model_data:
            raise HTTPException(status_code=404, detail="Model not found")
            
        # Verify model type is Cox
        model_type = model_data.get("model_type")
        if model_type != "cox_ph":
            raise HTTPException(status_code=400, detail="Hazard ratio plots are only available for Cox PH models")
            
        # Get associated analysis
        analysis_id = model_data.get("analysis_id")
        if not analysis_id:
            raise HTTPException(status_code=400, detail="Model has no associated analysis")
            
        # Create visualization data
        viz_data = VisualizationCreate(
            name=name or f"Hazard Ratios - {model_data.get('name', 'Cox Model')}",
            description=description or "Interactive Cox model hazard ratio visualization",
            type="hazard_ratios",
            analysis_id=analysis_id,
            config={
                "interactive": True,
                "show_confidence_intervals": include_confidence_intervals,
                "top_n_features": top_n_features,
                "sort_by": sort_by,
                "plot_options": {
                    "grid": True,
                    "legend": False,
                    "tooltip": True,
                    "zoom": True,
                    "download": True
                }
            },
            data_settings={
                "title": f"Hazard Ratios - {model_data.get('name', 'Cox Model')}",
                "confidence_intervals": include_confidence_intervals,
                "sort_by": sort_by,
                "top_n_features": top_n_features
            }
        )
        
        # Create visualization using base method
        return await self.create_visualization(viz_data)
    
    async def create_feature_importance_plot(
        self,
        model_id: str,
        plot_type: str = "bar",
        top_n_features: int = 10,
        color_scheme: str = "default",
        name: Optional[str] = None,
        description: Optional[str] = None
    ) -> Visualization:
        """
        Create an interactive feature importance visualization with explainability
        
        Args:
            model_id: ID of the trained model
            plot_type: Type of plot ('bar', 'shap', or 'lime')
            top_n_features: Number of top features to display
            color_scheme: Color scheme for the plot
            name: Name for the visualization
            description: Description for the visualization
            
        Returns:
            Created visualization object
        """
        if not self.model_repository:
            raise ValueError("Model repository is required for model-based visualizations")
            
        # Get model data
        model_data = await self.model_repository.get_model(model_id)
        if not model_data:
            raise HTTPException(status_code=404, detail="Model not found")
            
        # Get associated analysis
        analysis_id = model_data.get("analysis_id")
        if not analysis_id:
            raise HTTPException(status_code=400, detail="Model has no associated analysis")
            
        # Validate plot type
        valid_plot_types = ["bar", "shap", "lime"]
        if plot_type not in valid_plot_types:
            raise HTTPException(
                status_code=400, 
                detail=f"Invalid plot type. Must be one of: {', '.join(valid_plot_types)}"
            )
            
        # Create visualization data
        viz_data = VisualizationCreate(
            name=name or f"Feature Importance - {model_data.get('name', 'Model')}",
            description=description or f"Interactive feature importance visualization using {plot_type.upper()}",
            type="feature_importance",
            analysis_id=analysis_id,
            config={
                "interactive": True,
                "top_n_features": top_n_features,
                "color_scheme": color_scheme,
                "plot_type": plot_type,
                "plot_options": {
                    "grid": True,
                    "legend": plot_type != "bar",
                    "tooltip": True,
                    "zoom": True,
                    "download": True
                }
            },
            data_settings={
                "title": f"Feature Importance - {model_data.get('name', 'Model')}",
                "plot_type": plot_type,
                "top_n_features": top_n_features,
                "color_scheme": color_scheme
            }
        )
        
        # Create visualization using base method
        return await self.create_visualization(viz_data)
    
    async def create_multi_state_transition_plot(
        self,
        model_id: str,
        plot_type: str = "sankey",
        time_points: Optional[List[float]] = None,
        color_scheme: str = "default",
        name: Optional[str] = None,
        description: Optional[str] = None
    ) -> Visualization:
        """
        Create an interactive multi-state transition visualization
        
        Args:
            model_id: ID of the trained multi-state model
            plot_type: Type of plot ('sankey', 'heatmap', or 'chord')
            time_points: Specific time points to evaluate transitions
            color_scheme: Color scheme for the plot
            name: Name for the visualization
            description: Description for the visualization
            
        Returns:
            Created visualization object
        """
        if not self.model_repository:
            raise ValueError("Model repository is required for model-based visualizations")
            
        # Get model data
        model_data = await self.model_repository.get_model(model_id)
        if not model_data:
            raise HTTPException(status_code=404, detail="Model not found")
            
        # Check if model supports multi-state transitions
        model_type = model_data.get("model_type")
        if model_type not in ["multi_state", "illness_death"]:
            raise HTTPException(
                status_code=400, 
                detail="Multi-state transition plots are only available for multi-state models"
            )
            
        # Get associated analysis
        analysis_id = model_data.get("analysis_id")
        if not analysis_id:
            raise HTTPException(status_code=400, detail="Model has no associated analysis")
            
        # Validate plot type
        valid_plot_types = ["sankey", "heatmap", "chord"]
        if plot_type not in valid_plot_types:
            raise HTTPException(
                status_code=400, 
                detail=f"Invalid plot type. Must be one of: {', '.join(valid_plot_types)}"
            )
            
        # Create visualization data
        viz_data = VisualizationCreate(
            name=name or f"State Transitions - {model_data.get('name', 'Multi-State Model')}",
            description=description or "Interactive multi-state transition visualization",
            type="multi_state_transitions",
            analysis_id=analysis_id,
            config={
                "interactive": True,
                "plot_type": plot_type,
                "time_points": time_points,
                "color_scheme": color_scheme,
                "plot_options": {
                    "tooltip": True,
                    "download": True,
                    "animation": plot_type == "sankey"
                }
            },
            data_settings={
                "title": f"State Transitions - {model_data.get('name', 'Multi-State Model')}",
                "plot_type": plot_type,
                "time_points": time_points,
                "color_scheme": color_scheme
            }
        )
        
        # Create visualization using base method
        return await self.create_visualization(viz_data)
    
    async def create_risk_score_heatmap(
        self,
        model_id: str,
        features: Optional[List[str]] = None,
        plot_type: str = "heatmap",
        color_scheme: str = "RdYlGn_r",
        risk_thresholds: Optional[List[float]] = None,
        name: Optional[str] = None,
        description: Optional[str] = None
    ) -> Visualization:
        """
        Create an interactive risk score heatmap visualization
        
        Args:
            model_id: ID of the trained model
            features: Two feature names to use for the heatmap axes (if None, uses top 2)
            plot_type: Type of plot ('heatmap' or 'contour')
            color_scheme: Color scheme for the plot
            risk_thresholds: Optional thresholds to highlight risk segments
            name: Name for the visualization
            description: Description for the visualization
            
        Returns:
            Created visualization object
        """
        if not self.model_repository:
            raise ValueError("Model repository is required for model-based visualizations")
            
        # Get model data
        model_data = await self.model_repository.get_model(model_id)
        if not model_data:
            raise HTTPException(status_code=404, detail="Model not found")
            
        # Get associated analysis
        analysis_id = model_data.get("analysis_id")
        if not analysis_id:
            raise HTTPException(status_code=400, detail="Model has no associated analysis")
            
        # Validate plot type
        valid_plot_types = ["heatmap", "contour"]
        if plot_type not in valid_plot_types:
            raise HTTPException(
                status_code=400, 
                detail=f"Invalid plot type. Must be one of: {', '.join(valid_plot_types)}"
            )
            
        # Default thresholds if not provided
        if risk_thresholds is None:
            risk_thresholds = [0.25, 0.5, 0.75]
            
        # Create visualization data
        viz_data = VisualizationCreate(
            name=name or f"Risk Score Heatmap - {model_data.get('name', 'Model')}",
            description=description or "Interactive risk score heatmap visualization",
            type="risk_heatmap",
            analysis_id=analysis_id,
            config={
                "interactive": True,
                "plot_type": plot_type,
                "color_scheme": color_scheme,
                "risk_thresholds": risk_thresholds,
                "selected_features": features,
                "plot_options": {
                    "grid": True,
                    "tooltip": True,
                    "zoom": True,
                    "download": True,
                    "colorbar": True
                }
            },
            data_settings={
                "title": f"Risk Score Heatmap - {model_data.get('name', 'Model')}",
                "plot_type": plot_type,
                "color_scheme": color_scheme,
                "risk_thresholds": risk_thresholds
            }
        )
        
        # Create visualization using base method
        return await self.create_visualization(viz_data)
