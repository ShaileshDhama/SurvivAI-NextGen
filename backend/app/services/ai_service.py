"""
AI Service - Handles AI-powered insights generation and natural language explanations.
"""

import os
import json
import logging
from typing import Dict, List, Any, Optional

import numpy as np
import pandas as pd
from fastapi import HTTPException
from langchain.llms import OpenAI
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain.output_parsers import PydanticOutputParser

from app.core.config import settings
from app.ml.utils.explainability import extract_feature_importance
from app.schemas.report import (
    ReportType, ReportFormat, ReportConfig, 
    SurvivalSummaryReport, RiskStratificationReport, SurvivalTrendReport
)

logger = logging.getLogger(__name__)

class AIService:
    """Service for generating AI-powered insights and explanations."""
    
    def __init__(self):
        """Initialize the AI service with language models and templates."""
        # Initialize language model
        self.llm = OpenAI(
            temperature=0.2,
            model_name="gpt-4",
            api_key=settings.OPENAI_API_KEY
        )
        
        # Load prompt templates
        self.templates = self._load_prompt_templates()
        
        # Initialize chains
        self.insight_chain = LLMChain(
            llm=self.llm,
            prompt=self.templates["report_insights"]
        )
        
        self.explanation_chain = LLMChain(
            llm=self.llm,
            prompt=self.templates["feature_explanation"]
        )
        
        self.chatbot_chain = LLMChain(
            llm=self.llm,
            prompt=self.templates["chatbot_response"]
        )
    
    def _load_prompt_templates(self) -> Dict[str, PromptTemplate]:
        """Load prompt templates for different AI tasks."""
        templates_dir = os.path.join(settings.TEMPLATES_DIR, "prompts")
        templates = {}
        
        # Report insights template
        with open(os.path.join(templates_dir, "report_insights.txt"), "r") as f:
            templates["report_insights"] = PromptTemplate(
                template=f.read(),
                input_variables=["model_data", "dataset", "insight_type", "context"]
            )
        
        # Feature explanation template
        with open(os.path.join(templates_dir, "feature_explanation.txt"), "r") as f:
            templates["feature_explanation"] = PromptTemplate(
                template=f.read(),
                input_variables=["feature_name", "feature_value", "importance", "model_type", "context"]
            )
        
        # Chatbot response template
        with open(os.path.join(templates_dir, "chatbot_response.txt"), "r") as f:
            templates["chatbot_response"] = PromptTemplate(
                template=f.read(),
                input_variables=["user_question", "report_data", "model_data", "context"]
            )
        
        return templates
    
    async def generate_report_insights(self, context: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Generate AI-powered insights for reports based on context data.
        
        Args:
            context: Dictionary containing model data, dataset info, and other relevant context
            
        Returns:
            List of insights with title, description, importance, and category
        """
        try:
            # Extract key data from context
            model_data = context.get("model_data", {})
            dataset = context.get("dataset", {})
            insight_type = context.get("insight_type", "general")
            
            # Remove large data to prevent token limit issues
            context_copy = context.copy()
            for key in ["model_data", "dataset"]:
                if key in context_copy:
                    del context_copy[key]
            
            # Run the insight generation chain
            result = await self.insight_chain.arun(
                model_data=json.dumps(model_data, default=str),
                dataset=json.dumps(dataset, default=str),
                insight_type=insight_type,
                context=json.dumps(context_copy, default=str)
            )
            
            # Parse the result into structured insights
            try:
                insights = json.loads(result)
                if not isinstance(insights, list):
                    insights = [insights]
                
                # Validate and clean insights
                cleaned_insights = []
                for insight in insights:
                    if isinstance(insight, dict) and "title" in insight and "description" in insight:
                        cleaned_insight = {
                            "title": insight.get("title", ""),
                            "description": insight.get("description", ""),
                            "importance": insight.get("importance", "medium"),
                            "category": insight.get("category", "general"),
                            "related_features": insight.get("related_features", []),
                            "confidence": insight.get("confidence", 0.8),
                        }
                        cleaned_insights.append(cleaned_insight)
                
                return cleaned_insights
            except json.JSONDecodeError:
                # Fallback for parsing errors
                logger.error(f"Failed to parse AI insights: {result}")
                return [{
                    "title": "Analysis Summary",
                    "description": result[:500],  # Truncate to avoid very long texts
                    "importance": "medium",
                    "category": "general"
                }]
        
        except Exception as e:
            logger.error(f"Error generating AI insights: {str(e)}")
            return [{
                "title": "Automated Analysis",
                "description": "An automated analysis of the model and data shows patterns typical for this type of survival analysis.",
                "importance": "medium",
                "category": "general"
            }]
    
    async def explain_feature(
        self, 
        feature_name: str, 
        feature_value: Any, 
        model_data: Dict[str, Any],
        context: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Generate a natural language explanation of a feature's impact on the model.
        
        Args:
            feature_name: Name of the feature to explain
            feature_value: Value of the feature
            model_data: Model metadata including feature importance
            context: Additional context for the explanation
            
        Returns:
            Natural language explanation of the feature's impact
        """
        try:
            # Extract feature importance if available
            importance = "unknown"
            if "feature_importance" in model_data and feature_name in model_data["feature_importance"]:
                importance = model_data["feature_importance"][feature_name]
            
            # Run the explanation chain
            explanation = await self.explanation_chain.arun(
                feature_name=feature_name,
                feature_value=str(feature_value),
                importance=str(importance),
                model_type=model_data.get("type", "Unknown"),
                context=json.dumps(context or {}, default=str)
            )
            
            return explanation
        
        except Exception as e:
            logger.error(f"Error explaining feature: {str(e)}")
            return f"The feature '{feature_name}' with value '{feature_value}' has an impact on the model's predictions. Higher values typically indicate increased risk."
    
    async def generate_chatbot_response(
        self, 
        user_question: str,
        report_data: Dict[str, Any],
        model_data: Dict[str, Any],
        context: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Generate a conversational AI response to user questions about the report.
        
        Args:
            user_question: Question from the user
            report_data: Data from the generated report
            model_data: Model metadata
            context: Additional context for the response
            
        Returns:
            Natural language response to the user's question
        """
        try:
            # Run the chatbot response chain
            response = await self.chatbot_chain.arun(
                user_question=user_question,
                report_data=json.dumps(report_data, default=str),
                model_data=json.dumps(model_data, default=str),
                context=json.dumps(context or {}, default=str)
            )
            
            return response
        
        except Exception as e:
            logger.error(f"Error generating chatbot response: {str(e)}")
            return f"I understand you're asking about '{user_question}'. The report contains analysis of the survival model and risk factors. Could you clarify what specific aspect you'd like me to explain?"

    async def answer_question(
        self, 
        question: str, 
        model_id: Optional[str] = None, 
        report: Optional[Any] = None
    ) -> str:
        """
        Answer a question about a survival analysis model or report using AI.
        
        Args:
            question: The user's question about the model or report
            model_id: Optional ID of the model to reference
            report: Optional report object to reference
            
        Returns:
            str: AI-generated answer to the question
        """
        logger.info(f"Generating answer for question: {question}")
        
        # Gather context about the model and/or report
        context = []
        if model_id:
            context.append(f"Model ID: {model_id}")
            # Here we would normally fetch more details about the model
            # such as features, performance metrics, and other attributes
        
        if report:
            context.append(f"Report Type: {report.report_type}")
            context.append(f"Report Created: {report.created_at}")
            # Add additional report details to context
            
        # Create the prompt template
        prompt_template = PromptTemplate(
            input_variables=["question", "context"],
            template="""You are an AI assistant specializing in survival analysis and medical statistics. 
            Your role is to help users understand their survival analysis models and reports.
            
            Context information about the model and report:
            {context}
            
            User question: {question}
            
            Provide a clear, informative answer that explains concepts in an accessible way.
            Focus on accuracy and helpfulness. If you don't have enough information to answer confidently,
            acknowledge the limitations and suggest what additional information would be helpful.
            
            Answer:"""
        )
        
        # Create and run the chain
        chain = LLMChain(llm=self.llm, prompt=prompt_template)
        response = await chain.arun(
            question=question,
            context="\n".join(context) if context else "No specific context available."
        )
        
        return response.strip()
