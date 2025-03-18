from typing import Dict, Any, Optional, List
import logging
import os
import tempfile
from datetime import datetime
from weasyprint import HTML, CSS
from jinja2 import Environment, FileSystemLoader
from app.core.config import settings
from app.schemas.report import (
    ReportType, ReportFormat, SurvivalSummaryReport, 
    RiskStratificationReport, SurvivalTrendReport
)

logger = logging.getLogger(__name__)

class PDFService:
    """
    Service for generating PDF reports using Weasyprint and Jinja2 templating.
    """
    
    def __init__(self):
        # Set up the Jinja2 template environment
        template_dir = os.path.join(settings.BASE_DIR, "app", "templates", "reports")
        self.template_env = Environment(
            loader=FileSystemLoader(template_dir),
            autoescape=True
        )
        
        # Set up CSS for PDF styling
        self.css = CSS(string="""
            @page {
                size: letter;
                margin: 2.5cm 1.5cm;
                @top-center {
                    content: "SurvivAI Report";
                    font-size: 9pt;
                    color: #666;
                }
                @bottom-center {
                    content: "Page " counter(page) " of " counter(pages);
                    font-size: 9pt;
                    color: #666;
                }
            }
            body {
                font-family: 'Helvetica', 'Arial', sans-serif;
                font-size: 10pt;
                line-height: 1.6;
                color: #333;
            }
            h1 {
                font-size: 18pt;
                color: #2c3e50;
                margin: 0.5cm 0;
                padding-bottom: 0.2cm;
                border-bottom: 1pt solid #ddd;
            }
            h2 {
                font-size: 14pt;
                color: #2980b9;
                margin: 0.3cm 0;
            }
            h3 {
                font-size: 12pt;
                color: #3498db;
                margin: 0.2cm 0;
            }
            table {
                width: 100%;
                border-collapse: collapse;
                margin: 0.5cm 0;
            }
            th {
                background-color: #f5f5f5;
                border: 1pt solid #ddd;
                padding: 0.2cm;
                text-align: left;
                font-weight: bold;
            }
            td {
                border: 1pt solid #ddd;
                padding: 0.2cm;
            }
            .risk-high {
                color: #e74c3c;
                font-weight: bold;
            }
            .risk-medium {
                color: #f39c12;
                font-weight: bold;
            }
            .risk-low {
                color: #27ae60;
                font-weight: bold;
            }
            .footnote {
                font-size: 8pt;
                color: #666;
                margin-top: 1cm;
                border-top: 1pt solid #ddd;
                padding-top: 0.2cm;
            }
            .chart-container {
                margin: 0.5cm 0;
                page-break-inside: avoid;
            }
            .header-logo {
                width: 120px;
                float: right;
                margin-top: -15px;
            }
            .report-metadata {
                margin: 0.3cm 0;
                font-size: 9pt;
                color: #666;
            }
            .insights-box {
                background-color: #f8f9fa;
                border: 1pt solid #e9ecef;
                border-left: 4pt solid #3498db;
                padding: 0.3cm;
                margin: 0.3cm 0;
            }
        """)

    async def generate_pdf(self, report_data: Dict[str, Any], report_type: ReportType) -> bytes:
        """
        Generate a PDF report based on the report type and data.
        
        Args:
            report_data: The data to include in the report
            report_type: The type of report to generate
            
        Returns:
            bytes: The generated PDF document as bytes
        """
        try:
            logger.info(f"Generating PDF report for type: {report_type}")
            
            # Select the appropriate template based on report type
            template_name = self._get_template_name(report_type)
            template = self.template_env.get_template(template_name)
            
            # Add generation timestamp to report data
            report_data["generated_at"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            report_data["report_type"] = report_type.value
            
            # Render the HTML template with the report data
            html_content = template.render(**report_data)
            
            # Generate PDF from the HTML content
            with tempfile.NamedTemporaryFile(suffix='.pdf', delete=False) as temp_file:
                HTML(string=html_content).write_pdf(
                    target=temp_file.name,
                    stylesheets=[self.css]
                )
                
                # Read the generated PDF file
                with open(temp_file.name, 'rb') as pdf_file:
                    pdf_content = pdf_file.read()
                
                # Clean up the temporary file
                os.unlink(temp_file.name)
                
                return pdf_content
                
        except Exception as e:
            logger.error(f"Error generating PDF report: {str(e)}")
            raise Exception(f"Failed to generate PDF report: {str(e)}")
    
    def _get_template_name(self, report_type: ReportType) -> str:
        """
        Get the appropriate template filename based on report type.
        
        Args:
            report_type: The type of report
            
        Returns:
            str: Template filename
        """
        template_mapping = {
            ReportType.SURVIVAL_SUMMARY: "survival_summary.html",
            ReportType.RISK_STRATIFICATION: "risk_stratification.html", 
            ReportType.SURVIVAL_TREND: "survival_trend.html"
        }
        
        return template_mapping.get(report_type, "base_report.html")
