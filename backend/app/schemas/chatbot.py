from pydantic import BaseModel, Field
from typing import List, Optional

class ChatbotQuestionRequest(BaseModel):
    """
    Schema for chatbot question requests.
    """
    question: str = Field(..., description="The question being asked about reports, models, or survival analysis")
    model_id: Optional[str] = Field(None, description="ID of the model to reference when answering the question")
    report_id: Optional[str] = Field(None, description="ID of the specific report to reference when answering the question")

class ChatbotResponse(BaseModel):
    """
    Schema for chatbot responses.
    """
    answer: str = Field(..., description="The AI-generated answer to the question")
    sources: List[str] = Field(default_factory=list, description="Sources of information used to generate the answer")
    suggested_follow_ups: Optional[List[str]] = Field(None, description="Suggested follow-up questions the user might ask")
