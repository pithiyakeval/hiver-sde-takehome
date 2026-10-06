from typing import Literal
from pydantic import BaseModel,Field
from app.models.case import CaseStatus

class SupportAnalyzeRequest(BaseModel):
    customer_message: str = Field(
        ...,
        min_length=1,
        max_length=5000,
        description="Customer's support message.",
    )


class RetrievedCase(BaseModel):
    customer_text: str
    historical_response: str
    similarity: float = Field(ge=0.0, le=1.0)


class EscalationResult(BaseModel):
    should_escalate: bool
    reason: str

class PerformanceMetrics(BaseModel):
    classification_ms: float
    retrieval_ms: float
    generation_ms: float
    escalation_ms: float
    total_ms: float

class SupportAnalyzeResponse(BaseModel):
    customer_message: str
    intent: str
    intent_confidence: float = Field(ge=0.0, le=1.0)
    classification_reason: str
    case_id: str
    status: CaseStatus
    retrieved_cases: list[RetrievedCase]
    performance: PerformanceMetrics
    draft_response: str
    generation_model: str
    grounded: bool
    escalation: EscalationResult

class SupportCaseListItem(BaseModel):
    case_id: str
    status: CaseStatus
    customer_message: str
    intent: str
    intent_confidence: float = Field(ge=0.0, le=1.0)
    draft_response: str
    generation_model: str
    grounded: bool
    escalation: EscalationResult
    performance: PerformanceMetrics