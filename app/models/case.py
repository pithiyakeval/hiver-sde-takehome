from dataclasses import dataclass
from datetime import datetime
from enum import Enum

from src.pipeline.agent import AgentResult


class CaseStatus(str, Enum):
    ANALYZING = "ANALYZING"
    READY_FOR_REVIEW = "READY_FOR_REVIEW"
    AUTO_HANDLED = "AUTO_HANDLED"
    ESCALATED = "ESCALATED"
    RESOLVED = "RESOLVED"
    FAILED = "FAILED"


@dataclass
class SupportCase:
    case_id: str
    customer_message: str
    status: CaseStatus
    created_at: datetime
    updated_at: datetime
    agent_result: AgentResult