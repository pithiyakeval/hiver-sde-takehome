from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import Any


class CaseEventType(str, Enum):
    """Supported support-case audit events."""

    CASE_CREATED = "CASE_CREATED"
    CASE_APPROVED = "CASE_APPROVED"
    CASE_ESCALATED = "CASE_ESCALATED"
    CASE_RESOLVED = "CASE_RESOLVED"
    CASE_FAILED = "CASE_FAILED"


@dataclass(frozen=True)
class CaseEvent:
    """
    Immutable domain representation of a support-case audit event.
    """

    case_id: str
    event_type: CaseEventType
    actor: str
    created_at: datetime
    metadata: dict[str, Any] | None = None