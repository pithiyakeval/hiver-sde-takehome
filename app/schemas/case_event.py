from datetime import datetime

from pydantic import BaseModel, ConfigDict

from app.models.case_event import CaseEventType


class CaseEventResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    event_type: CaseEventType
    actor: str
    created_at: datetime
    metadata: dict | None = None