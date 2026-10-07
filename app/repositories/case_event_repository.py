from abc import ABC, abstractmethod

from app.models.case_event import CaseEvent


class CaseEventRepository(ABC):
    """
    Abstract persistence contract for support-case audit events.
    """

    @abstractmethod
    def save(self, event: CaseEvent) -> CaseEvent:
        """Persist an audit event."""
        raise NotImplementedError

    @abstractmethod
    def list_by_case(self, case_id: str) -> list[CaseEvent]:
        """Return audit events for a case in chronological order."""
        raise NotImplementedError