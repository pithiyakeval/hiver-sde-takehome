from abc import ABC, abstractmethod
from typing import Optional

from app.models.case import SupportCase


class CaseRepository(ABC):
    """Abstract storage contract for support cases."""

    @abstractmethod
    def save(self, case: SupportCase) -> SupportCase:
        """Persist a support case."""
        raise NotImplementedError

    @abstractmethod
    def get(self, case_id: str) -> Optional[SupportCase]:
        """Return a case by ID, or None when it does not exist."""
        raise NotImplementedError

    @abstractmethod
    def list(self) -> list[SupportCase]:
        """Return all stored cases."""
        raise NotImplementedError

    @abstractmethod
    def update(self, case: SupportCase) -> SupportCase:
        """Update an existing case."""
        raise NotImplementedError