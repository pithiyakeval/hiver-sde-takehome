from abc import ABC, abstractmethod

from app.repositories.case_event_repository import CaseEventRepository
from app.repositories.case_repository import CaseRepository


class UnitOfWork(ABC):
    """
    Transaction boundary for application persistence.

    Repositories participate in the same transaction. The application
    service decides when the transaction is committed or rolled back.
    """

    cases: CaseRepository
    case_events: CaseEventRepository

    @abstractmethod
    def commit(self) -> None:
        """Commit the current transaction."""
        raise NotImplementedError

    @abstractmethod
    def rollback(self) -> None:
        """Roll back the current transaction."""
        raise NotImplementedError