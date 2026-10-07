from sqlalchemy.orm import Session

from app.repositories.sqlalchemy_case_event_repository import (
    SqlAlchemyCaseEventRepository,
)
from app.repositories.sqlalchemy_case_repository import (
    SqlAlchemyCaseRepository,
)
from app.repositories.unit_of_work import UnitOfWork


class SqlAlchemyUnitOfWork(UnitOfWork):
    """
    SQLAlchemy-backed unit of work.

    All repositories share the same SQLAlchemy session, which gives us
    a single transaction boundary for related operations.
    """

    def __init__(self, session: Session) -> None:
        self._session = session

        self.cases = SqlAlchemyCaseRepository(session)
        self.case_events = SqlAlchemyCaseEventRepository(session)

    def commit(self) -> None:
        self._session.commit()

    def rollback(self) -> None:
        self._session.rollback()