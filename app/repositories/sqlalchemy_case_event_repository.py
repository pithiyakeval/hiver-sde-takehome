from datetime import timezone

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.db.models import CaseEventModel
from app.models.case_event import CaseEvent,CaseEventType
from app.repositories.case_event_repository import CaseEventRepository


class SqlAlchemyCaseEventRepository(CaseEventRepository):
    """
    SQLAlchemy implementation of the case-event repository.

    This repository participates in the caller's transaction. It does
    not commit or roll back the session.
    """

    def __init__(self, session: Session) -> None:
        self._session = session

    def save(self, event: CaseEvent) -> CaseEvent:
        event_model = CaseEventModel(
            case_id=event.case_id,
            event_type=event.event_type.value,
            actor=event.actor,
            event_metadata=event.metadata,
            created_at=event.created_at,
        )

        self._session.add(event_model)
        self._session.flush()

        return self._to_domain(event_model)

    def list_by_case(
        self,
        case_id: str,
    ) -> list[CaseEvent]:
        statement = (
            select(CaseEventModel)
            .where(
                CaseEventModel.case_id == case_id
            )
            .order_by(
                CaseEventModel.created_at.asc()
            )
        )

        event_models = self._session.scalars(
            statement
        ).all()

        return [
            self._to_domain(event_model)
            for event_model in event_models
        ]

    @staticmethod
    def _to_domain(
        event_model: CaseEventModel,
    ) -> CaseEvent:
        created_at = event_model.created_at

        if created_at.tzinfo is None:
            created_at = created_at.replace(
                tzinfo=timezone.utc
            )

        return CaseEvent(
            case_id=event_model.case_id,
            event_type=event_model.event_type,
            actor=event_model.actor,
            created_at=created_at,
            metadata=event_model.event_metadata,
        )