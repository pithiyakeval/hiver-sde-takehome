from app.models.case_event import CaseEvent
from app.repositories.unit_of_work import UnitOfWork


class CaseEventService:
    def __init__(self, unit_of_work: UnitOfWork) -> None:
        self._unit_of_work = unit_of_work

    def list_events(self, case_id: str) -> list[CaseEvent]:
        return self._unit_of_work.case_events.list_by_case(case_id)