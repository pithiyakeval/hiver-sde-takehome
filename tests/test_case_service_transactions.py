from datetime import datetime, timezone
from types import SimpleNamespace

import pytest

from app.models.case import CaseStatus, SupportCase
from app.models.case_event import CaseEvent
from app.repositories.case_event_repository import CaseEventRepository
from app.repositories.case_repository import CaseRepository
from app.repositories.unit_of_work import UnitOfWork
from app.services.case_service import CaseService


class FailingEventRepository(CaseEventRepository):
    def save(self, event: CaseEvent) -> CaseEvent:
        raise RuntimeError("audit write failed")

    def list_by_case(self, case_id: str) -> list[CaseEvent]:
        return []


class FakeCaseRepository(CaseRepository):
    def __init__(self, case: SupportCase) -> None:
        self.case = case

    def save(self, case: SupportCase) -> SupportCase:
        self.case = case
        return case

    def get(self, case_id: str) -> SupportCase | None:
        if case_id != self.case.case_id:
            return None
        return self.case

    def list(self) -> list[SupportCase]:
        return [self.case]

    def update(self, case: SupportCase) -> SupportCase:
        self.case = case
        return case


class FakeUnitOfWork(UnitOfWork):
    def __init__(self, case: SupportCase) -> None:
        self.cases = FakeCaseRepository(case)
        self.case_events = FailingEventRepository()
        self.rollback_called = False
        self.commit_called = False

    def commit(self) -> None:
        self.commit_called = True

    def rollback(self) -> None:
        self.rollback_called = True


def build_case() -> SupportCase:
    now = datetime.now(timezone.utc)

    agent_result = SimpleNamespace(
        intent="order_status",
        intent_confidence=0.95,
        classification_reason="test",
        retrieved_examples=[],
        draft_response="Check your order status.",
        generation_model="test",
        grounded=True,
        escalation=SimpleNamespace(
            should_escalate=False,
            reason="test",
        ),
        timing=SimpleNamespace(
            classification_ms=1.0,
            retrieval_ms=1.0,
            generation_ms=1.0,
            escalation_ms=1.0,
            total_ms=4.0,
        ),
    )

    return SupportCase(
        case_id="CASE-ROLLBACK",
        customer_message="Where is my order?",
        status=CaseStatus.AUTO_HANDLED,
        created_at=now,
        updated_at=now,
        agent_result=agent_result,
    )


def test_status_change_rolls_back_when_audit_write_fails():
    case = build_case()

    unit_of_work = FakeUnitOfWork(case)
    service = CaseService(unit_of_work)

    with pytest.raises(RuntimeError, match="audit write failed"):
        service.update_status(
            case_id=case.case_id,
            status=CaseStatus.RESOLVED,
        )

    assert unit_of_work.rollback_called is True
    assert unit_of_work.commit_called is False