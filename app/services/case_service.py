from __future__ import annotations

from datetime import datetime, timezone
from typing import ClassVar
from uuid import uuid4

from app.api.errors import (
    CaseNotFoundError,
    InvalidCaseStatusTransitionError,
)
from app.models.case import CaseStatus, SupportCase
from app.models.case_event import CaseEvent, CaseEventType
from app.repositories.unit_of_work import UnitOfWork
from src.pipeline.agent import AgentResult


class CaseService:
    """
    Application service responsible for the support case lifecycle.

    The service owns transaction boundaries for case lifecycle operations.
    Case changes and their corresponding audit events are committed as
    one atomic transaction.
    """

    ALLOWED_TRANSITIONS: ClassVar[
        dict[CaseStatus, frozenset[CaseStatus]]
    ] = {
        CaseStatus.ANALYZING: frozenset(
            {
                CaseStatus.READY_FOR_REVIEW,
                CaseStatus.FAILED,
            }
        ),
        CaseStatus.READY_FOR_REVIEW: frozenset(
            {
                CaseStatus.AUTO_HANDLED,
                CaseStatus.ESCALATED,
            }
        ),
        CaseStatus.AUTO_HANDLED: frozenset(
            {
                CaseStatus.RESOLVED,
            }
        ),
        CaseStatus.ESCALATED: frozenset(
            {
                CaseStatus.RESOLVED,
            }
        ),
        CaseStatus.RESOLVED: frozenset(),
        CaseStatus.FAILED: frozenset(),
    }

    def __init__(self, unit_of_work: UnitOfWork) -> None:
        self._unit_of_work = unit_of_work

    # ------------------------------------------------------------------
    # Creation
    # ------------------------------------------------------------------

    def create_from_agent_result(
        self,
        customer_message: str,
        result: AgentResult,
    ) -> SupportCase:
        """
        Create a support case and its audit event atomically.
        """

        now = datetime.now(timezone.utc)

        status = (
            CaseStatus.ESCALATED
            if result.escalation.should_escalate
            else CaseStatus.READY_FOR_REVIEW
        )

        case = SupportCase(
            case_id=self._generate_case_id(),
            customer_message=customer_message,
            status=status,
            created_at=now,
            updated_at=now,
            agent_result=result,
        )

        try:
            self._unit_of_work.cases.save(case)

            self._unit_of_work.case_events.save(
                CaseEvent(
                    case_id=case.case_id,
                    event_type=CaseEventType.CASE_CREATED,
                    actor="system",
                    created_at=now,
                    metadata={
                        "initial_status": status.value,
                        "intent": result.intent,
                        "should_escalate": (
                            result.escalation.should_escalate
                        ),
                    },
                )
            )

            self._unit_of_work.commit()

            return case

        except Exception:
            self._unit_of_work.rollback()
            raise

    # ------------------------------------------------------------------
    # Retrieval
    # ------------------------------------------------------------------

    def get_case(self, case_id: str) -> SupportCase:
        """
        Retrieve a support case by ID.

        Raises:
            CaseNotFoundError:
                If the case does not exist.
        """

        case = self._unit_of_work.cases.get(case_id)

        if case is None:
            raise CaseNotFoundError(case_id)

        return case

    def list_cases(self) -> list[SupportCase]:
        """
        Return all support cases ordered newest first.
        """

        cases = self._unit_of_work.cases.list()

        return sorted(
            cases,
            key=lambda case: case.created_at,
            reverse=True,
        )

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def update_status(
        self,
        case_id: str,
        status: CaseStatus,
        *,
        actor: str = "agent",
    ) -> SupportCase:
        """
        Transition a case and record the lifecycle event atomically.

        The status update and audit event share the same database
        transaction. If either operation fails, both are rolled back.
        """

        case = self.get_case(case_id)

        self._validate_status_transition(
            case=case,
            requested_status=status,
        )

        previous_status = case.status
        now = datetime.now(timezone.utc)

        case.status = status
        case.updated_at = now

        event_type = self._event_type_for_status(status)

        try:
            self._unit_of_work.cases.update(case)

            self._unit_of_work.case_events.save(
                CaseEvent(
                    case_id=case.case_id,
                    event_type=event_type,
                    actor=actor,
                    created_at=now,
                    metadata={
                        "previous_status": previous_status.value,
                        "new_status": status.value,
                    },
                )
            )

            self._unit_of_work.commit()

            return case

        except Exception:
            self._unit_of_work.rollback()
            raise

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------

    @classmethod
    def _validate_status_transition(
        cls,
        *,
        case: SupportCase,
        requested_status: CaseStatus,
    ) -> None:
        allowed_statuses = cls.ALLOWED_TRANSITIONS.get(
            case.status,
            frozenset(),
        )

        if requested_status in allowed_statuses:
            return

        raise InvalidCaseStatusTransitionError(
            case_id=case.case_id,
            current_status=case.status.value,
            requested_status=requested_status.value,
        )

    # ------------------------------------------------------------------
    # Event mapping
    # ------------------------------------------------------------------

    @staticmethod
    def _event_type_for_status(
        status: CaseStatus,
    ) -> CaseEventType:
        mapping = {
            CaseStatus.AUTO_HANDLED: CaseEventType.CASE_APPROVED,
            CaseStatus.ESCALATED: CaseEventType.CASE_ESCALATED,
            CaseStatus.RESOLVED: CaseEventType.CASE_RESOLVED,
            CaseStatus.FAILED: CaseEventType.CASE_FAILED,
        }

        try:
            return mapping[status]
        except KeyError as exc:
            raise ValueError(
                f"No audit event is defined for status '{status.value}'."
            ) from exc

    # ------------------------------------------------------------------
    # Case ID
    # ------------------------------------------------------------------

    @staticmethod
    def _generate_case_id() -> str:
        """Generate a human-readable unique case identifier."""

        return f"CASE-{uuid4().hex[:8].upper()}"