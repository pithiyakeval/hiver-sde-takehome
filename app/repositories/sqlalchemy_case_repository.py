from datetime import datetime, timezone
from typing import List

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.db.models import CaseModel, CaseRetrievalModel
from app.models.case import CaseStatus, SupportCase
from app.repositories.case_repository import CaseRepository
from src.escalation.policy import EscalationDecision
from src.generation.models import RetrievedExample
from src.pipeline.agent import AgentResult, AgentTiming


class SqlAlchemyCaseRepository(CaseRepository):
    """
    SQLAlchemy implementation of the support-case repository.

    This repository is transaction-neutral.

    It never commits or rolls back the database transaction. The owning
    UnitOfWork controls transaction boundaries.
    """

    def __init__(self, session: Session) -> None:
        self._session = session

    def save(self, case: SupportCase) -> SupportCase:
        """
        Persist a new case within the current transaction.

        The transaction is not committed here.
        """

        case_model = self._to_model(case)

        self._session.add(case_model)
        self._save_retrievals(case)

        self._session.flush()

        return case

    def get(self, case_id: str) -> SupportCase | None:
        statement = select(CaseModel).where(
            CaseModel.case_id == case_id
        )

        case_model = self._session.scalar(statement)

        if case_model is None:
            return None

        retrievals = self._get_retrievals(case_id)

        return self._to_domain(
            case_model,
            retrievals,
        )

    def list(self) -> List[SupportCase]:
        statement = (
            select(CaseModel)
            .order_by(CaseModel.created_at.desc())
        )

        case_models = self._session.scalars(statement).all()

        return [
            self._to_domain(
                case_model,
                self._get_retrievals(case_model.case_id),
            )
            for case_model in case_models
        ]

    def update(self, case: SupportCase) -> SupportCase:
        """
        Update an existing case within the current transaction.

        The transaction is not committed here.
        """

        statement = select(CaseModel).where(
            CaseModel.case_id == case.case_id
        )

        case_model = self._session.scalar(statement)

        if case_model is None:
            raise KeyError(
                f"Case not found: {case.case_id}"
            )

        self._update_model(
            case_model,
            case,
        )

        self._session.flush()

        return case

    def _get_retrievals(
        self,
        case_id: str,
    ) -> List[CaseRetrievalModel]:
        statement = (
            select(CaseRetrievalModel)
            .where(
                CaseRetrievalModel.case_id == case_id
            )
            .order_by(
                CaseRetrievalModel.rank.asc()
            )
        )

        return list(
            self._session.scalars(statement).all()
        )

    def _save_retrievals(
        self,
        case: SupportCase,
    ) -> None:
        for rank, example in enumerate(
            case.agent_result.retrieved_examples,
            start=1,
        ):
            retrieval = CaseRetrievalModel(
                case_id=case.case_id,
                customer_text=example.customer_text,
                historical_response=example.historical_response,
                similarity=example.similarity,
                rank=rank,
                created_at=datetime.now(timezone.utc),
            )

            self._session.add(retrieval)

    @staticmethod
    def _to_model(
        case: SupportCase,
    ) -> CaseModel:
        result = case.agent_result

        return CaseModel(
            case_id=case.case_id,
            customer_message=case.customer_message,
            status=case.status.value,
            intent=result.intent,
            intent_confidence=result.intent_confidence,
            classification_reason=result.classification_reason,
            draft_response=result.draft_response,
            generation_model=result.generation_model,
            grounded=result.grounded,
            should_escalate=result.escalation.should_escalate,
            escalation_reason=result.escalation.reason,
            classification_ms=result.timing.classification_ms,
            retrieval_ms=result.timing.retrieval_ms,
            generation_ms=result.timing.generation_ms,
            escalation_ms=result.timing.escalation_ms,
            total_ms=result.timing.total_ms,
            created_at=case.created_at,
            updated_at=case.updated_at,
        )

    @staticmethod
    def _update_model(
        model: CaseModel,
        case: SupportCase,
    ) -> None:
        model.status = case.status.value
        model.updated_at = case.updated_at

    @staticmethod
    def _to_domain(
        model: CaseModel,
        retrievals: List[CaseRetrievalModel],
    ) -> SupportCase:
        retrieved_examples = [
            RetrievedExample(
                customer_text=retrieval.customer_text,
                historical_response=retrieval.historical_response,
                similarity=retrieval.similarity,
            )
            for retrieval in retrievals
        ]

        escalation = EscalationDecision(
            should_escalate=model.should_escalate,
            reason=model.escalation_reason,
        )

        timing = AgentTiming(
            classification_ms=model.classification_ms,
            retrieval_ms=model.retrieval_ms,
            generation_ms=model.generation_ms,
            escalation_ms=model.escalation_ms,
            total_ms=model.total_ms,
        )

        agent_result = AgentResult(
            customer_message=model.customer_message,
            intent=model.intent,
            intent_confidence=model.intent_confidence,
            classification_reason=model.classification_reason,
            retrieved_examples=retrieved_examples,
            draft_response=model.draft_response,
            generation_model=model.generation_model,
            grounded=model.grounded,
            escalation=escalation,
            timing=timing,
        )

        return SupportCase(
            case_id=model.case_id,
            customer_message=model.customer_message,
            status=CaseStatus(model.status),
            created_at=model.created_at,
            updated_at=model.updated_at,
            agent_result=agent_result,
        )