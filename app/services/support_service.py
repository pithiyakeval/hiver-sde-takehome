import logging

from app.models.case import CaseStatus, SupportCase
from app.observability.metrics import (
    CLASSIFICATION_DURATION,
    ESCALATION_DURATION,
    GENERATION_DURATION,
    RETRIEVAL_DURATION,
    SUPPORT_ANALYSES_FAILED_TOTAL,
    SUPPORT_ANALYSES_TOTAL,
    SUPPORT_ANALYSIS_DURATION,
    SUPPORT_ESCALATIONS_TOTAL,
    SUPPORT_LOW_CONFIDENCE_TOTAL,
)
from app.schemas.support import (
    EscalationResult,
    PerformanceMetrics,
    RetrievedCase,
    SupportAnalyzeRequest,
    SupportAnalyzeResponse,
    SupportCaseListItem
)


from app.services.case_service import CaseService

from src.pipeline.agent import AgentResult, SupportAgent


logger = logging.getLogger("app.support")


class SupportService:
    """
    Application service responsible for orchestrating the AI support workflow.

    Responsibilities:
    - Execute the AI support agent.
    - Record application and pipeline metrics.
    - Persist the resulting support case.
    - Retrieve persisted support cases.
    - Convert domain results into API responses.
    """

    def __init__(
        self,
        agent: SupportAgent,
        case_service: CaseService,
    ) -> None:
        self._agent = agent
        self._case_service = case_service

    def list_cases(self) -> list[SupportCaseListItem]:
        """
        Return persisted support cases as lightweight summaries.

        Retrieval evidence is intentionally excluded from the list response.
        Detailed retrieval results remain available through the single-case
        endpoint.
        """

        cases = self._case_service.list_cases()

        return [
            self._to_list_item(case)
            for case in cases
        ]

    def analyze(
        self,
        request: SupportAnalyzeRequest,
    ) -> SupportAnalyzeResponse:
        """
        Analyze a customer support message and create a support case.
        """

        SUPPORT_ANALYSES_TOTAL.inc()

        try:
            # ---------------------------------------------------------
            # 1. Execute AI support workflow
            # ---------------------------------------------------------
            result = self._agent.handle(
                request.customer_message
            )

            # ---------------------------------------------------------
            # 2. Persist support case
            # ---------------------------------------------------------
            case = self._case_service.create_from_agent_result(
                customer_message=request.customer_message,
                result=result,
            )

            # ---------------------------------------------------------
            # 3. Record pipeline performance metrics
            # ---------------------------------------------------------
            self._record_performance_metrics(result)

            # ---------------------------------------------------------
            # 4. Record business metrics
            # ---------------------------------------------------------
            self._record_business_metrics(result)

            # ---------------------------------------------------------
            # 5. Structured application logging
            # ---------------------------------------------------------
            self._log_analysis_completed(
                case=case,
                result=result,
            )

            # ---------------------------------------------------------
            # 6. Convert domain case to API response
            # ---------------------------------------------------------
            return self.to_response(case)

        except Exception:
            SUPPORT_ANALYSES_FAILED_TOTAL.inc()

            logger.exception(
                "support_analysis_failed",
                extra={
                    "event": "support_analysis_failed",
                },
            )

            raise

    def get_case(
        self,
        case_id: str,
    ) -> SupportAnalyzeResponse:
        """
        Retrieve a previously analyzed support case.
        """

        case = self._case_service.get_case(case_id)

        return self.to_response(case)

    @staticmethod
    def _record_performance_metrics(
        result: AgentResult,
    ) -> None:
        """Record AI pipeline timing metrics."""

        SUPPORT_ANALYSIS_DURATION.observe(
            result.timing.total_ms / 1000
        )

        CLASSIFICATION_DURATION.observe(
            result.timing.classification_ms / 1000
        )

        RETRIEVAL_DURATION.observe(
            result.timing.retrieval_ms / 1000
        )

        GENERATION_DURATION.observe(
            result.timing.generation_ms / 1000
        )

        ESCALATION_DURATION.observe(
            result.timing.escalation_ms / 1000
        )

    @staticmethod
    def _record_business_metrics(
        result: AgentResult,
    ) -> None:
        """Record support-level business metrics."""

        if result.escalation.should_escalate:
            SUPPORT_ESCALATIONS_TOTAL.inc()

        if result.escalation.reason_code == "low_confidence":
            SUPPORT_LOW_CONFIDENCE_TOTAL.inc()

    @staticmethod
    def _log_analysis_completed(
        case: SupportCase,
        result: AgentResult,
    ) -> None:
        """Write structured information about a completed analysis."""

        logger.info(
            "support_analysis_completed",
            extra={
                "event": "support_analysis_completed",
                "case_id": case.case_id,
                "status": case.status.value,
                "intent": result.intent,
                "intent_confidence": result.intent_confidence,
                "should_escalate": (
                    result.escalation.should_escalate
                ),
                "retrieval_count": len(
                    result.retrieved_examples
                ),
                "classification_ms": (
                    result.timing.classification_ms
                ),
                "retrieval_ms": (
                    result.timing.retrieval_ms
                ),
                "generation_ms": (
                    result.timing.generation_ms
                ),
                "escalation_ms": (
                    result.timing.escalation_ms
                ),
                "total_ms": result.timing.total_ms,
            },
        )

    @staticmethod
    def _to_list_item(case: SupportCase) -> SupportCaseListItem:
        result = case.agent_result

        escalation = EscalationResult(
            should_escalate=result.escalation.should_escalate,
            reason=result.escalation.reason,
            reason_code=result.escalation.reason_code,
        )
        performance = PerformanceMetrics(
            classification_ms=result.timing.classification_ms,
            retrieval_ms=result.timing.retrieval_ms,
            generation_ms=result.timing.generation_ms,
            escalation_ms=result.timing.escalation_ms,
            total_ms=result.timing.total_ms,
        )

        return SupportCaseListItem(
            case_id=case.case_id,
            status=case.status,
            customer_message=case.customer_message,
            intent=result.intent,
            intent_confidence=result.intent_confidence,
            draft_response=result.draft_response,
            generation_model=result.generation_model,
            grounded=result.grounded,
            escalation=escalation,
            performance=performance,
        )

    @staticmethod
    def to_response(
        case: SupportCase,
    ) -> SupportAnalyzeResponse:
        """
        Convert a persisted support case into the public API response.
        """

        result = case.agent_result

        # -------------------------------------------------------------
        # Historical retrieval results
        # -------------------------------------------------------------
        retrieved_cases = [
            RetrievedCase(
                customer_text=example.customer_text,
                historical_response=example.historical_response,
                similarity=example.similarity,
            )
            for example in result.retrieved_examples
        ]

        # -------------------------------------------------------------
        # Escalation decision
        # -------------------------------------------------------------
        escalation = EscalationResult(
            should_escalate=result.escalation.should_escalate,
            reason=result.escalation.reason,
            reason_code=result.escalation.reason_code,
        )

        # -------------------------------------------------------------
        # Pipeline performance
        # -------------------------------------------------------------
        performance = PerformanceMetrics(
            classification_ms=result.timing.classification_ms,
            retrieval_ms=result.timing.retrieval_ms,
            generation_ms=result.timing.generation_ms,
            escalation_ms=result.timing.escalation_ms,
            total_ms=result.timing.total_ms,
        )

        # -------------------------------------------------------------
        # Public API response
        # -------------------------------------------------------------
        return SupportAnalyzeResponse(
            case_id=case.case_id,
            status=case.status,
            customer_message=case.customer_message,
            intent=result.intent,
            intent_confidence=result.intent_confidence,
            classification_reason=result.classification_reason,
            retrieved_cases=retrieved_cases,
            draft_response=result.draft_response,
            generation_model=result.generation_model,
            grounded=result.grounded,
            escalation=escalation,
            performance=performance,
        )
    def approve_case(self, case_id: str) -> SupportAnalyzeResponse:
        """
        Approve a support case for automatic handling.
        """

        case = self._case_service.update_status(
            case_id=case_id,
            status=CaseStatus.AUTO_HANDLED,
        )

        logger.info(
            "support_case_approved",
            extra={
                "event": "support_case_approved",
                "case_id": case.case_id,
            },
        )

        return self.to_response(case)

    def escalate_case(self, case_id: str) -> SupportAnalyzeResponse:
        """
        Escalate a support case for human handling.
        """

        case = self._case_service.update_status(
            case_id=case_id,
            status=CaseStatus.ESCALATED,
        )

        logger.info(
            "support_case_escalated",
            extra={
                "event": "support_case_escalated",
                "case_id": case.case_id,
            },
        )

        return self.to_response(case)
    def resolve_case(
        self,
        case_id: str,
        *,
        actor: str = "agent",
    ) -> SupportAnalyzeResponse:
        case = self._case_service.update_status(
            case_id=case_id,
            status=CaseStatus.RESOLVED,
            actor=actor,
        )

        logger.info(
            "support_case_resolved",
            extra={
                "event": "support_case_resolved",
                "case_id": case.case_id,
                "actor": actor,
            },
        )

        return self.to_response(case)