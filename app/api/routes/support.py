from fastapi import APIRouter, Depends

from app.container import (
    get_case_event_service,
    get_case_service,
    get_support_service,
)
from app.models.case import CaseStatus, SupportCase
from app.schemas.case_event import CaseEventResponse
from app.schemas.support import (
    EscalationResult,
    PerformanceMetrics,
    RetrievedCase,
    SupportAnalyzeRequest,
    SupportAnalyzeResponse,
    SupportCaseListItem,
)
from app.services.case_event_service import CaseEventService
from app.services.case_service import CaseService
from app.services.support_service import SupportService


router = APIRouter(
    prefix="/support",
    tags=["Support"],
)


# ---------------------------------------------------------------------------
# Response mapping
# ---------------------------------------------------------------------------


def _case_to_response(case: SupportCase) -> SupportAnalyzeResponse:
    """
    Convert the domain support case into the detailed API response schema.

    Keeps domain models isolated from FastAPI/Pydantic response models.
    """

    result = case.agent_result

    return SupportAnalyzeResponse(
        customer_message=case.customer_message,
        intent=result.intent,
        intent_confidence=result.intent_confidence,
        classification_reason=result.classification_reason,
        case_id=case.case_id,
        status=case.status.value,
        retrieved_cases=[
            RetrievedCase(
                customer_text=example.customer_text,
                historical_response=example.historical_response,
                similarity=example.similarity,
            )
            for example in result.retrieved_examples
        ],
        performance=PerformanceMetrics(
            classification_ms=result.timing.classification_ms,
            retrieval_ms=result.timing.retrieval_ms,
            generation_ms=result.timing.generation_ms,
            escalation_ms=result.timing.escalation_ms,
            total_ms=result.timing.total_ms,
        ),
        draft_response=result.draft_response,
        generation_model=result.generation_model,
        grounded=result.grounded,
        escalation=EscalationResult(
        should_escalate=result.escalation.should_escalate,
        reason=result.escalation.reason,
        reason_code=result.escalation.reason_code,
    ),
    )


def _case_to_list_item(case: SupportCase) -> SupportCaseListItem:
    """
    Convert a domain support case into the lightweight case-list response.

    The list representation intentionally excludes retrieved examples so the
    API does not expose unnecessary retrieval payloads.
    """

    result = case.agent_result

    return SupportCaseListItem(
        case_id=case.case_id,
        status=case.status.value,
        customer_message=case.customer_message,
        intent=result.intent,
        intent_confidence=result.intent_confidence,
        draft_response=result.draft_response,
        generation_model=result.generation_model,
        grounded=result.grounded,
        escalation=EscalationResult(
            should_escalate=result.escalation.should_escalate,
            reason=result.escalation.reason,
            reason_code=result.escalation.reason_code,
        ),
        performance=PerformanceMetrics(
            classification_ms=result.timing.classification_ms,
            retrieval_ms=result.timing.retrieval_ms,
            generation_ms=result.timing.generation_ms,
            escalation_ms=result.timing.escalation_ms,
            total_ms=result.timing.total_ms,
        ),
    )


# ---------------------------------------------------------------------------
# AI analysis
# ---------------------------------------------------------------------------


@router.post(
    "/analyze",
    response_model=SupportAnalyzeResponse,
)
def analyze_support_case(
    request: SupportAnalyzeRequest,
    service: SupportService = Depends(get_support_service),
) -> SupportAnalyzeResponse:
    """
    Analyze a customer-support message with the AI support agent.

    This is the only route in this module that initializes the full AI
    dependency graph, including classification, retrieval, generation,
    response policy, and escalation policy.
    """

    return service.analyze(request)


# ---------------------------------------------------------------------------
# Case retrieval
# ---------------------------------------------------------------------------


@router.get(
    "/cases",
    response_model=list[SupportCaseListItem],
)
def list_support_cases(
    service: CaseService = Depends(get_case_service),
) -> list[SupportCaseListItem]:
    """
    List persisted support cases, newest first.

    This endpoint intentionally uses the persistence-only CaseService so
    listing cases does not initialize the AI/retrieval stack.
    """

    cases = service.list_cases()

    return [
        _case_to_list_item(case)
        for case in cases
    ]


@router.get(
    "/cases/{case_id}",
    response_model=SupportAnalyzeResponse,
)
def get_support_case(
    case_id: str,
    service: CaseService = Depends(get_case_service),
) -> SupportAnalyzeResponse:
    """
    Retrieve a previously analyzed support case by case ID.
    """

    case = service.get_case(case_id)

    return _case_to_response(case)


# ---------------------------------------------------------------------------
# Case lifecycle
# ---------------------------------------------------------------------------


@router.post(
    "/cases/{case_id}/approve",
    response_model=SupportAnalyzeResponse,
)
def approve_support_case(
    case_id: str,
    service: CaseService = Depends(get_case_service),
) -> SupportAnalyzeResponse:
    """
    Approve a reviewable case for automatic handling.

    The status transition and audit event are committed atomically by the
    CaseService unit-of-work.
    """

    case = service.update_status(
        case_id,
        CaseStatus.AUTO_HANDLED,
        actor="agent",
    )

    return _case_to_response(case)


@router.post(
    "/cases/{case_id}/escalate",
    response_model=SupportAnalyzeResponse,
)
def escalate_support_case(
    case_id: str,
    service: CaseService = Depends(get_case_service),
) -> SupportAnalyzeResponse:
    """
    Escalate a reviewable case for human handling.

    The status transition and audit event are committed atomically by the
    CaseService unit-of-work.
    """

    case = service.update_status(
        case_id,
        CaseStatus.ESCALATED,
        actor="agent",
    )

    return _case_to_response(case)


@router.post(
    "/cases/{case_id}/resolve",
    response_model=SupportAnalyzeResponse,
)
def resolve_support_case(
    case_id: str,
    service: CaseService = Depends(get_case_service),
) -> SupportAnalyzeResponse:
    """
    Mark an automatically handled or escalated case as resolved.

    The status transition and audit event are committed atomically by the
    CaseService unit-of-work.
    """

    case = service.update_status(
        case_id,
        CaseStatus.RESOLVED,
        actor="agent",
    )

    return _case_to_response(case)


# ---------------------------------------------------------------------------
# Audit events
# ---------------------------------------------------------------------------


@router.get(
    "/cases/{case_id}/events",
    response_model=list[CaseEventResponse],
)
def get_case_events(
    case_id: str,
    case_service: CaseService = Depends(get_case_service),
    event_service: CaseEventService = Depends(get_case_event_service),
) -> list[CaseEventResponse]:
    """
    Return the immutable audit history for a support case.

    The case is validated first so an unknown case ID does not return an
    apparently valid empty event list.
    """

    case_service.get_case(case_id)

    events = event_service.list_events(case_id)

    return [
        CaseEventResponse(
            event_type=event.event_type,
            actor=event.actor,
            created_at=event.created_at,
            metadata=event.metadata,
        )
        for event in events
    ]