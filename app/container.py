from fastapi import Depends

from app.config import get_settings
from app.db.session import get_db

from app.services.case_service import CaseService
from app.services.support_service import SupportService
from app.services.case_event_service import CaseEventService
from src.escalation.policy import EscalationPolicy
from src.generation.generator import ResponseGenerator
from src.generation.ollama_provider import OllamaProvider
from src.generation.response_policy import ResponsePolicy
from src.intents.llm_classifier import LLMIntentClassifier
from src.pipeline.agent import SupportAgent
from src.retrieval.tfidf_retriever import TfidfRetriever

from app.repositories.sqlalchemy_unit_of_work import (
    SqlAlchemyUnitOfWork,
)
def get_support_service(
    db=Depends(get_db),
) -> SupportService:
    """
    Build the production support service graph.

    The database session is request-scoped. AI components remain
    independent of persistence, while the case repository owns
    database interaction.
    """
    settings = get_settings()

    # -------------------------
    # AI components
    # -------------------------

    classifier = LLMIntentClassifier(
        model_name=settings.ollama_model,
    )

    retriever = TfidfRetriever()

    generator = ResponseGenerator(
        OllamaProvider(),
    )

    response_policy = ResponsePolicy()
    escalation_policy = EscalationPolicy()

    # -------------------------
    # AI agent
    # -------------------------

    agent = SupportAgent(
        classifier=classifier,
        retriever=retriever,
        generator=generator,
        escalation_policy=escalation_policy,
        response_policy=response_policy,
    )

    # -------------------------
    # Persistence
    # -------------------------

    unit_of_work = SqlAlchemyUnitOfWork(session=db)
    case_service = CaseService(
            unit_of_work=unit_of_work,
        )
    case_event_service = CaseEventService(
        unit_of_work=unit_of_work,
)

    # -------------------------
    # Application service
    # -------------------------

    return SupportService(
        agent=agent,
        case_service=case_service,
    )



def get_case_event_service(
    db=Depends(get_db),
) -> CaseEventService:
    unit_of_work = SqlAlchemyUnitOfWork(session=db)

    return CaseEventService(
        unit_of_work=unit_of_work,
    )

def get_case_service(
    db=Depends(get_db),
) -> CaseService:
    unit_of_work = SqlAlchemyUnitOfWork(session=db)

    return CaseService(
        unit_of_work=unit_of_work,
    )