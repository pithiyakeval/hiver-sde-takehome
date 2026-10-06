from src.escalation.policy import EscalationPolicy
from src.generation.generator import ResponseGenerator
from src.generation.mock_provider import MockLLMProvider
from src.generation.response_policy import ResponsePolicy
from src.intents.classifier import IntentClassifier
from src.pipeline.agent import SupportAgent
from src.retrieval.tfidf_retriever import TfidfRetriever


def build_agent(
    provider: MockLLMProvider,
) -> SupportAgent:
    classifier = IntentClassifier()
    retriever = TfidfRetriever()
    generator = ResponseGenerator(provider)
    escalation_policy = EscalationPolicy()
    response_policy = ResponsePolicy()

    return SupportAgent(
        classifier=classifier,
        retriever=retriever,
        generator=generator,
        escalation_policy=escalation_policy,
        response_policy=response_policy,
        retrieval_top_k=3,
    )


def test_support_agent_returns_complete_result():
    agent = build_agent(
        MockLLMProvider(
            response=(
                "We're sorry for the delay. "
                "Please check your latest tracking update."
            )
        )
    )

    result = agent.handle(
        "My order was supposed to arrive yesterday "
        "but it is still not here."
    )

    assert result.customer_message
    assert result.intent
    assert 0.0 <= result.intent_confidence <= 1.0

    assert len(result.retrieved_examples) == 3

    assert result.draft_response

    assert isinstance(result.grounded, bool)

    assert isinstance(result.escalation.should_escalate, bool)
    assert result.escalation.reason

    assert result.timing.classification_ms >= 0
    assert result.timing.retrieval_ms >= 0
    assert result.timing.generation_ms >= 0
    assert result.timing.escalation_ms >= 0
    assert result.timing.total_ms >= 0


def test_unclear_request_is_escalated():
    agent = build_agent(
        MockLLMProvider(
            response="Could you provide more details?"
        )
    )

    result = agent.handle("asdf qwerty xyz")

    assert result.escalation.should_escalate is True
    assert result.escalation.reason