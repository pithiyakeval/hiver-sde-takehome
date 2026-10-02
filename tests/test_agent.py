from src.escalation.policy import EscalationPolicy
from src.generation.generator import ResponseGenerator
from src.generation.mock_provider import MockLLMProvider
from src.intents.classifier import IntentClassifier
from src.pipeline.agent import SupportAgent
from src.retrieval.tfidf_retriever import TfidfRetriever


def test_support_agent_returns_complete_result():
    classifier = IntentClassifier()
    retriever = TfidfRetriever()

    provider = MockLLMProvider(
        response="We're sorry for the delay. Please check your latest tracking update."
    )
    generator = ResponseGenerator(provider)

    escalation_policy = EscalationPolicy()

    agent = SupportAgent(
        classifier=classifier,
        retriever=retriever,
        generator=generator,
        escalation_policy=escalation_policy,
        retrieval_top_k=3,
    )

    result = agent.handle(
        "My order was supposed to arrive yesterday but it is still not here."
    )

    assert result.customer_message
    assert result.intent
    assert 0.0 <= result.intent_confidence <= 1.0

    assert len(result.retrieved_examples) == 3

    assert result.draft_response == (
        "We're sorry for the delay. Please check your latest tracking update."
    )

    assert result.grounded is True

    assert isinstance(result.escalation.should_escalate, bool)
    assert result.escalation.reason


def test_unclear_request_is_escalated():
    classifier = IntentClassifier()
    retriever = TfidfRetriever()

    generator = ResponseGenerator(
        MockLLMProvider(response="Could you provide more details?")
    )

    agent = SupportAgent(
        classifier=classifier,
        retriever=retriever,
        generator=generator,
        escalation_policy=EscalationPolicy(),
    )

    result = agent.handle("asdf qwerty xyz")

    assert result.escalation.should_escalate is True
    assert result.escalation.reason