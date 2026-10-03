import pytest

from src.intents.classifier import IntentClassifier
from src.intents.models import IntentPrediction


@pytest.fixture(scope="module")
def classifier():
    """Create one classifier instance for the test module."""
    return IntentClassifier()


@pytest.mark.parametrize(
    ("message", "expected_intent"),
    [
        ("Where is my Amazon order?", "OS"),
        (
            "My package says delivered but I never received it.",
            "DNR",
        ),
        ("I was charged twice for the same order.", "PC"),
        ("My account is locked and I cannot log in.", "AA"),
        ("I want a refund for my returned item.", "RR"),
    ],
)
def test_classifier_predicts_representative_intents(
    classifier,
    message,
    expected_intent,
):
    """Classifier should correctly handle representative support intents."""

    result = classifier.predict(message)

    assert result.intent == expected_intent, (
        f"\nMessage: {message}"
        f"\nExpected: {expected_intent}"
        f"\nPredicted: {result.intent}"
    )


@pytest.mark.parametrize(
    "message",
    [
        "Where is my Amazon order?",
        "My package says delivered but I never received it.",
        "I was charged twice for the same order.",
        "My account is locked and I cannot log in.",
        "I want a refund for my returned item.",
    ],
)
def test_classifier_response_contract(classifier, message):
    """Prediction should always follow the IntentPrediction contract."""

    result = classifier.predict(message)

    assert isinstance(result, IntentPrediction)

    assert isinstance(result.intent, str)
    assert result.intent.strip()

    assert isinstance(result.confidence, float)
    assert 0.0 <= result.confidence <= 1.0

    assert isinstance(result.reason, str)
    assert result.reason.strip()


def test_classifier_returns_deterministic_prediction(classifier):
    """Same input should produce the same prediction."""

    message = "My package says delivered but I never received it."

    first = classifier.predict(message)
    second = classifier.predict(message)

    assert first.intent == second.intent
    assert first.confidence == second.confidence
    assert first.reason == second.reason