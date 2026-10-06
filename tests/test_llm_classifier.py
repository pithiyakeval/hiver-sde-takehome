from unittest.mock import Mock

import pytest

from src.intents.llm_classifier import LLMIntentClassifier
from src.intents.models import IntentPrediction


def make_classifier(
    response_content: str,
    *,
    fallback_threshold: float = 0.45,
):
    client = Mock()

    client.chat.return_value = {
        "message": {
            "content": response_content,
        }
    }

    return LLMIntentClassifier(
        client=client,
        fallback_threshold=fallback_threshold,
    ), client


def test_valid_llm_prediction():
    """
    Verify that a valid LLM response is parsed correctly.

    The threshold is deliberately set above the fast classifier's
    rule confidence so this test exercises the LLM fallback path.
    """

    classifier, client = make_classifier(
        """
        {
          "intent": "delivered_not_received",
          "confidence": 0.91,
          "reason": "The customer says the order is marked delivered but was not received."
        }
        """,
        fallback_threshold=1.0,
    )

    result = classifier.predict(
        "My order says delivered but I never received it."
    )

    assert isinstance(result, IntentPrediction)
    assert result.intent == "delivered_not_received"
    assert result.confidence == 0.91
    assert result.reason

    client.chat.assert_called_once()


def test_high_confidence_fast_prediction_skips_llm():
    """
    High-confidence deterministic/ML predictions should not call Ollama.
    """

    classifier, client = make_classifier(
        """
        {
          "intent": "other_or_unclear",
          "confidence": 0.0,
          "reason": "This response should not be used."
        }
        """
    )

    result = classifier.predict("My order says delivered but I never received it.")

    assert isinstance(result, IntentPrediction)
    assert result.intent == "delivered_not_received"
    assert result.confidence == 0.99

    client.chat.assert_not_called()


def test_rule_alias_is_normalized_to_canonical_intent():
    """
    Compact rule labels such as DNR must never leak outside
    the classifier boundary.
    """

    classifier, client = make_classifier(
        """
        {
          "intent": "other_or_unclear",
          "confidence": 0.0,
          "reason": "This response should not be used."
        }
        """
    )

    result = classifier.predict(
        "My order says delivered but I never received it."
    )

    assert result.intent == "delivered_not_received"
    assert result.intent != "DNR"

    client.chat.assert_not_called()


def test_invalid_intent_falls_back_to_unclear():
    """
    An unsupported intent returned by the LLM must fail safely.
    """

    classifier, client = make_classifier(
        """
        {
          "intent": "some_random_intent",
          "confidence": 0.95,
          "reason": "Something."
        }
        """,
        fallback_threshold=1.0,
    )

    result = classifier.predict(
        "Where is my order?"
    )

    assert result.intent == "other_or_unclear"
    assert result.confidence == 0.0

    client.chat.assert_called_once()


def test_invalid_json_falls_back_to_unclear():
    """
    Invalid LLM output must fail safely instead of crashing
    the support pipeline.
    """

    classifier, client = make_classifier(
        "I cannot classify this message.",
        fallback_threshold=1.0,
    )

    result = classifier.predict(
        "Where is my order?"
    )

    assert result.intent == "other_or_unclear"
    assert result.confidence == 0.0

    client.chat.assert_called_once()


def test_confidence_outside_range_falls_back_to_unclear():
    """
    LLM confidence values outside [0, 1] must be rejected.
    """

    classifier, client = make_classifier(
        """
        {
          "intent": "order_status",
          "confidence": 1.5,
          "reason": "The customer asks for order status."
        }
        """,
        fallback_threshold=1.0,
    )

    result = classifier.predict(
        "Where is my order?"
    )

    assert result.intent == "other_or_unclear"
    assert result.confidence == 0.0

    client.chat.assert_called_once()


def test_empty_message_is_rejected():
    """
    Empty customer messages should fail before classification.
    """

    classifier, _ = make_classifier("{}")

    with pytest.raises(ValueError, match="text must not be empty"):
        classifier.predict("   ")


def test_non_string_message_is_rejected():
    """
    Non-string input should fail before classification.
    """

    classifier, _ = make_classifier("{}")

    with pytest.raises(TypeError, match="text must be a string"):
        classifier.predict(None)