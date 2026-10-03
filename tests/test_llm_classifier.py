from unittest.mock import Mock

from src.intents.llm_classifier import LLMIntentClassifier
from src.intents.models import IntentPrediction


def make_classifier(response_content: str):
    client = Mock()

    client.chat.return_value = {
        "message": {
            "content": response_content,
        }
    }

    return LLMIntentClassifier(
        client=client,
    )


def test_valid_llm_prediction():
    classifier = make_classifier(
        """
        {
          "intent": "delivered_not_received",
          "confidence": 0.91,
          "reason": "The customer says the order is marked delivered but was not received."
        }
        """
    )

    result = classifier.predict(
        "My order says delivered but I never received it."
    )

    assert isinstance(result, IntentPrediction)
    assert result.intent == "delivered_not_received"
    assert result.confidence == 0.91
    assert result.reason


def test_invalid_intent_falls_back_to_unclear():
    classifier = make_classifier(
        """
        {
          "intent": "some_random_intent",
          "confidence": 0.95,
          "reason": "Something."
        }
        """
    )

    result = classifier.predict("Where is my order?")

    assert result.intent == "other_or_unclear"
    assert result.confidence == 0.0


def test_invalid_json_falls_back_to_unclear():
    classifier = make_classifier(
        "I cannot classify this message."
    )

    result = classifier.predict("Where is my order?")

    assert result.intent == "other_or_unclear"
    assert result.confidence == 0.0


def test_confidence_outside_range_falls_back_to_unclear():
    classifier = make_classifier(
        """
        {
          "intent": "order_status",
          "confidence": 1.5,
          "reason": "The customer asks for order status."
        }
        """
    )

    result = classifier.predict("Where is my order?")

    assert result.intent == "other_or_unclear"
    assert result.confidence == 0.0