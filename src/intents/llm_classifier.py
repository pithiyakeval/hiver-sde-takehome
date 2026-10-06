import json
import re
from typing import Any

import ollama

from src.intents.classifier import IntentClassifier
from src.intents.models import IntentPrediction


MODEL_NAME = "ministral-3:3b"
OLLAMA_HOST = "http://127.0.0.1:11434"

# The fast classifier is trusted above this confidence.
# Ambiguous cases are sent to the LLM fallback.
LLM_FALLBACK_THRESHOLD = 0.45


# ---------------------------------------------------------------------------
# Canonical intent taxonomy
# ---------------------------------------------------------------------------

SUPPORTED_INTENTS = {
    "order_status",
    "delivery_late",
    "delivered_not_received",
    "delivery_promise",
    "missing_or_wrong_item",
    "damaged_item_or_package",
    "return_or_refund",
    "payment_or_charge",
    "prime_membership",
    "account_or_access",
    "product_or_device_help",
    "customer_service_followup",
    "other_or_unclear",
}


# ---------------------------------------------------------------------------
# Legacy / compact intent aliases
#
# The deterministic rule layer uses short labels such as DNR and RR.
# The public application contract uses the canonical names above.
# Keep this normalization inside the classifier boundary so downstream
# services never need to understand multiple label formats.
# ---------------------------------------------------------------------------

INTENT_ALIASES = {
    "OS": "order_status",
    "DL": "delivery_late",
    "DNR": "delivered_not_received",
    "DP": "delivery_promise",
    "MWI": "missing_or_wrong_item",
    "DIP": "damaged_item_or_package",
    "RR": "return_or_refund",
    "PC": "payment_or_charge",
    "PM": "prime_membership",
    "AA": "account_or_access",
    "PDH": "product_or_device_help",
    "CSF": "customer_service_followup",
    "OU": "other_or_unclear",
}


# ---------------------------------------------------------------------------
# Intent definitions used by the LLM fallback
# ---------------------------------------------------------------------------

INTENT_DEFINITIONS = {
    "order_status": "Where is my order or what is its current status?",
    "delivery_late": "Expected delivery date passed; order is late.",
    "delivered_not_received": (
        "Tracking says delivered, but the customer did not receive it."
    ),
    "delivery_promise": (
        "A promised, guaranteed, Prime, same-day, or next-day "
        "delivery commitment was not met."
    ),
    "missing_or_wrong_item": (
        "An item is missing or the wrong item was received."
    ),
    "damaged_item_or_package": (
        "The product or package arrived damaged, broken, leaking, "
        "or defective."
    ),
    "return_or_refund": (
        "The primary issue is returning an item or obtaining a refund."
    ),
    "payment_or_charge": (
        "The customer has a payment, charge, billing, or transaction problem."
    ),
    "prime_membership": (
        "The issue concerns Prime membership, benefits, subscription, "
        "or another Prime-specific matter."
    ),
    "account_or_access": (
        "The issue concerns account access, login, password, "
        "or account information."
    ),
    "product_or_device_help": (
        "The customer needs product, device, compatibility, setup, "
        "or functionality help."
    ),
    "customer_service_followup": (
        "The customer is following up on an unresolved or repeated "
        "customer-service interaction."
    ),
    "other_or_unclear": (
        "The message does not clearly fit another supported intent."
    ),
}


INTENT_DEFINITIONS_TEXT = "\n".join(
    f"- {intent}: {description}"
    for intent, description in INTENT_DEFINITIONS.items()
)


# ---------------------------------------------------------------------------
# LLM prompt
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = f"""
Classify the customer message into exactly one supported intent.

Supported intents:
{INTENT_DEFINITIONS_TEXT}

Classification rules:
- delivered_not_received requires the tracking/status to say delivered
  AND the customer to say they did not receive the order.
- delivery_late means the expected delivery date has passed.
- delivery_promise means a specific promised, guaranteed, Prime,
  same-day, or next-day delivery commitment was not met.
- order_status is a general order-location or order-status question.
- For an issue plus a requested remedy, classify the underlying
  primary issue rather than the requested remedy.
- Use customer_service_followup when the main issue is an unresolved
  or repeated support interaction.
- Use other_or_unclear when no supported intent is sufficiently clear.

Return ONLY valid JSON:
{{"intent":"...","confidence":0.0,"reason":"short reason"}}

Requirements:
- intent must be one of the supported intents.
- confidence must be a number between 0.0 and 1.0.
- reason must be short and explain the classification.
- Do not add additional fields.
- Do not return markdown.
""".strip()


class LLMIntentClassifier:
    """
    Hybrid production intent classifier.

    Classification strategy:

    1. The fast deterministic/ML classifier handles normal cases.
    2. High-confidence fast predictions bypass the LLM entirely.
    3. Low-confidence predictions are sent to the local LLM.
    4. All classifier outputs are normalized to the canonical intent
       taxonomy before being returned to the application.

    This architecture keeps the normal request path fast while
    retaining an LLM fallback for genuinely ambiguous messages.
    """

    def __init__(
        self,
        model_name: str = MODEL_NAME,
        client: Any | None = None,
        fallback_threshold: float = LLM_FALLBACK_THRESHOLD,
    ) -> None:
        if not 0.0 <= fallback_threshold <= 1.0:
            raise ValueError(
                "fallback_threshold must be between 0 and 1"
            )

        self.model_name = model_name
        self.fallback_threshold = fallback_threshold

        self.client = client or ollama.Client(
            host=OLLAMA_HOST,
        )

        # Fit the fast classifier once when the service starts.
        # It is reused for every prediction.
        self.fast_classifier = IntentClassifier()

    def predict(self, text: str) -> IntentPrediction:
        """
        Predict the customer's intent.

        Fast path:
            rules / TF-IDF + Logistic Regression

        Fallback path:
            local Ollama LLM for low-confidence predictions
        """

        text = self._validate_text(text)

        # ----------------------------------------------------------
        # 1. Fast classification
        # ----------------------------------------------------------

        fast_prediction = self.fast_classifier.predict(text)

        normalized_prediction = self._normalize_prediction(
            fast_prediction
        )

        # ----------------------------------------------------------
        # 2. High-confidence fast path
        # ----------------------------------------------------------

        if (
            normalized_prediction.confidence
            >= self.fallback_threshold
        ):
            return normalized_prediction

        # ----------------------------------------------------------
        # 3. LLM fallback
        # ----------------------------------------------------------

        return self._predict_with_llm(text)

    def _predict_with_llm(
        self,
        text: str,
    ) -> IntentPrediction:
        """
        Classify an ambiguous message using the local LLM.
        """

        response = self.client.chat(
            model=self.model_name,
            messages=[
                {
                    "role": "system",
                    "content": SYSTEM_PROMPT,
                },
                {
                    "role": "user",
                    "content": text,
                },
            ],
            format="json",
            options={
                "temperature": 0,
                "num_predict": 60,
            },
        )

        raw_content = response["message"]["content"]

        prediction = self._parse_prediction(raw_content)

        return IntentPrediction(
            intent=prediction.intent,
            confidence=prediction.confidence,
            reason=(
                "LLM fallback used because the fast classifier "
                "was uncertain. "
                f"{prediction.reason}"
            ),
        )

    def _parse_prediction(
        self,
        raw_content: str,
    ) -> IntentPrediction:
        """
        Parse and validate the LLM JSON response.
        """

        try:
            data = self._extract_json(raw_content)

            intent = data.get("intent")
            confidence = data.get("confidence")
            reason = data.get("reason")

            # Normalize compact aliases if the LLM happens to return one.
            intent = self._normalize_intent(intent)

            if intent not in SUPPORTED_INTENTS:
                raise ValueError(
                    f"Unsupported intent returned by LLM: {intent!r}"
                )

            if not isinstance(confidence, (int, float)):
                raise ValueError(
                    "LLM confidence must be numeric"
                )

            confidence = float(confidence)

            if not 0.0 <= confidence <= 1.0:
                raise ValueError(
                    "LLM confidence must be between 0 and 1"
                )

            if not isinstance(reason, str) or not reason.strip():
                raise ValueError(
                    "LLM reason must be a non-empty string"
                )

            return IntentPrediction(
                intent=intent,
                confidence=confidence,
                reason=reason.strip(),
            )

        except (
            json.JSONDecodeError,
            TypeError,
            ValueError,
            AttributeError,
        ):
            return IntentPrediction(
                intent="other_or_unclear",
                confidence=0.0,
                reason="LLM returned an invalid classification.",
            )

    def _normalize_prediction(
        self,
        prediction: IntentPrediction,
    ) -> IntentPrediction:
        """
        Normalize a prediction produced by the fast classifier.

        The rule layer may return compact labels such as DNR.
        The application contract always receives canonical names.
        """

        normalized_intent = self._normalize_intent(
            prediction.intent
        )

        return IntentPrediction(
            intent=normalized_intent,
            confidence=prediction.confidence,
            reason=prediction.reason,
        )

    @staticmethod
    def _normalize_intent(
        intent: object,
    ) -> str:
        """
        Convert an intent into the canonical application taxonomy.
        """

        if not isinstance(intent, str):
            raise ValueError(
                "Intent must be a string"
            )

        normalized = intent.strip()

        normalized = INTENT_ALIASES.get(
            normalized,
            normalized,
        )

        return normalized

    @staticmethod
    def _validate_text(text: str) -> str:
        """
        Validate and normalize the incoming customer message.
        """

        if not isinstance(text, str):
            raise TypeError("text must be a string")

        text = text.strip()

        if not text:
            raise ValueError("text must not be empty")

        return text

    @staticmethod
    def _extract_json(content: str) -> dict:
        """
        Extract a JSON object from the LLM response.

        Handles both:
        - pure JSON responses
        - JSON surrounded by accidental text
        """

        if not isinstance(content, str):
            raise TypeError(
                "LLM response content must be a string"
            )

        content = content.strip()

        # Preferred path: response is already valid JSON.
        try:
            data = json.loads(content)

            if not isinstance(data, dict):
                raise TypeError(
                    "LLM response JSON must be an object"
                )

            return data

        except json.JSONDecodeError:
            pass

        # Defensive fallback: locate the first JSON object.
        match = re.search(
            r"\{.*\}",
            content,
            flags=re.DOTALL,
        )

        if not match:
            raise json.JSONDecodeError(
                "No JSON object found",
                content,
                0,
            )

        data = json.loads(match.group(0))

        if not isinstance(data, dict):
            raise TypeError(
                "Extracted LLM JSON must be an object"
            )

        return data