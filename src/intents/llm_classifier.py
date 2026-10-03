import json
import re

import ollama

from src.intents.models import IntentPrediction


MODEL_NAME = "ministral-3:3b"


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


INTENT_DEFINITIONS = {
    "order_status": "Customer wants to know where their order is or its current status.",
    "delivery_late": "Expected delivery date has passed or the order is delayed.",
    "delivered_not_received": "Tracking says delivered, but the customer did not receive the package.",
    "delivery_promise": "A promised, guaranteed, Prime, same-day, or next-day delivery commitment was not met.",
    "missing_or_wrong_item": "An item is missing from the order or the customer received the wrong item.",
    "damaged_item_or_package": "The product or package arrived damaged, broken, leaking, or defective on arrival.",
    "return_or_refund": "Customer primarily asks about returning an item or receiving a refund.",
    "payment_or_charge": "Customer has a payment, charge, billing, or transaction problem.",
    "prime_membership": "Customer asks about Amazon Prime membership, benefits, subscription, or Prime-specific account issues.",
    "account_or_access": "Customer has an account, login, password, access, or account-information problem.",
    "product_or_device_help": "Customer needs help with a product, device, product page, compatibility, setup, or product functionality.",
    "customer_service_followup": "Customer is following up on an existing support interaction or says they have contacted support without resolution.",
    "other_or_unclear": "The message does not clearly fit any supported intent.",
}


BOUNDARIES = """
Important classification boundaries:

1. order_status:
   "Where is my order?" or a request for the current order status.

2. delivery_late:
   The expected delivery date/time has passed and the customer says the delivery is late.

3. delivered_not_received:
   Tracking explicitly says delivered, but the customer says they did not receive it.

4. delivery_promise:
   The customer specifically refers to a promised, guaranteed, Prime, same-day,
   or next-day delivery commitment that was not met.

5. return_or_refund:
   Use this when the primary issue is returning an item or obtaining a refund.
   Do not choose it merely because the customer mentions a refund while describing
   an underlying delivery problem.

6. customer_service_followup:
   Use this when the main point is an unresolved/repeated support interaction,
   lack of response, or following up with customer service.

Choose the underlying primary issue when a message mentions both an issue and a
requested remedy.
"""


SYSTEM_PROMPT = f"""
You are an intent classifier for an Amazon customer-support system.

Classify the customer's message into exactly one supported intent.

Supported intents:

{json.dumps(INTENT_DEFINITIONS, indent=2)}

{BOUNDARIES}

Return ONLY valid JSON with exactly these fields:

{{
  "intent": "one supported intent",
  "confidence": 0.0,
  "reason": "short explanation based only on the customer message"
}}

Confidence must be a number between 0.0 and 1.0.

Confidence guidance:
- 0.90-1.00: very clear intent
- 0.75-0.89: clear intent
- 0.55-0.74: somewhat ambiguous
- 0.00-0.54: uncertain

Do not invent facts.
Do not assume account information, order information, policies, or actions.
Do not include markdown.
Do not include additional JSON fields.
"""


class LLMIntentClassifier:
    """Intent classifier backed by a local Ollama LLM."""

    def __init__(
        self,
        model_name: str = MODEL_NAME,
        client=None,
    ):
        self.model_name = model_name
        self.client = client or ollama.Client(
        host="http://127.0.0.1:11434",
         )

    def predict(self, text: str) -> IntentPrediction:
        if not isinstance(text, str):
            raise TypeError("text must be a string")

        text = text.strip()

        if not text:
            raise ValueError("text must not be empty")

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
            options={
                "temperature": 0,
                "num_predict": 120,
            },
        )

        raw_content = response["message"]["content"]

        return self._parse_prediction(raw_content)

    def _parse_prediction(self, raw_content: str) -> IntentPrediction:
        try:
            data = self._extract_json(raw_content)

            intent = data.get("intent")
            confidence = data.get("confidence")
            reason = data.get("reason")

            if intent not in SUPPORTED_INTENTS:
                raise ValueError(
                    f"Unsupported intent returned by LLM: {intent!r}"
                )

            if not isinstance(confidence, (int, float)):
                raise ValueError("LLM confidence must be numeric")

            confidence = float(confidence)

            if not 0.0 <= confidence <= 1.0:
                raise ValueError("LLM confidence must be between 0 and 1")

            if not isinstance(reason, str) or not reason.strip():
                raise ValueError("LLM reason must be a non-empty string")

            return IntentPrediction(
                intent=intent,
                confidence=confidence,
                reason=reason.strip(),
            )

        except (json.JSONDecodeError, TypeError, ValueError, AttributeError):
            return IntentPrediction(
                intent="other_or_unclear",
                confidence=0.0,
                reason="LLM returned an invalid classification.",
            )

    @staticmethod
    def _extract_json(content: str) -> dict:
        content = content.strip()

        try:
            return json.loads(content)
        except json.JSONDecodeError:
            pass

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

        return json.loads(match.group(0))