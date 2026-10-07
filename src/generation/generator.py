from src.generation.models import GenerationInput, GenerationResult
from src.generation.prompts import SYSTEM_PROMPT, build_generation_prompt
from src.generation.provider import LLMProvider
from src.generation.response_cleaner import clean_model_response

class ResponseGenerator:
    """Generates safe customer-support responses."""

    DETERMINISTIC_RESPONSES = {
        "order_status": (
            "You can check your latest order status and tracking information "
            "in your order details. If you need further assistance, please "
            "contact support."
        ),
        "delivery_late": (
            "Sorry your delivery is taking longer than expected. Please "
            "check the latest tracking information in your order details, "
            "and contact support if you need further assistance."
        ),
        "delivery_promise": (
            "Sorry the expected delivery time was missed. Please check the "
            "latest order and tracking information, and contact support "
            "for further assistance."
        ),
        "prime_membership": (
            "For Prime membership questions, please check your membership "
            "details in your account or contact support for account-specific "
            "assistance."
        ),
    }

    def __init__(self, provider: LLMProvider):
        self.provider = provider

    def generate(self, data: GenerationInput) -> GenerationResult:
        """
        Generate a response using the configured LLM provider.

        Historical examples are supplied as grounding evidence.
        """

        user_prompt = build_generation_prompt(data)

        response = self.provider.generate(
            system_prompt=SYSTEM_PROMPT,
            user_prompt=user_prompt,
        )

        response = clean_model_response(response)

        return GenerationResult(
            draft_response=response,
            model=type(self.provider).__name__,
            grounded=bool(data.retrieved_examples),
        )

    def generate_deterministic(
        self,
        *,
        customer_message: str,
        intent: str,
        historical_response: str,
    ) -> GenerationResult:
        """
        Produce a fast response without calling an LLM.

        Historical responses are used only as evidence that the intent has
        historical support. They are NEVER reused as response text because
        they may contain customer-specific actions, identifiers, links,
        account information, or claims about actions taken for another
        customer.
        """

        has_historical_evidence = bool(
            historical_response and historical_response.strip()
        )

        response = self.DETERMINISTIC_RESPONSES.get(intent)

        if not response or not has_historical_evidence:
            response = self._safe_fallback(customer_message)

        return GenerationResult(
            draft_response=response,
            model="DeterministicResponseGenerator",
            grounded=has_historical_evidence,
        )

    def generate_fallback(
        self,
        *,
        customer_message: str,
    ) -> GenerationResult:
        """
        Produce a conservative response when AI generation is not safe.
        """

        response = self._safe_fallback(customer_message)

        return GenerationResult(
            draft_response=response,
            model="FallbackResponseGenerator",
            grounded=False,
        )

    @staticmethod
    def _safe_fallback(customer_message: str) -> str:
        """
        Return a conservative response when there is insufficient evidence.

        The response intentionally avoids claims about orders, accounts,
        tracking, refunds, or completed actions.
        """

        return (
            "Thanks for reaching out. We’re sorry we couldn’t provide "
            "a more specific answer at this time."
        )