from src.generation.models import GenerationInput, GenerationResult
from src.generation.prompts import SYSTEM_PROMPT, build_generation_prompt
from src.generation.provider import LLMProvider
from src.generation.response_cleaner import clean_model_response


class ResponseGenerator:
    """Generates safe customer-support responses."""

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

        The historical support response is treated as the source material.
        We remove Twitter-specific addressing/signatures but do not invent
        new support actions or facts.
        """

        response = self._clean_historical_response(
            historical_response
        )

        if not response:
            response = self._safe_fallback(customer_message)

        return GenerationResult(
            draft_response=response,
            model="DeterministicResponseGenerator",
            grounded=bool(historical_response),
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
    def _clean_historical_response(response: str) -> str:
        """
        Remove obvious Twitter-specific artifacts from historical replies.

        This method intentionally performs only conservative transformations.
        It must not invent support policies or customer-specific information.
        """

        if not response:
            return ""

        cleaned = response.strip()

        # Remove leading Twitter mentions such as:
        # @176370
        while cleaned.startswith("@"):
            parts = cleaned.split(maxsplit=1)

            if len(parts) != 2:
                break

            mention = parts[0]

            # Only remove simple numeric Twitter-style handles.
            if mention[1:].isdigit():
                cleaned = parts[1].strip()
            else:
                break

        # Remove historical agent signatures such as:
        # ^EM
        # ^AS
        # ^GD
        import re

        cleaned = re.sub(
            r"\s+\^[A-Za-z]{1,4}\s*$",
            "",
            cleaned,
        ).strip()

        return cleaned

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