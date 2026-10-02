from src.generation.models import GenerationInput, GenerationResult
from src.generation.prompts import SYSTEM_PROMPT, build_generation_prompt
from src.generation.provider import LLMProvider
from src.generation.response_cleaner import clean_model_response


class ResponseGenerator:
    """Generates grounded customer-support responses."""

    def __init__(self, provider: LLMProvider):
        self.provider = provider

    def generate(self, data: GenerationInput) -> GenerationResult:
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