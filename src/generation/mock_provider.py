from src.generation.provider import LLMProvider


class MockLLMProvider(LLMProvider):
    """Deterministic provider used for local development and tests."""

    def __init__(self, response: str = "Thank you for contacting us. We’re happy to help."):
        self.response = response

    def generate(self, system_prompt: str, user_prompt: str) -> str:
        return self.response