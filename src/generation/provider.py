from abc import ABC, abstractmethod


class LLMProvider(ABC):
    """Abstract interface for an LLM used by the support agent."""

    @abstractmethod
    def generate(self, system_prompt: str, user_prompt: str) -> str:
        """Generate a customer-facing response."""
        raise NotImplementedError