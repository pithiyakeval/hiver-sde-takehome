import ollama

from src.generation.config import GenerationConfig
from src.generation.provider import LLMProvider


class OllamaProvider(LLMProvider):
    """LLM provider backed by a local Ollama model."""

    def __init__(
        self,
        config: GenerationConfig | None = None,
    ) -> None:
        self.config = config or GenerationConfig.from_environment()
        self.client = ollama.Client(
            host=self.config.host,
        )

    def generate(
        self,
        system_prompt: str,
        user_prompt: str,
    ) -> str:
        response = self.client.chat(
            model=self.config.model,
            messages=[
                {
                    "role": "system",
                    "content": system_prompt,
                },
                {
                    "role": "user",
                    "content": user_prompt,
                },
            ],
            options={
                "temperature": self.config.temperature,
                "num_predict": self.config.max_tokens,
            },
        )

        return response["message"]["content"]