from dataclasses import dataclass
import os


@dataclass(frozen=True)
class GenerationConfig:
    model: str = "ministral-3:3b"
    host: str = "http://localhost:11434"
    temperature: float = 0.1
    max_tokens: int = 48

    @classmethod
    def from_environment(cls):
        return cls(
            model=os.getenv("OLLAMA_MODEL", cls.model),
            host=os.getenv("OLLAMA_HOST_URL", cls.host),
            temperature=float(
                os.getenv("LLM_TEMPERATURE", cls.temperature)
            ),
            max_tokens=int(
                os.getenv("LLM_MAX_TOKENS", cls.max_tokens)
            ),
        )