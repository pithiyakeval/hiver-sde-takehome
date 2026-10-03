from src.generation.config import GenerationConfig
from src.generation.ollama_provider import OllamaProvider


def test_ollama_provider_uses_configuration():
    config = GenerationConfig(
        model="test-model",
        host="http://localhost:11434",
        temperature=0.2,
        max_tokens=64,
    )

    provider = OllamaProvider(config)

    assert provider.config.model == "test-model"
    assert provider.config.host == "http://localhost:11434"
    assert provider.config.temperature == 0.2
    assert provider.config.max_tokens == 64