from src.generation.generator import ResponseGenerator
from src.generation.mock_provider import MockLLMProvider
from src.generation.models import GenerationInput, RetrievedExample
from src.generation.response_cleaner import clean_model_response

def test_generation_returns_grounded_response():
    provider = MockLLMProvider(
        response="Thanks for reaching out. We’re happy to help with your order."
    )

    generator = ResponseGenerator(provider)

    data = GenerationInput(
        customer_message="Where is my order?",
        intent="order_status",
        retrieved_examples=[
            RetrievedExample(
                customer_text="Can you tell me where my package is?",
                historical_response="We'd be happy to help track your order.",
                similarity=0.82,
            )
        ],
    )

    result = generator.generate(data)

    assert result.draft_response == (
        "Thanks for reaching out. We’re happy to help with your order."
    )
    assert result.grounded is True
    assert result.model == "MockLLMProvider"


def test_generation_without_evidence_is_not_grounded():
    provider = MockLLMProvider(response="Could you provide more details?")

    generator = ResponseGenerator(provider)

    data = GenerationInput(
        customer_message="I need help.",
        intent="other_or_unclear",
        retrieved_examples=[],
    )

    result = generator.generate(data)

    assert result.draft_response == "Could you provide more details?"
    assert result.grounded is False

def test_response_cleaner_removes_thinking_block():
    raw_response = """
    <think>
    The customer has a delayed delivery.
    I should apologize and provide a helpful response.
    </think>
    We’re sorry your order is delayed. Please check the latest tracking update.
    """

    cleaned = clean_model_response(raw_response)

    assert cleaned == (
        "We’re sorry your order is delayed. Please check the latest tracking update."
    )


def test_response_cleaner_preserves_clean_response():
    response = "Thanks for reaching out. We’re happy to help."

    assert clean_model_response(response) == response

def test_response_cleaner_removes_unclosed_thinking_block():
    raw_response = """
    <think>
    Internal reasoning that was truncated
    """

    assert clean_model_response(raw_response) == ""