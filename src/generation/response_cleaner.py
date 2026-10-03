import re


def clean_model_response(response: str) -> str:
    """Remove model reasoning blocks and return customer-facing text."""

    cleaned = re.sub(
        r"<think>.*?</think>",
        "",
        response,
        flags=re.DOTALL | re.IGNORECASE,
    )

    # If the model started a reasoning block but did not close it,
    # discard everything from <think> onward.
    cleaned = re.sub(
        r"<think>.*$",
        "",
        cleaned,
        flags=re.DOTALL | re.IGNORECASE,
    )

    return cleaned.strip()