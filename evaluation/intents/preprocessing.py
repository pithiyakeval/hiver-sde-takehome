import re
import pandas as pd

from evaluation.intents.discovery_config import CONTRACTION_REPLACEMENTS


def clean_text(text):
    """
    Clean customer text for topic discovery.

    The original customer_text is never overwritten.
    This function creates only the analysis representation.
    """

    if pd.isna(text):
        return ""

    text = str(text)

    # Lowercase.
    text = text.lower()

    # Expand common contractions before removing punctuation.
    for old, new in CONTRACTION_REPLACEMENTS.items():
        text = text.replace(old, new)

    # Remove URLs.
    text = re.sub(
        r"https?://\S+|www\.\S+",
        " ",
        text,
    )

    # Remove Twitter mentions.
    text = re.sub(
        r"@\w+",
        " ",
        text,
    )

    # Remove hashtags but preserve the word.
    text = re.sub(
        r"#(\w+)",
        r"\1",
        text,
    )

    # Remove email addresses.
    text = re.sub(
        r"\S+@\S+",
        " ",
        text,
    )

    # Remove HTML-ish entities.
    text = re.sub(
        r"&\w+;",
        " ",
        text,
    )

    # Keep alphabetic characters and spaces.
    text = re.sub(
        r"[^a-z\s]",
        " ",
        text,
    )

    # Normalize whitespace.
    text = re.sub(
        r"\s+",
        " ",
        text,
    )

    return text.strip()