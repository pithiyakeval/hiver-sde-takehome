import pandas as pd

from evaluation.golden.text_utils import (
    normalize_text,
    word_count,
    has_url,
    has_many_mentions,
    has_many_punctuation,
    contains_order_number,
    contains_contact_information,
)
from evaluation.golden.intent_signals import signal_intents
from evaluation.golden.language import detect_language_bucket


def bucket_priority_score(
    row: pd.Series,
    bucket: str,
) -> float:
    """
    Score candidates within a bucket.

    Higher score means more useful for annotation.
    """
    text = str(row["customer_text"])
    normalized = normalize_text(text)

    wc = word_count(normalized)
    matched_intents = signal_intents(normalized)

    score = 0.0

    # Prefer readable examples.
    if 12 <= wc <= 45:
        score += 3

    # Avoid extremely long tweets.
    if wc > 100:
        score -= 2

    # Avoid almost-empty examples except for short/noisy bucket.
    if wc <= 2 and bucket != "short_noisy":
        score -= 3

    if bucket == "common_normal":
        if len(matched_intents) == 1:
            score += 5

    elif bucket == "boundary_case":
        if len(matched_intents) >= 2:
            score += 8

        if contains_order_number(text):
            score += 1

    elif bucket == "short_noisy":
        if wc <= 7:
            score += 5

        if has_url(text):
            score += 2

        if has_many_mentions(text):
            score += 2

        if has_many_punctuation(text):
            score += 2

    elif bucket == "multilingual":
        language = detect_language_bucket(normalized)

        if language != "english_or_unknown":
            score += 7

    elif bucket == "multi_intent":
        if len(matched_intents) >= 2:
            score += 8

        if len(matched_intents) >= 3:
            score += 3

    elif bucket == "likely_escalation":
        score += 5

        if contains_contact_information(text):
            score += 1

    elif bucket == "other_unclear_candidate":
        if len(matched_intents) == 0:
            score += 7

        if wc <= 5:
            score += 3

    # Slight preference for messages with richer context.
    if 15 <= wc <= 50:
        score += 1

    return score