import re
import pandas as pd

from evaluation.golden.text_utils import (
    normalize_text,
    word_count,
    sentence_count,
    has_url,
    has_many_mentions,
    has_many_punctuation,
)
from evaluation.golden.language import detect_language_bucket
from evaluation.golden.intent_signals import signal_intents


def classify_sampling_buckets(row: pd.Series) -> set[str]:
    """
    Determine which evaluation sampling buckets a message is
    eligible for.
    A message can belong to multiple buckets.
    """
    text = str(row["customer_text"])
    normalized = normalize_text(text)
    lower = normalized.lower()

    buckets = set()

    wc = word_count(normalized)
    sc = sentence_count(normalized)

    matched_intents = signal_intents(normalized)

    # --------------------------------------------------------
    # Short / noisy
    # --------------------------------------------------------

    if wc <= 7:
        buckets.add("short_noisy")

    if has_many_punctuation(text):
        buckets.add("short_noisy")

    if has_many_mentions(text):
        buckets.add("short_noisy")

    if has_url(text):
        buckets.add("short_noisy")

    # --------------------------------------------------------
    # Multilingual
    # --------------------------------------------------------

    language_bucket = detect_language_bucket(normalized)

    if language_bucket != "english_or_unknown":
        buckets.add("multilingual")

    # --------------------------------------------------------
    # Multi-intent
    # --------------------------------------------------------

    if len(matched_intents) >= 2:
        buckets.add("multi_intent")

    if sc >= 3 and len(matched_intents) >= 2:
        buckets.add("multi_intent")

    # Explicit conjunction patterns are useful multi-intent signals.
    multi_patterns = [
        r"\band\b",
        r"\balso\b",
        r"\band now\b",
        r"\bbut\b",
        r"\bplus\b",
        r"\bas well as\b",
    ]

    if len(matched_intents) >= 2:
        for pattern in multi_patterns:
            if re.search(pattern, lower):
                buckets.add("multi_intent")
                break

    # --------------------------------------------------------
    # Boundary cases
    # --------------------------------------------------------

    boundary_pairs = [
        ("order_status", "delivery_late"),
        ("delivery_late", "delivery_promise"),
        ("delivery_late", "delivered_not_received"),
        ("missing_or_wrong_item", "damaged_item_or_package"),
        ("return_or_refund", "payment_or_charge"),
        ("prime_membership", "payment_or_charge"),
        ("customer_service_followup", "delivery_late"),
        ("customer_service_followup", "return_or_refund"),
    ]

    matched_set = set(matched_intents)

    for a, b in boundary_pairs:
        if a in matched_set and b in matched_set:
            buckets.add("boundary_case")
            break

    # Messages with multiple possible operational signals are
    # inherently valuable for boundary review.
    if len(matched_intents) >= 2:
        buckets.add("boundary_case")

    # --------------------------------------------------------
    # Likely escalation
    # --------------------------------------------------------

    escalation_patterns = [
        "hacked",
        "someone changed",
        "unauthorized",
        "fraud",
        "stolen",
        "police",
        "legal",
        "lawsuit",
        "charge i don't recognize",
        "charge I don't recognize",
        "account blocked",
        "account locked",
        "identity",
        "security",
        "fraudulent",
        "scammed",
        "scam",
        "money stolen",
        "missing money",
        "refund denied",
        "still no refund",
        "nobody will help",
        "no one will help",
        "multiple times",
        "ten transfers",
        "still unresolved",
    ]

    if any(pattern in lower for pattern in escalation_patterns):
        buckets.add("likely_escalation")

    # Strong unresolved/support-history signals.
    if (
        "still waiting" in lower
        or "no response" in lower
        or "no reply" in lower
        or "haven't heard back" in lower
        or "havent heard back" in lower
    ):
        buckets.add("likely_escalation")

    # --------------------------------------------------------
    # Other / unclear candidates
    # --------------------------------------------------------

    # Very short messages with no identifiable operational signal.
    if wc <= 5 and len(matched_intents) == 0:
        buckets.add("other_unclear_candidate")

    # Generic help/questions without enough operational detail.
    generic_phrases = [
        "can you help",
        "please help",
        "help me",
        "what is this",
        "what's this",
        "why",
        "hello",
        "hi",
        "thanks",
        "thank you",
        "anyone there",
    ]

    if any(phrase in lower for phrase in generic_phrases):
        if len(matched_intents) == 0:
            buckets.add("other_unclear_candidate")

    # --------------------------------------------------------
    # Common normal
    # --------------------------------------------------------

    # A message with exactly one clear operational signal and
    # reasonable length is a good normal example.
    if (
        len(matched_intents) == 1
        and 8 <= wc <= 60
        and "boundary_case" not in buckets
        and "multi_intent" not in buckets
        and "likely_escalation" not in buckets
    ):
        buckets.add("common_normal")

    return buckets