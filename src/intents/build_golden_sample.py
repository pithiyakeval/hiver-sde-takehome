"""
Build a reproducible, annotation-ready golden-set candidate sample
for the AmazonHelp customer-support agent.

The script intentionally does NOT assign final intent labels.

It samples real AmazonHelp customer messages into several useful
evaluation buckets so that the final human-labelled golden set is
not dominated by easy/common examples.

Input:
    data/processed/amazonhelp_conversations.csv

Outputs:
    golden/golden_candidates.csv
    golden/golden_sampling_report.csv
    golden/labeling_notes.md

Design goals:
    - deterministic sampling
    - no duplicate customer tweets
    - preserve original customer text
    - include difficult/boundary cases
    - include multilingual examples
    - include short/noisy Twitter messages
    - include likely multi-intent messages
    - provide transparent sampling metadata
"""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd


# ============================================================
# Configuration
# ============================================================

INPUT_PATH = Path(
    "data/processed/amazonhelp_conversations.csv"
)

OUTPUT_DIR = Path("golden")

RANDOM_STATE = 42

# Required final golden-set size.
TARGET_SIZE = 200

# We deliberately oversample candidate examples slightly and
# then enforce a final deterministic selection.
CANDIDATE_POOL_MULTIPLIER = 4

# Maximum candidate pool size per sampling bucket.
MAX_BUCKET_CANDIDATES = 350

# ============================================================
# Sampling plan
# ============================================================

# Target distribution for the 200-example golden set.
#
# These are candidate buckets, NOT final intent labels.
# Human annotators still assign the actual intent later.

SAMPLING_TARGETS = {
    "common_normal": 100,
    "boundary_case": 40,
    "short_noisy": 20,
    "multilingual": 15,
    "multi_intent": 10,
    "likely_escalation": 10,
    "other_unclear_candidate": 5,
}


# ============================================================
# Text utilities
# ============================================================

URL_RE = re.compile(r"https?://\S+|www\.\S+", re.IGNORECASE)

MENTION_RE = re.compile(r"@\w+")

HASHTAG_RE = re.compile(r"#\w+")

ORDER_NUMBER_RE = re.compile(
    r"\b\d{3}-\d{7}-\d{7}\b"
)

PHONE_RE = re.compile(
    r"\b(?:\+?\d[\d\s().-]{7,}\d)\b"
)

EMAIL_RE = re.compile(
    r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b"
)

REPEATED_PUNCT_RE = re.compile(r"([!?])\1{2,}")

WHITESPACE_RE = re.compile(r"\s+")


def normalize_text(text: object) -> str:
    """Basic normalization used only for feature detection."""
    if pd.isna(text):
        return ""

    text = str(text)

    text = URL_RE.sub(" URL ", text)
    text = EMAIL_RE.sub(" EMAIL ", text)
    text = PHONE_RE.sub(" PHONE ", text)
    text = ORDER_NUMBER_RE.sub(" ORDER_NUMBER ", text)

    text = MENTION_RE.sub(" MENTION ", text)
    text = HASHTAG_RE.sub(" HASHTAG ", text)

    text = WHITESPACE_RE.sub(" ", text)

    return text.strip()


def word_count(text: str) -> int:
    return len(re.findall(r"\b[\w']+\b", text, flags=re.UNICODE))


def sentence_count(text: str) -> int:
    if not text.strip():
        return 0

    pieces = re.split(r"[.!?]+", text)
    return len([p for p in pieces if p.strip()])


def has_url(text: str) -> bool:
    return bool(URL_RE.search(text))


def has_many_mentions(text: str) -> bool:
    return len(MENTION_RE.findall(text)) >= 2


def has_many_punctuation(text: str) -> bool:
    return bool(REPEATED_PUNCT_RE.search(text))


def contains_order_number(text: str) -> bool:
    return bool(ORDER_NUMBER_RE.search(text))


def contains_contact_information(text: str) -> bool:
    return bool(
        EMAIL_RE.search(text)
        or PHONE_RE.search(text)
        or ORDER_NUMBER_RE.search(text)
    )


# ============================================================
# Language heuristics
# ============================================================

# This is intentionally conservative.
# We only need a useful multilingual sampling bucket.
#
# We do NOT use language as an intent.

LANGUAGE_PATTERNS = {
    "spanish": [
        r"\bque\b",
        r"\bcomo\b",
        r"\bpedido\b",
        r"\bpaquete\b",
        r"\bno\s+he\s+recibido\b",
        r"\brecibido\b",
        r"\bentrega\b",
        r"\bdonde\b",
        r"\bquiero\b",
        r"\bgracias\b",
        r"\bhola\b",
    ],
    "french": [
        r"\bbonjour\b",
        r"\bmerci\b",
        r"\bcolis\b",
        r"\bcommande\b",
        r"\blivraison\b",
        r"\breçu\b",
        r"\bpas\b",
        r"\bavec\b",
        r"\bje\b",
        r"\bvous\b",
    ],
    "german": [
        r"\bnicht\b",
        r"\bich\b",
        r"\bder\b",
        r"\bdie\b",
        r"\bdas\b",
        r"\bund\b",
        r"\bpaket\b",
        r"\blieferung\b",
        r"\bbestellung\b",
        r"\bdanke\b",
    ],
    "italian": [
        r"\bciao\b",
        r"\bgrazie\b",
        r"\bordine\b",
        r"\bconsegna\b",
        r"\bpacco\b",
        r"\bnon\b",
        r"\bperché\b",
        r"\bcome\b",
    ],
    "portuguese": [
        r"\bolá\b",
        r"\bobrigado\b",
        r"\bpedido\b",
        r"\bentrega\b",
        r"\bnão\b",
        r"\bporque\b",
        r"\bcomo\b",
    ],
    "japanese": [
        r"[\u3040-\u30ff]",
        r"[\u4e00-\u9faf]",
    ],
}


def detect_language_bucket(text: str) -> str:
    """
    Lightweight language bucket detector.

    This is not intended to be a production language classifier.
    It exists only to ensure the golden sample contains multilingual
    examples.
    """

    lower = text.lower()

    scores = {}

    for language, patterns in LANGUAGE_PATTERNS.items():
        score = 0

        for pattern in patterns:
            if re.search(pattern, lower, flags=re.IGNORECASE):
                score += 1

        scores[language] = score

    best_language = max(scores, key=scores.get)

    if scores[best_language] >= 2:
        return best_language

    return "english_or_unknown"


# ============================================================
# Intent-signal detection
# ============================================================

# IMPORTANT:
# These are NOT labels.
#
# They are lexical signals used to discover boundary and difficult
# examples for human annotation.

INTENT_SIGNALS = {
    "order_status": [
        "where is my order",
        "where is my package",
        "where's my package",
        "where is my parcel",
        "track my order",
        "tracking",
        "order status",
        "package status",
        "when will my order",
        "when will my package",
    ],
    "delivery_late": [
        "late",
        "delayed",
        "still hasn't arrived",
        "still hasnt arrived",
        "hasn't arrived",
        "hasnt arrived",
        "supposed to arrive",
        "supposed to be delivered",
        "was due",
        "overdue",
        "days late",
        "delivery date passed",
        "didn't arrive",
        "didnt arrive",
    ],
    "delivered_not_received": [
        "says delivered",
        "marked delivered",
        "shows delivered",
        "was delivered",
        "package was delivered",
        "order was delivered",
        "but i didn't receive",
        "but i did not receive",
        "never received",
        "not received",
        "nothing arrived",
        "not here",
    ],
    "delivery_promise": [
        "next day",
        "one day delivery",
        "same day",
        "two day delivery",
        "2 day delivery",
        "guaranteed delivery",
        "delivery guarantee",
        "prime delivery",
        "promised delivery",
        "guaranteed date",
    ],
    "missing_or_wrong_item": [
        "wrong item",
        "wrong product",
        "incorrect item",
        "missing item",
        "item missing",
        "missing product",
        "not what i ordered",
        "different product",
        "received the wrong",
    ],
    "damaged_item_or_package": [
        "damaged",
        "broken",
        "cracked",
        "damaged package",
        "damaged box",
        "broken item",
        "arrived broken",
        "packaging",
        "opened package",
        "crushed",
    ],
    "return_or_refund": [
        "refund",
        "refunded",
        "return",
        "returning",
        "send it back",
        "money back",
        "reimbursement",
        "cancelled and refund",
        "cancelled but no refund",
    ],
    "payment_or_charge": [
        "charged",
        "charge",
        "payment",
        "credit card",
        "debit card",
        "card",
        "billing",
        "charged twice",
        "don't recognize this charge",
        "do not recognize this charge",
        "payment failed",
        "payment verification",
    ],
    "prime_membership": [
        "prime membership",
        "prime member",
        "amazon prime",
        "prime subscription",
        "prime renewal",
        "prime fee",
        "prime charge",
        "cancel prime",
        "prime student",
        "prime music",
        "prime video",
    ],
    "account_or_access": [
        "account",
        "password",
        "login",
        "log in",
        "logged out",
        "account blocked",
        "account locked",
        "hacked",
        "email changed",
        "someone changed",
        "can't access",
        "cannot access",
    ],
    "product_or_device_help": [
        "echo",
        "alexa",
        "fire tablet",
        "fire tv",
        "kindle",
        "device",
        "not working",
        "doesn't work",
        "doesnt work",
        "compatible",
        "compatibility",
        "how do i use",
        "how does this work",
    ],
    "customer_service_followup": [
        "still waiting",
        "waiting for a reply",
        "waiting for response",
        "no response",
        "no reply",
        "nobody replied",
        "haven't heard back",
        "havent heard back",
        "call me",
        "called me",
        "get back to me",
        "follow up",
        "follow-up",
        "update",
    ],
}


def signal_intents(text: str) -> list[str]:
    """Return all intent areas with at least one lexical signal."""

    lower = text.lower()

    matched = []

    for intent, phrases in INTENT_SIGNALS.items():
        for phrase in phrases:
            if phrase in lower:
                matched.append(intent)
                break

    return matched


# ============================================================
# Bucket detection
# ============================================================

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
        (
            "order_status",
            "delivery_late",
        ),
        (
            "delivery_late",
            "delivery_promise",
        ),
        (
            "delivery_late",
            "delivered_not_received",
        ),
        (
            "missing_or_wrong_item",
            "damaged_item_or_package",
        ),
        (
            "return_or_refund",
            "payment_or_charge",
        ),
        (
            "prime_membership",
            "payment_or_charge",
        ),
        (
            "customer_service_followup",
            "delivery_late",
        ),
        (
            "customer_service_followup",
            "return_or_refund",
        ),
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


# ============================================================
# Scoring
# ============================================================

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


# ============================================================
# Deduplication
# ============================================================

def deduplicate_messages(df: pd.DataFrame) -> pd.DataFrame:
    """
    Remove duplicate customer tweets.

    Primary key:
        customer_tweet_id

    Secondary text normalization prevents duplicate content from
    consuming multiple golden slots.
    """

    required_columns = [
        "customer_tweet_id",
        "customer_text",
    ]

    missing = [
        col for col in required_columns
        if col not in df.columns
    ]

    if missing:
        raise ValueError(
            f"Input is missing required columns: {missing}"
        )

    df = df.copy()

    df["customer_tweet_id"] = (
        df["customer_tweet_id"]
        .astype(str)
    )

    df["customer_text"] = (
        df["customer_text"]
        .fillna("")
        .astype(str)
        .str.strip()
    )

    df = df[
        df["customer_text"].str.len() > 0
    ].copy()

    df["normalized_text"] = (
        df["customer_text"]
        .map(normalize_text)
        .str.lower()
        .str.strip()
    )

    # First remove duplicate customer tweet IDs.
    df = (
        df
        .drop_duplicates(
            subset=["customer_tweet_id"],
            keep="first",
        )
    )

    # Then prevent exact normalized duplicates.
    df = (
        df
        .drop_duplicates(
            subset=["normalized_text"],
            keep="first",
        )
    )

    return df


# ============================================================
# Candidate generation
# ============================================================

def build_bucket_candidates(
    df: pd.DataFrame,
) -> dict[str, pd.DataFrame]:
    """
    Build ranked candidate pools for every sampling bucket.
    """

    bucket_rows = {
        bucket: []
        for bucket in SAMPLING_TARGETS
    }

    print("\nDetecting sampling buckets...")

    for idx, row in df.iterrows():
        buckets = classify_sampling_buckets(row)

        for bucket in buckets:
            bucket_rows[bucket].append(idx)

    candidates = {}

    for bucket, indices in bucket_rows.items():

        if not indices:
            candidates[bucket] = df.iloc[0:0].copy()
            continue

        bucket_df = df.loc[indices].copy()

        bucket_df["sampling_score"] = bucket_df.apply(
            lambda row: bucket_priority_score(
                row,
                bucket,
            ),
            axis=1,
        )

        bucket_df = (
            bucket_df
            .sort_values(
                [
                    "sampling_score",
                    "customer_tweet_id",
                ],
                ascending=[False, True],
            )
            .head(MAX_BUCKET_CANDIDATES)
            .copy()
        )

        bucket_df["sampling_bucket"] = bucket

        candidates[bucket] = bucket_df

    return candidates


# ============================================================
# Final deterministic selection
# ============================================================

def select_final_candidates(
    candidates: dict[str, pd.DataFrame],
) -> pd.DataFrame:
    """
    Select the final 200 candidate examples while avoiding duplicate
    customer tweets.

    We use the requested bucket quotas in priority order.

    Hard/difficult buckets are selected first so that common examples
    cannot consume the entire sample.
    """

    # Hard buckets first.
    selection_order = [
        "boundary_case",
        "multilingual",
        "multi_intent",
        "likely_escalation",
        "short_noisy",
        "other_unclear_candidate",
        "common_normal",
    ]

    selected_rows = []
    selected_ids = set()

    remaining_target = TARGET_SIZE

    # Track actual requested quotas.
    quotas = dict(SAMPLING_TARGETS)

    for bucket in selection_order:

        if remaining_target <= 0:
            break

        bucket_df = candidates.get(bucket)

        if bucket_df is None or bucket_df.empty:
            print(
                f"WARNING: no candidates available for "
                f"'{bucket}'."
            )
            continue

        target = min(
            quotas[bucket],
            remaining_target,
        )

        count = 0

        for _, row in bucket_df.iterrows():

            tweet_id = str(
                row["customer_tweet_id"]
            )

            if tweet_id in selected_ids:
                continue

            selected_ids.add(tweet_id)

            selected_rows.append(row)

            count += 1

            if count >= target:
                break

        print(
            f"{bucket:25s}: "
            f"selected {count:3d} / {target:3d}"
        )

        remaining_target -= count

    # --------------------------------------------------------
    # Fill any remaining slots using unused common candidates.
    # --------------------------------------------------------

    if remaining_target > 0:

        print(
            f"\nFilling remaining {remaining_target} "
            "slots from unused candidates..."
        )

        all_candidates = []

        for bucket_df in candidates.values():

            if bucket_df is None or bucket_df.empty:
                continue

            all_candidates.append(bucket_df)

        if all_candidates:

            combined = pd.concat(
                all_candidates,
                ignore_index=True,
            )

            combined = (
                combined
                .sort_values(
                    [
                        "sampling_score",
                        "customer_tweet_id",
                    ],
                    ascending=[False, True],
                )
            )

            for _, row in combined.iterrows():

                tweet_id = str(
                    row["customer_tweet_id"]
                )

                if tweet_id in selected_ids:
                    continue

                selected_ids.add(tweet_id)

                # Preserve the bucket in which this row was
                # originally selected.
                row = row.copy()

                selected_rows.append(row)

                remaining_target -= 1

                if remaining_target <= 0:
                    break

    if not selected_rows:
        raise RuntimeError(
            "No golden candidates could be selected."
        )

    result = pd.DataFrame(selected_rows)

    # --------------------------------------------------------
    # Final deduplication.
    # --------------------------------------------------------

    result = (
        result
        .drop_duplicates(
            subset=["customer_tweet_id"],
            keep="first",
        )
    )

    # --------------------------------------------------------
    # Deterministic ordering.
    # --------------------------------------------------------

    result = (
        result
        .sort_values(
            "customer_tweet_id"
        )
        .reset_index(drop=True)
    )

    # --------------------------------------------------------
    # Assign stable example IDs.
    # --------------------------------------------------------

    result.insert(
        0,
        "example_id",
        [
            f"G{i:03d}"
            for i in range(
                1,
                len(result) + 1,
            )
        ],
    )

    return result


# ============================================================
# Annotation columns
# ============================================================

def add_annotation_columns(
    df: pd.DataFrame,
) -> pd.DataFrame:

    df = df.copy()

    # These columns intentionally start blank.
    # They must be filled by human annotation.

    df["intent"] = ""

    df["confidence"] = ""

    df["is_ambiguous"] = ""

    df["ambiguity_reason"] = ""

    df["language"] = df["customer_text"].map(
        lambda x: detect_language_bucket(
            normalize_text(x)
        )
    )

    df["expected_action"] = ""

    df["should_escalate"] = ""

    df["escalation_reason"] = ""

    df["annotator"] = ""

    # Keep useful sampling metadata separate from labels.
    #
    # This helps us audit how the golden set was created.

    return df


# ============================================================
# Sampling report
# ============================================================

def build_sampling_report(
    candidates: dict[str, pd.DataFrame],
    final_df: pd.DataFrame,
) -> pd.DataFrame:

    rows = []

    for bucket, target in SAMPLING_TARGETS.items():

        available = len(
            candidates.get(
                bucket,
                pd.DataFrame(),
            )
        )

        selected = int(
            (
                final_df["sampling_bucket"]
                == bucket
            ).sum()
        )

        rows.append(
            {
                "sampling_bucket": bucket,
                "target_count": target,
                "available_candidates": available,
                "selected_count": selected,
                "target_met": selected >= target,
            }
        )

    # Add overall row.
    rows.append(
        {
            "sampling_bucket": "TOTAL",
            "target_count": TARGET_SIZE,
            "available_candidates": sum(
                len(v)
                for v in candidates.values()
                if v is not None
            ),
            "selected_count": len(final_df),
            "target_met": len(final_df) >= TARGET_SIZE,
        }
    )

    return pd.DataFrame(rows)


# ============================================================
# Labeling notes
# ============================================================
# ============================================================
# Labeling notes
# ============================================================

def write_labeling_notes(report, final_df):

    output_path = OUTPUT_DIR / "labeling_notes.md"

    lines = []

    lines.append("# Golden Set Sampling Notes")
    lines.append("")
    lines.append("## Dataset")
    lines.append("")
    lines.append(
        "The golden-set candidates were sampled from "
        "`data/processed/amazonhelp_conversations.csv`."
    )
    lines.append("")
    lines.append(
        "The source contains AmazonHelp customer-support conversation pairs."
    )
    lines.append("")
    lines.append(
        "The sampling process first deduplicates customer messages "
        "and then selects examples using deterministic sampling rules."
    )
    lines.append("")

    lines.append("## Important distinction")
    lines.append("")
    lines.append(
        "This file contains **candidate examples for human annotation**."
    )
    lines.append("")
    lines.append(
        "The sampling buckets are **not intent labels**."
    )
    lines.append("")
    lines.append(
        "Final intent labels must be assigned by a human using "
        "`golden/intent_taxonomy.md`."
    )
    lines.append("")
    lines.append(
        "The model must not be allowed to determine the golden labels."
    )
    lines.append("")

    lines.append("## Reproducibility")
    lines.append("")
    lines.append("Random seed:")
    lines.append("")
    lines.append(f"`{RANDOM_STATE}`")
    lines.append("")
    lines.append("Target golden-set size:")
    lines.append("")
    lines.append(f"`{TARGET_SIZE}`")
    lines.append("")
    lines.append("Sampling script:")
    lines.append("")
    lines.append("`src/intents/build_golden_sample.py`")
    lines.append("")
    lines.append("Run:")
    lines.append("")
    lines.append("`python -m src.intents.build_golden_sample`")
    lines.append("")

    lines.append("## Sampling strategy")
    lines.append("")
    lines.append(
        "The candidate set deliberately contains:"
    )
    lines.append("")
    lines.append("1. Common, clearly expressed customer problems")
    lines.append("2. Boundary cases between similar intents")
    lines.append("3. Short/noisy Twitter messages")
    lines.append("4. Multilingual examples")
    lines.append("5. Multi-intent messages")
    lines.append("6. Likely escalation cases")
    lines.append("7. Ambiguous/unclear examples")
    lines.append("")

    lines.append(
        "This is preferable to a purely random sample because "
        "a golden evaluation set should test both ordinary behavior "
        "and failure-prone cases."
    )
    lines.append("")

    lines.append("## Sampling distribution")
    lines.append("")
    lines.append(
        "| Bucket | Target | Available candidates | Selected | Target met |"
    )
    lines.append(
        "|---|---:|---:|---:|---|"
    )

    for _, row in report.iterrows():

        bucket = row["sampling_bucket"]

        target = int(row["target_count"])
        available = int(row["available_candidates"])
        selected = int(row["selected_count"])

        target_met = "Yes" if row["target_met"] else "No"

        lines.append(
            f"| `{bucket}` | {target} | {available} | "
            f"{selected} | {target_met} |"
        )

    lines.append("")

    lines.append("## Human annotation")
    lines.append("")
    lines.append(
        "Annotators should use the taxonomy and boundary rules in:"
    )
    lines.append("")
    lines.append("`golden/intent_taxonomy.md`")
    lines.append("")
    lines.append("The following fields must be completed manually:")
    lines.append("")
    lines.append("- `intent`")
    lines.append("- `confidence`")
    lines.append("- `is_ambiguous`")
    lines.append("- `ambiguity_reason`")
    lines.append("- `expected_action`")
    lines.append("- `should_escalate`")
    lines.append("- `escalation_reason`")
    lines.append("- `annotator`")
    lines.append("")
    lines.append(
        "The original `customer_text` must not be rewritten."
    )
    lines.append("")

    lines.append("## Golden-set integrity")
    lines.append("")
    lines.append(
        "After annotation, the golden set should be frozen and "
        "kept separate from model training data."
    )
    lines.append("")
    lines.append(
        "No golden example should be used to train or tune the "
        "final classifier after it becomes part of the held-out "
        "evaluation set."
    )
    lines.append("")
    lines.append(
        "If taxonomy definitions change, affected examples should "
        "be re-reviewed and the taxonomy version recorded in the "
        "evaluation results."
    )
    lines.append("")

    output_path.write_text(
        "\n".join(lines),
        encoding="utf-8",
    )


# ============================================================
# Validation
# ============================================================

def validate_final_set(final_df):

    print("\n" + "=" * 70)
    print("FINAL GOLDEN CANDIDATE VALIDATION")
    print("=" * 70)

    print(f"Examples selected: {len(final_df)}")

    print(
        "Unique customer tweet IDs:",
        final_df["customer_tweet_id"].nunique(),
    )

    print(
        "Duplicate customer tweet IDs:",
        final_df["customer_tweet_id"].duplicated().sum(),
    )

    print(
        "Empty customer texts:",
        final_df["customer_text"]
        .str.strip()
        .eq("")
        .sum(),
    )

    print("\nSampling distribution:")

    print(
        final_df["sampling_bucket"]
        .value_counts()
        .to_string()
    )

    print("\nLanguage distribution:")

    print(
        final_df["language"]
        .value_counts()
        .to_string()
    )

    assert (
        final_df["customer_tweet_id"].nunique()
        == len(final_df)
    ), "Duplicate customer tweet IDs found."

    assert (
        final_df["customer_text"]
        .str.strip()
        .ne("")
        .all()
    ), "Empty customer text found."

    assert len(final_df) == TARGET_SIZE, (
        f"Expected {TARGET_SIZE} examples, "
        f"got {len(final_df)}."
    )

    print("\nValidation: PASSED")


# ============================================================
# Main
# ============================================================

def main():

    np.random.seed(RANDOM_STATE)

    print("=" * 70)
    print("AMAZONHELP GOLDEN-SET CANDIDATE BUILDER")
    print("=" * 70)

    print(f"\nInput: {INPUT_PATH}")
    print(f"Output directory: {OUTPUT_DIR}")
    print(f"Random seed: {RANDOM_STATE}")
    print(f"Target size: {TARGET_SIZE}")

    if not INPUT_PATH.exists():
        raise FileNotFoundError(
            f"Input file not found: {INPUT_PATH}"
        )

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    # --------------------------------------------------------
    # Load data
    # --------------------------------------------------------

    print("\nLoading conversation data...")

    df = pd.read_csv(
        INPUT_PATH,
        low_memory=False,
    )

    print(
        f"Loaded {len(df):,} conversation rows."
    )

    # --------------------------------------------------------
    # Deduplicate
    # --------------------------------------------------------

    print("\nDeduplicating customer messages...")

    df = deduplicate_messages(df)

    print(
        f"Usable unique customer messages: "
        f"{len(df):,}"
    )

    # --------------------------------------------------------
    # Build candidate buckets
    # --------------------------------------------------------

    candidates = build_bucket_candidates(df)

    # --------------------------------------------------------
    # Candidate availability
    # --------------------------------------------------------

    print("\nCandidate availability:")

    for bucket in SAMPLING_TARGETS:

        count = len(
            candidates.get(
                bucket,
                pd.DataFrame(),
            )
        )

        target = SAMPLING_TARGETS[bucket]

        print(
            f"{bucket:25s}: "
            f"{count:4d} available "
            f"(target {target})"
        )

    # --------------------------------------------------------
    # Select final candidates
    # --------------------------------------------------------

    print("\nSelecting final candidate set...")

    final_df = select_final_candidates(
        candidates
    )

    # --------------------------------------------------------
    # Add human annotation columns
    # --------------------------------------------------------

    final_df = add_annotation_columns(
        final_df
    )

    # --------------------------------------------------------
    # Keep clean output columns
    # --------------------------------------------------------

    preferred_columns = [
        "example_id",
        "customer_tweet_id",
        "customer_text",

        # Human annotation fields.
        "intent",
        "confidence",
        "is_ambiguous",
        "ambiguity_reason",
        "language",
        "expected_action",
        "should_escalate",
        "escalation_reason",
        "annotator",

        # Sampling audit fields.
        "sampling_bucket",
        "sampling_score",
    ]

    final_df = final_df[
        [
            col
            for col in preferred_columns
            if col in final_df.columns
        ]
    ]

    # --------------------------------------------------------
    # Validate
    # --------------------------------------------------------

    validate_final_set(
        final_df
    )

    # --------------------------------------------------------
    # Save candidate CSV
    # --------------------------------------------------------

    candidate_path = (
        OUTPUT_DIR
        / "golden_candidates.csv"
    )

    final_df.to_csv(
        candidate_path,
        index=False,
        encoding="utf-8-sig",
    )

    print(
        f"\nSaved candidate set:\n"
        f"{candidate_path.resolve()}"
    )

    # --------------------------------------------------------
    # Sampling report
    # --------------------------------------------------------

    report = build_sampling_report(
        candidates,
        final_df,
    )

    report_path = (
        OUTPUT_DIR
        / "golden_sampling_report.csv"
    )

    report.to_csv(
        report_path,
        index=False,
        encoding="utf-8-sig",
    )

    print(
        f"Saved sampling report:\n"
        f"{report_path.resolve()}"
    )

    # --------------------------------------------------------
    # Labeling notes
    # --------------------------------------------------------

    write_labeling_notes(
        report,
        final_df,
    )

    notes_path = (
        OUTPUT_DIR
        / "labeling_notes.md"
    )

    print(
        f"Saved labeling notes:\n"
        f"{notes_path.resolve()}"
    )

    # --------------------------------------------------------
    # Final summary
    # --------------------------------------------------------

    print("\n" + "=" * 70)
    print("GOLDEN CANDIDATE SET READY")
    print("=" * 70)
    
    print(
        f"\n{len(final_df)} examples ready "
        "for human annotation."
    )

    print(
        "\nNext step:"
        "\nOpen golden/golden_candidates.csv "
        "and label the examples using "
        "golden/intent_taxonomy.md."
    )

    print(
        "\nIMPORTANT:"
        "\nDo not train the classifier on this file."
        "\nThis will become our held-out golden evaluation set."
    )


if __name__ == "__main__":
    main()