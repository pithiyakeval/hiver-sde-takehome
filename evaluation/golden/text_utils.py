import re
import pandas as pd


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