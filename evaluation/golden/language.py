import re


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