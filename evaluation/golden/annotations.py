import pandas as pd

from evaluation.golden.text_utils import normalize_text
from evaluation.golden.language import detect_language_bucket


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