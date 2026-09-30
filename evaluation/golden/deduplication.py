import pandas as pd

from evaluation.golden.text_utils import normalize_text


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