import pandas as pd

from evaluation.intents.discovery_config import INPUT_PATH
from evaluation.intents.preprocessing import clean_text

def load_data():
    print("=" * 70)
    print("AMAZONHELP INTENT DISCOVERY")
    print("=" * 70)

    print("\nInput:")
    print(INPUT_PATH)

    if not INPUT_PATH.exists():
        raise FileNotFoundError(
            f"\nConversation file not found:\n{INPUT_PATH}\n\n"
            "Run this first:\n"
            "python -m src.data.conversations"
        )

    df = pd.read_csv(
        INPUT_PATH,
        low_memory=False,
    )

    print(
        f"\nConversation/reply rows loaded: "
        f"{len(df):,}"
    )

    required_columns = [
        "customer_tweet_id",
        "customer_author_id",
        "customer_created_at",
        "customer_text",
    ]

    missing = [
        column
        for column in required_columns
        if column not in df.columns
    ]

    if missing:
        raise ValueError(
            f"Missing required columns: {missing}"
        )

    return df  

def prepare_customer_messages(df):
    """
    Keep one row per customer tweet.

    A customer tweet may have more than one AmazonHelp reply.
    For intent discovery, the customer tweet itself is the unit
    of analysis.
    """

    customer_df = (
        df[
            [
                "customer_tweet_id",
                "customer_author_id",
                "customer_created_at",
                "customer_text",
            ]
        ]
        .drop_duplicates(
            subset="customer_tweet_id"
        )
        .copy()
    )

    customer_df["customer_text"] = (
        customer_df["customer_text"]
        .fillna("")
        .astype(str)
        .str.strip()
    )

    # Remove empty customer messages.
    customer_df = customer_df[
        customer_df["customer_text"].str.len() > 0
    ].copy()

    # Analysis text.
    customer_df["clean_text"] = (
        customer_df["customer_text"]
        .map(clean_text)
    )

    # Remove empty cleaned messages.
    customer_df = customer_df[
        customer_df["clean_text"].str.len() > 0
    ].copy()

    # Very short messages are generally poor candidates
    # for intent discovery.
    customer_df = customer_df[
        customer_df["clean_text"].str.len() >= 10
    ].copy()

    customer_df["text_length"] = (
        customer_df["clean_text"].str.len()
    )

    customer_df = customer_df.reset_index(
        drop=True
    )

    print(
        f"Unique customer tweets available: "
        f"{len(customer_df):,}"
    )

    return customer_df 