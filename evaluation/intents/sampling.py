import pandas as pd

from evaluation.intents.discovery_config import (
    DISCOVERY_SAMPLE_SIZE,
    RANDOM_STATE,
    OUTPUT_DIR,
    SAMPLE_PATH,
)


def create_discovery_sample(customer_df):
    """
    Build a reproducible stratified sample.

    We stratify by:
        1. message length
        2. approximate message volume

    Length stratification prevents the discovery set from being
    dominated by one particular style of Twitter message.
    """

    print("\n" + "=" * 70)
    print("CREATING DISCOVERY SAMPLE")
    print("=" * 70)

    df = customer_df.copy()

    target_size = min(
        DISCOVERY_SAMPLE_SIZE,
        len(df),
    )

    # Five quantile-based length groups.
    df["length_bucket"] = pd.qcut(
        df["text_length"],
        q=5,
        labels=[
            "very_short",
            "short",
            "medium",
            "long",
            "very_long",
        ],
        duplicates="drop",
    )

    bucket_count = (
        df["length_bucket"]
        .nunique()
    )

    per_bucket = target_size // bucket_count

    sample_parts = []

    for bucket in sorted(
        df["length_bucket"]
        .dropna()
        .unique()
    ):
        bucket_df = df[
            df["length_bucket"] == bucket
        ]

        n = min(
            per_bucket,
            len(bucket_df),
        )

        sample_parts.append(
            bucket_df.sample(
                n=n,
                random_state=RANDOM_STATE,
            )
        )

    sample = pd.concat(
        sample_parts,
        ignore_index=True,
    )

    # Fill any remainder.
    remaining = target_size - len(sample)

    if remaining > 0:
        selected_ids = set(
            sample["customer_tweet_id"]
        )

        remaining_df = df[
            ~df["customer_tweet_id"].isin(
                selected_ids
            )
        ]

        if not remaining_df.empty:
            extra = remaining_df.sample(
                n=min(
                    remaining,
                    len(remaining_df),
                ),
                random_state=RANDOM_STATE,
            )

            sample = pd.concat(
                [
                    sample,
                    extra,
                ],
                ignore_index=True,
            )

    sample = sample.sample(
        frac=1,
        random_state=RANDOM_STATE,
    ).reset_index(drop=True)

    sample.insert(
        0,
        "discovery_id",
        range(
            1,
            len(sample) + 1,
        ),
    )

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    sample[
        [
            "discovery_id",
            "customer_tweet_id",
            "customer_author_id",
            "customer_created_at",
            "customer_text",
            "clean_text",
            "text_length",
        ]
    ].to_csv(
        SAMPLE_PATH,
        index=False,
        encoding="utf-8",
    )

    print(
        f"Discovery sample size: "
        f"{len(sample):,}"
    )

    print(
        f"Saved:\n{SAMPLE_PATH}"
    )

    return sample