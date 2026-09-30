import pandas as pd

from evaluation.golden.config import SAMPLE_SIZE


def validate_final_set(final_df: pd.DataFrame):
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

    assert len(final_df) == SAMPLE_SIZE, (
        f"Expected {SAMPLE_SIZE} examples, "
        f"got {len(final_df)}."
    )

    print("\nValidation: PASSED")