from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split


INPUT_PATH = Path("golden/golden_evaluation.csv")
TRAIN_PATH = Path("golden/evaluation_train.csv")
TEST_PATH = Path("golden/evaluation_test.csv")

RANDOM_STATE = 42
TEST_SIZE = 0.20


def create_evaluation_split():
    """Create a reproducible stratified train/test split."""

    df = pd.read_csv(INPUT_PATH)

    train_df, test_df = train_test_split(
        df,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=df["Intent"],
    )

    train_df = train_df.sort_values("ID").reset_index(
        drop=True
    )

    test_df = test_df.sort_values("ID").reset_index(
        drop=True
    )

    TRAIN_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    train_df.to_csv(
        TRAIN_PATH,
        index=False,
        encoding="utf-8",
    )

    test_df.to_csv(
        TEST_PATH,
        index=False,
        encoding="utf-8",
    )

    print("=" * 60)
    print("EVALUATION SPLIT CREATED")
    print("=" * 60)

    print(f"\nTotal examples: {len(df):,}")
    print(f"Training examples: {len(train_df):,}")
    print(f"Final test examples: {len(test_df):,}")

    print("\nTraining distribution:")
    print(
        train_df["Intent"]
        .value_counts()
        .sort_index()
        .to_string()
    )

    print("\nFinal test distribution:")
    print(
        test_df["Intent"]
        .value_counts()
        .sort_index()
        .to_string()
    )

    print(f"\nSaved training set: {TRAIN_PATH}")
    print(f"Saved final test set: {TEST_PATH}")


if __name__ == "__main__":
    create_evaluation_split()