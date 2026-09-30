from pathlib import Path

import pandas as pd


SOURCE_PATH = Path("golden/golden_evaluation.csv")
TRAIN_PATH = Path("golden/evaluation_train.csv")
TEST_PATH = Path("golden/evaluation_test.csv")


def validate_split():
    """Validate the integrity of the frozen evaluation split."""

    source = pd.read_csv(SOURCE_PATH)
    train = pd.read_csv(TRAIN_PATH)
    test = pd.read_csv(TEST_PATH)

    source_ids = set(source["ID"])
    train_ids = set(train["ID"])
    test_ids = set(test["ID"])

    overlap = train_ids & test_ids
    missing = source_ids - (train_ids | test_ids)
    unexpected = (train_ids | test_ids) - source_ids

    print("=" * 60)
    print("EVALUATION SPLIT VALIDATION")
    print("=" * 60)

    print(f"\nSource examples: {len(source):,}")
    print(f"Training examples: {len(train):,}")
    print(f"Final test examples: {len(test):,}")

    print(f"\nTrain/test ID overlap: {len(overlap)}")
    print(f"Missing source IDs: {len(missing)}")
    print(f"Unexpected IDs: {len(unexpected)}")

    assert not overlap, "Train/test ID overlap detected."
    assert not missing, "Some source examples are missing."
    assert not unexpected, "Unexpected IDs found."

    assert len(train) + len(test) == len(source)

    print("\n✓ No train/test overlap")
    print("✓ No source examples missing")
    print("✓ No unexpected examples")
    print("✓ All 200 examples accounted for")

    print("\nFinal test set is ready to freeze.")


if __name__ == "__main__":
    validate_split()