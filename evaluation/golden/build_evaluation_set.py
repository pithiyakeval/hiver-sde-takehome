from pathlib import Path

import pandas as pd


GOLDEN_PATH = Path("golden/golden_set.csv")
CANDIDATES_PATH = Path("golden/golden_candidates.csv")
OUTPUT_PATH = Path("golden/golden_evaluation.csv")


def build_evaluation_set():
    """Join human annotations with the original customer messages."""

    golden = pd.read_csv(GOLDEN_PATH)
    candidates = pd.read_csv(CANDIDATES_PATH)

    candidates = candidates[
        [
            "example_id",
            "customer_tweet_id",
            "customer_text",
        ]
    ].copy()

    evaluation = golden.merge(
        candidates,
        left_on="ID",
        right_on="example_id",
        how="left",
        validate="one_to_one",
    )

    evaluation = evaluation.drop(
        columns=["example_id"]
    )

    columns = [
        "ID",
        "customer_tweet_id",
        "customer_text",
        "Intent",
        "Confidence",
        "Ambiguous",
        "Action",
        "Escalate",
        "Ambiguity_Reason",
        "Escalation_Reason",
        "Taxonomy_Gap",
        "Taxonomy_Gap_Reason",
        "Evidence",
        "Notes",
    ]

    evaluation = evaluation[columns]

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    evaluation.to_csv(
        OUTPUT_PATH,
        index=False,
        encoding="utf-8",
    )

    print(f"Saved: {OUTPUT_PATH}")
    print(f"Rows: {len(evaluation):,}")
    print(
        "Missing customer text:",
        evaluation["customer_text"].isna().sum(),
    )
    print(
        "Missing intent:",
        evaluation["Intent"].isna().sum(),
    )


if __name__ == "__main__":
    build_evaluation_set()