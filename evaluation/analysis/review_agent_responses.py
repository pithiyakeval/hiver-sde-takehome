from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
PREDICTIONS_PATH = PROJECT_ROOT / "evaluation" / "results" / "agent_predictions.csv"


REQUIRED_COLUMNS = [
    "example_id",
    "customer_text",
    "gold_intent",
    "predicted_intent",
    "intent_confidence",
    "draft_response",
    "retrieval_top1_similarity",
    "retrieval_count",
    "grounded",
    "should_escalate",
    "escalation_reason",
]


def main() -> None:
    if not PREDICTIONS_PATH.exists():
        raise FileNotFoundError(
            f"Missing predictions file: {PREDICTIONS_PATH}"
        )

    df = pd.read_csv(PREDICTIONS_PATH)

    missing = [column for column in REQUIRED_COLUMNS if column not in df.columns]
    if missing:
        raise ValueError(f"Missing columns: {missing}")

    df = df.sort_values("example_id")

    print("=" * 100)
    print("AGENT RESPONSE REVIEW")
    print("=" * 100)

    print(f"\nExamples : {len(df)}")
    print(f"Grounded : {df['grounded'].sum()}/{len(df)}")
    print(
        f"Escalated: "
        f"{df['should_escalate'].sum()}/{len(df)}"
    )

    print("\n" + "=" * 100)
    print("GENERATED RESPONSES")
    print("=" * 100)

    for _, row in df.iterrows():
        print("\n" + "-" * 100)

        print(
            f"{row['example_id']} | "
            f"Gold={row['gold_intent']} | "
            f"Pred={row['predicted_intent']} | "
            f"Conf={row['intent_confidence']:.2f} | "
            f"Retrieval={row['retrieval_top1_similarity']:.3f}"
        )

        print("\nCUSTOMER")
        print(row["customer_text"])

        print("\nDRAFT")
        print(row["draft_response"])

        print(
            f"\nGROUNDED={row['grounded']} | "
            f"ESCALATE={row['should_escalate']}"
        )

        print("ESCALATION")
        print(row["escalation_reason"])

    print("\n" + "=" * 100)
    print("REVIEW COMPLETE")
    print("=" * 100)


if __name__ == "__main__":
    main()