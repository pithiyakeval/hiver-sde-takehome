from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
PREDICTIONS_PATH = PROJECT_ROOT / "evaluation" / "results" / "agent_predictions.csv"


INTENT_CODE_MAP = {
    "OS": "order_status",
    "DL": "delivery_late",
    "DNR": "delivered_not_received",
    "DP": "delivery_promise",
    "MWI": "missing_or_wrong_item",
    "DIP": "damaged_item_or_package",
    "RR": "return_or_refund",
    "PC": "payment_or_charge",
    "PM": "prime_membership",
    "AA": "account_or_access",
    "PDH": "product_or_device_help",
    "CSF": "customer_service_followup",
    "OU": "other_or_unclear",
}


def main() -> None:
    if not PREDICTIONS_PATH.exists():
        raise FileNotFoundError(
            f"Predictions file not found: {PREDICTIONS_PATH}\n"
            "Run evaluation/evaluate_agent.py first."
        )

    df = pd.read_csv(PREDICTIONS_PATH)

    required_columns = [
        "example_id",
        "customer_text",
        "gold_intent",
        "predicted_intent",
        "intent_confidence",
        "classification_reason",
        "draft_response",
        "should_escalate",
        "escalation_reason",
    ]

    missing_columns = [
        column for column in required_columns
        if column not in df.columns
    ]

    if missing_columns:
        raise ValueError(
            f"Missing required columns: {missing_columns}"
        )

    # Normalize gold intent codes into the same representation
    # used by the LLM classifier.
    df["gold_intent_normalized"] = (
        df["gold_intent"]
        .astype(str)
        .str.strip()
        .map(INTENT_CODE_MAP)
    )

    df["predicted_intent_normalized"] = (
        df["predicted_intent"]
        .astype(str)
        .str.strip()
    )

    # Fail loudly if an unknown gold code appears.
    unknown_gold = df[df["gold_intent_normalized"].isna()]

    if not unknown_gold.empty:
        raise ValueError(
            "Unknown gold intent codes found:\n"
            f"{unknown_gold['gold_intent'].unique().tolist()}"
        )

    # Compare normalized values.
    df["is_correct"] = (
        df["gold_intent_normalized"]
        == df["predicted_intent_normalized"]
    )

    errors = df[~df["is_correct"]].copy()

    errors = errors.sort_values("example_id")

    print("=" * 100)
    print("AGENT INTENT ERROR REVIEW")
    print("=" * 100)

    print(f"\nPredictions file  : {PREDICTIONS_PATH}")
    print(f"Total examples    : {len(df)}")
    print(f"Correct intents   : {df['is_correct'].sum()}")
    print(f"Incorrect intents : {len(errors)}")
    print(
        f"Intent accuracy   : "
        f"{df['is_correct'].mean():.4f}"
    )

    print("\n" + "=" * 100)
    print("INCORRECT PREDICTIONS")
    print("=" * 100)

    if errors.empty:
        print("\nNo intent errors found.")
    else:
        for _, row in errors.iterrows():
            print("\n" + "-" * 100)

            print(f"Example ID       : {row['example_id']}")
            print(f"Gold intent      : {row['gold_intent']}")
            print(f"Predicted intent : {row['predicted_intent']}")
            print(f"Confidence       : {row['intent_confidence']}")

            print("\nCustomer message:")
            print(row["customer_text"])

            print("\nClassification reason:")
            print(row["classification_reason"])

            print("\nDraft response:")
            print(row["draft_response"])

            print(f"\nShould escalate  : {row['should_escalate']}")

            print("\nEscalation reason:")
            print(row["escalation_reason"])

    print("\n" + "=" * 100)
    print("ERROR SUMMARY")
    print("=" * 100)

    error_summary = (
        errors.groupby(
            [
                "gold_intent",
                "predicted_intent",
            ]
        )
        .size()
        .reset_index(name="count")
        .sort_values(
            ["count", "gold_intent", "predicted_intent"],
            ascending=[False, True, True],
        )
    )

    if error_summary.empty:
        print("No intent errors found.")
    else:
        print(error_summary.to_string(index=False))


if __name__ == "__main__":
    main()