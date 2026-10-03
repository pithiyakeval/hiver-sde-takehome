import pandas as pd


TEST_PATH = "golden/evaluation_test.csv"
PREDICTIONS_PATH = "evaluation/results/agent_predictions.csv"


def parse_bool(value):
    value = str(value).strip().lower()

    if value in {"true", "yes", "y", "1"}:
        return True

    if value in {"false", "no", "n", "0"}:
        return False

    raise ValueError(f"Unsupported boolean value: {value}")


def main():
    gold = pd.read_csv(TEST_PATH)
    predictions = pd.read_csv(PREDICTIONS_PATH)

    merged = gold[["ID", "Escalate"]].merge(
        predictions[
            [
                "example_id",
                "intent_confidence",
                "retrieval_top1_similarity",
                "should_escalate",
                "escalation_reason",
            ]
        ],
        left_on="ID",
        right_on="example_id",
        how="inner",
    )

    merged["gold_escalate"] = merged["Escalate"].apply(parse_bool)

    print("\nEscalation Diagnosis")
    print("--------------------")

    print("\nGold distribution:")
    print(merged["gold_escalate"].value_counts().sort_index())

    print("\nPredicted distribution:")
    print(merged["should_escalate"].value_counts().sort_index())

    print("\nConfidence statistics:")
    print(merged["intent_confidence"].describe().round(4))

    print("\nRetrieval similarity statistics:")
    print(merged["retrieval_top1_similarity"].describe().round(4))

    print("\nEscalation reasons:")
    print(merged["escalation_reason"].value_counts())

    print("\nConfusion table:")
    print(
        pd.crosstab(
            merged["gold_escalate"],
            merged["should_escalate"],
            rownames=["gold"],
            colnames=["predicted"],
        )
    )


if __name__ == "__main__":
    main()