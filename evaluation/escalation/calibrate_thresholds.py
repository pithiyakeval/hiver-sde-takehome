import pandas as pd
from sklearn.metrics import accuracy_score


TRAIN_PATH = "golden/evaluation_train.csv"

INTENT_CONFIDENCE_VALUES = [
    0.08,
    0.09,
    0.10,
    0.11,
    0.12,
    0.13,
    0.14,
    0.15,
]

RETRIEVAL_VALUES = [
    0.20,
    0.25,
    0.30,
    0.35,
]


def parse_bool(value):
    value = str(value).strip().lower()

    if value in {"true", "yes", "y", "1"}:
        return True

    if value in {"false", "no", "n", "0"}:
        return False

    raise ValueError(f"Unsupported boolean value: {value}")


def main():
    df = pd.read_csv(TRAIN_PATH)

    gold = df["Escalate"].apply(parse_bool)

    # Development-time approximation:
    # the existing classifier is evaluated on these examples,
    # while the frozen test set remains untouched.
    from src.intents.classifier import IntentClassifier
    from src.retrieval.tfidf_retriever import TfidfRetriever

    classifier = IntentClassifier()
    retriever = TfidfRetriever()

    rows = []

    for _, row in df.iterrows():
        message = row["customer_text"]

        classification = classifier.predict(message)

        retrieved = retriever.search(
            message,
            top_k=3,
        )

        top_similarity = (
            float(retrieved.iloc[0]["similarity"])
            if not retrieved.empty
            else 0.0
        )

        rows.append(
            {
                "gold_escalate": parse_bool(row["Escalate"]),
                "intent": classification["intent"],
                "confidence": classification["confidence"],
                "top_similarity": top_similarity,
            }
        )

    scored = pd.DataFrame(rows)

    best = None

    for confidence_threshold in INTENT_CONFIDENCE_VALUES:
        for retrieval_threshold in RETRIEVAL_VALUES:
            predicted = (
                (scored["intent"] == "other_or_unclear")
                | (
                    scored["confidence"]
                    < confidence_threshold
                )
                | (
                    scored["top_similarity"]
                    < retrieval_threshold
                )
            )

            accuracy = accuracy_score(
                scored["gold_escalate"],
                predicted,
            )

            result = {
                "intent_threshold": confidence_threshold,
                "retrieval_threshold": retrieval_threshold,
                "accuracy": accuracy,
            }

            if best is None or accuracy > best["accuracy"]:
                best = result

    print("\nDevelopment Escalation Calibration")
    print("----------------------------------")
    print(f"Examples: {len(scored)}")
    print(f"Best intent threshold: {best['intent_threshold']:.2f}")
    print(
        f"Best retrieval threshold: "
        f"{best['retrieval_threshold']:.2f}"
    )
    print(f"Development accuracy: {best['accuracy']:.4f}")


if __name__ == "__main__":
    main()