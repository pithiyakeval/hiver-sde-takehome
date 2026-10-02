import pandas as pd

from sklearn.metrics import accuracy_score, f1_score

from src.intents.classifier import IntentClassifier


TEST_PATH = "golden/evaluation_test.csv"
OUTPUT_PATH = "evaluation/results/hybrid_predictions.csv"


def main():
    test = pd.read_csv(TEST_PATH)

    classifier = IntentClassifier()

    predictions = [
        classifier.predict(text)["intent"]
        for text in test["customer_text"].fillna("")
    ]

    accuracy = accuracy_score(
        test["Intent"],
        predictions,
    )

    macro_f1 = f1_score(
        test["Intent"],
        predictions,
        average="macro",
        zero_division=0,
    )

    # Save individual predictions for error analysis
    prediction_df = test.copy()

    prediction_df["predicted_intent"] = predictions

    prediction_df["correct"] = (
        prediction_df["Intent"].astype(str).str.strip()
        == prediction_df["predicted_intent"].astype(str).str.strip()
    )

    prediction_df.to_csv(
        OUTPUT_PATH,
        index=False,
    )

    print("=== Production Intent Classifier ===")
    print(f"Test examples: {len(test)}")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Macro F1: {macro_f1:.4f}")
    print(f"Saved predictions to: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()