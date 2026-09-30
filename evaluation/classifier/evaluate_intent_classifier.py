import pandas as pd

from sklearn.metrics import accuracy_score, f1_score

from evaluation.intents.classifier import IntentClassifier


# TRAIN_PATH = "golden/evaluation_train.csv"
TEST_PATH = "golden/evaluation_test.csv"


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

    print("=== Production Intent Classifier ===")
    print(f"Test examples: {len(test)}")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Macro F1: {macro_f1:.4f}")


if __name__ == "__main__":
    main()