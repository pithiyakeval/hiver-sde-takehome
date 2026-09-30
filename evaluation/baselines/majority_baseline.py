import pandas as pd

from sklearn.metrics import accuracy_score, f1_score


TRAIN_PATH = "golden/evaluation_train.csv"
TEST_PATH = "golden/evaluation_test.csv"


def main():
    train = pd.read_csv(TRAIN_PATH)
    test = pd.read_csv(TEST_PATH)

    y_train = train["Intent"].astype(str).str.strip()
    y_test = test["Intent"].astype(str).str.strip()

    # Majority intent is determined ONLY from training data.
    majority_intent = y_train.value_counts().idxmax()

    # Predict the same intent for every test example.
    y_pred = [majority_intent] * len(y_test)

    accuracy = accuracy_score(
        y_test,
        y_pred,
    )

    macro_f1 = f1_score(
        y_test,
        y_pred,
        average="macro",
        zero_division=0,
    )

    print("=== Majority Baseline ===")
    print(f"Train examples: {len(y_train)}")
    print(f"Test examples: {len(y_test)}")
    print(f"Majority intent: {majority_intent}")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Macro F1: {macro_f1:.4f}")


if __name__ == "__main__":
    main()