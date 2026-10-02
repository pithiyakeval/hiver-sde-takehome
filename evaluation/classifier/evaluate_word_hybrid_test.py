from pathlib import Path

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score

from src.intents.rules import classify_by_rule


PROJECT_ROOT = Path(__file__).resolve().parents[2]

TRAIN_PATH = PROJECT_ROOT / "golden" / "evaluation_train.csv"
TEST_PATH = PROJECT_ROOT / "golden" / "evaluation_test.csv"
OUTPUT_PATH = (
    PROJECT_ROOT
    / "evaluation"
    / "results"
    / "word_hybrid_test_predictions.csv"
)


def main():
    train = pd.read_csv(TRAIN_PATH)
    test = pd.read_csv(TEST_PATH)

    train_texts = train["customer_text"].fillna("").astype(str)
    train_labels = train["Intent"].astype(str).str.strip()

    test_texts = test["customer_text"].fillna("").astype(str)
    test_labels = test["Intent"].astype(str).str.strip()

    vectorizer = TfidfVectorizer(
        analyzer="word",
        ngram_range=(1, 2),
        min_df=1,
        max_features=50000,
        sublinear_tf=True,
    )

    train_features = vectorizer.fit_transform(train_texts)
    test_features = vectorizer.transform(test_texts)

    classifier = LogisticRegression(
        C=0.25,
        max_iter=3000,
        class_weight="balanced",
        solver="lbfgs",
    )

    classifier.fit(
        train_features,
        train_labels,
    )

    predictions = []

    for text, features in zip(
        test_texts,
        test_features,
    ):
        rule_prediction = classify_by_rule(text)

        if rule_prediction is not None:
            predictions.append(rule_prediction)
        else:
            predictions.append(
                classifier.predict(features)[0]
            )

    accuracy = accuracy_score(
        test_labels,
        predictions,
    )

    macro_f1 = f1_score(
        test_labels,
        predictions,
        average="macro",
        zero_division=0,
    )

    output_df = test.copy()

    output_df["predicted_intent"] = predictions
    output_df["correct"] = (
        output_df["Intent"].astype(str).str.strip()
        == output_df["predicted_intent"].astype(str).str.strip()
    )

    output_df.to_csv(
        OUTPUT_PATH,
        index=False,
    )

    print("=== Word TF-IDF + Rules ===")
    print(f"Test examples: {len(test)}")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Macro F1: {macro_f1:.4f}")
    print(f"Saved predictions to: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()