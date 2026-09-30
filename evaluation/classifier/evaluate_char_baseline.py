import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score


TRAIN_PATH = "golden/evaluation_train.csv"
TEST_PATH = "golden/evaluation_test.csv"
OUTPUT_PATH = "evaluation/results/char_tfidf_predictions.csv"


def main():
    train = pd.read_csv(TRAIN_PATH)
    test = pd.read_csv(TEST_PATH)

    X_train = train["customer_text"].fillna("")
    y_train = train["Intent"]

    X_test = test["customer_text"].fillna("")
    y_test = test["Intent"]

    vectorizer = TfidfVectorizer(
        analyzer="char",
        ngram_range=(3, 5),
        min_df=2,
        max_features=50000,
    )

    X_train_tfidf = vectorizer.fit_transform(X_train)
    X_test_tfidf = vectorizer.transform(X_test)

    model = LogisticRegression(
        max_iter=2000,
        class_weight="balanced",
    )

    model.fit(X_train_tfidf, y_train)

    predictions = model.predict(X_test_tfidf)

    accuracy = accuracy_score(y_test, predictions)

    macro_f1 = f1_score(
        y_test,
        predictions,
        average="macro",
        zero_division=0,
    )

    output = test.copy()

    output["predicted_intent"] = predictions
    output["correct"] = (
        output["Intent"] == output["predicted_intent"]
    )

    output.to_csv(
        OUTPUT_PATH,
        index=False,
    )

    print("Frozen Test Evaluation")
    print("----------------------")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Macro F1: {macro_f1:.4f}")
    print(f"Incorrect: {(~output['correct']).sum()}")
    print(f"Saved: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()