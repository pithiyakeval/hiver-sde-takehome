import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.pipeline import FeatureUnion, Pipeline


TRAIN_PATH = "golden/evaluation_train.csv"


def evaluate(vectorizer):
    df = pd.read_csv(TRAIN_PATH)

    X = df["customer_text"].fillna("")
    y = df["Intent"]

    pipeline = Pipeline([
        ("tfidf", vectorizer),
        (
            "classifier",
            LogisticRegression(
                max_iter=2000,
                class_weight="balanced",
            ),
        ),
    ])

    cv = StratifiedKFold(
        n_splits=5,
        shuffle=True,
        random_state=42,
    )

    results = cross_validate(
        pipeline,
        X,
        y,
        cv=cv,
        scoring=["accuracy", "f1_macro"],
    )

    return (
        results["test_accuracy"].mean(),
        results["test_f1_macro"].mean(),
    )


def main():
    # 1. Word TF-IDF
    word_accuracy, word_f1 = evaluate(
        TfidfVectorizer(
            lowercase=True,
            ngram_range=(1, 2),
            min_df=1,
            max_features=50000,
        )
    )

    # 2. Character TF-IDF
    char_accuracy, char_f1 = evaluate(
        TfidfVectorizer(
            analyzer="char",
            ngram_range=(3, 5),
            min_df=2,
            max_features=50000,
        )
    )

    # 3. Word + Character TF-IDF
    combined_vectorizer = FeatureUnion([
        (
            "word",
            TfidfVectorizer(
                lowercase=True,
                ngram_range=(1, 2),
                min_df=1,
                max_features=30000,
            ),
        ),
        (
            "char",
            TfidfVectorizer(
                analyzer="char",
                ngram_range=(3, 5),
                min_df=2,
                max_features=30000,
            ),
        ),
    ])

    combined_accuracy, combined_f1 = evaluate(
        combined_vectorizer
    )

    print("5-Fold CV Comparison")
    print("--------------------")

    print(
        f"Word TF-IDF  | "
        f"Accuracy: {word_accuracy:.4f} | "
        f"Macro F1: {word_f1:.4f}"
    )

    print(
        f"Char TF-IDF  | "
        f"Accuracy: {char_accuracy:.4f} | "
        f"Macro F1: {char_f1:.4f}"
    )

    print(
        f"Word + Char  | "
        f"Accuracy: {combined_accuracy:.4f} | "
        f"Macro F1: {combined_f1:.4f}"
    )


if __name__ == "__main__":
    main()