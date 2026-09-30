import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.pipeline import Pipeline


TRAIN_PATH = "golden/evaluation_train.csv"


def main():
    df = pd.read_csv(TRAIN_PATH)

    X = df["customer_text"].fillna("")
    y = df["Intent"]

    pipeline = Pipeline([
        (
            "tfidf",
            TfidfVectorizer(
                lowercase=True,
                ngram_range=(1, 2),
                min_df=1,
                max_features=50000,
            ),
        ),
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

    print("5-Fold Cross-Validation")
    print("-----------------------")
    print(
        f"Accuracy: "
        f"{results['test_accuracy'].mean():.4f} "
        f"+/- {results['test_accuracy'].std():.4f}"
    )
    print(
        f"Macro F1: "
        f"{results['test_f1_macro'].mean():.4f} "
        f"+/- {results['test_f1_macro'].std():.4f}"
    )


if __name__ == "__main__":
    main()