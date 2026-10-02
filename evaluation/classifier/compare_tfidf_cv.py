from pathlib import Path

import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import make_scorer, accuracy_score, f1_score
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.pipeline import FeatureUnion, Pipeline


PROJECT_ROOT = Path(__file__).resolve().parents[2]
TRAIN_PATH = PROJECT_ROOT / "golden" / "evaluation_train.csv"


CV = StratifiedKFold(
    n_splits=2,
    shuffle=True,
    random_state=42,
)


def evaluate_model(name, vectorizer, c_value):
    """Evaluate one classifier configuration using 5-fold CV."""

    df = pd.read_csv(TRAIN_PATH)

    X = df["customer_text"].fillna("")
    y = df["Intent"]

    pipeline = Pipeline(
        [
            ("tfidf", vectorizer),
            (
                "classifier",
                LogisticRegression(
                    C=c_value,
                    max_iter=3000,
                    class_weight="balanced",
                    solver="lbfgs",
                    random_state=42,
                ),
            ),
        ]
    )

    results = cross_validate(
        pipeline,
        X,
        y,
        cv=CV,
        scoring={
            "accuracy": make_scorer(accuracy_score),
            "macro_f1": make_scorer(f1_score, average="macro"),
        },
        n_jobs=-1,
    )

    return {
        "model": name,
        "C": c_value,
        "accuracy_mean": results["test_accuracy"].mean(),
        "accuracy_std": results["test_accuracy"].std(),
        "macro_f1_mean": results["test_macro_f1"].mean(),
        "macro_f1_std": results["test_macro_f1"].std(),
    }


def build_models():
    """Return classifier configurations to compare."""

    models = []

    # ------------------------------------------------------------------
    # Word TF-IDF
    # ------------------------------------------------------------------

    for c_value in [0.25, 0.5, 1.0, 2.0, 4.0]:
        models.append(
            (
                f"word_tfidf_C{c_value}",
                TfidfVectorizer(
                    lowercase=True,
                    ngram_range=(1, 2),
                    min_df=1,
                    max_features=50000,
                    sublinear_tf=True,
                ),
                c_value,
            )
        )

    # ------------------------------------------------------------------
    # Character TF-IDF
    # ------------------------------------------------------------------

    for ngram_range in [(3, 5), (3, 6), (2, 5)]:
        for c_value in [0.25, 0.5, 1.0, 2.0, 4.0]:
            models.append(
                (
                    f"char_{ngram_range}_C{c_value}",
                    TfidfVectorizer(
                        analyzer="char",
                        ngram_range=ngram_range,
                        min_df=2,
                        max_features=50000,
                        sublinear_tf=True,
                    ),
                    c_value,
                )
            )

    # ------------------------------------------------------------------
    # Word + Character TF-IDF
    # ------------------------------------------------------------------

    for c_value in [0.25, 0.5, 1.0, 2.0, 4.0]:
        combined_vectorizer = FeatureUnion(
            [
                (
                    "word",
                    TfidfVectorizer(
                        lowercase=True,
                        ngram_range=(1, 2),
                        min_df=1,
                        max_features=30000,
                        sublinear_tf=True,
                    ),
                ),
                (
                    "char",
                    TfidfVectorizer(
                        analyzer="char",
                        ngram_range=(3, 5),
                        min_df=2,
                        max_features=30000,
                        sublinear_tf=True,
                    ),
                ),
            ]
        )

        models.append(
            (
                f"word_char_C{c_value}",
                combined_vectorizer,
                c_value,
            )
        )

    return models


def main():
    if not TRAIN_PATH.exists():
        raise FileNotFoundError(
            f"Training dataset not found: {TRAIN_PATH}"
        )

    print("=" * 78)
    print("Classifier Development Experiment")
    print("=" * 78)
    print(f"Dataset : {TRAIN_PATH}")
    print("Evaluation: 5-fold stratified cross-validation")
    print("Frozen test set: NOT USED")
    print()

    results = []

    for index, (name, vectorizer, c_value) in enumerate(
        build_models(),
        start=1,
    ):
        print(
            f"[{index}] Evaluating {name}...",
            flush=True,
        )

        result = evaluate_model(
            name=name,
            vectorizer=vectorizer,
            c_value=c_value,
        )

        results.append(result)

        print(
            f"    Accuracy : "
            f"{result['accuracy_mean']:.4f} "
            f"+/- {result['accuracy_std']:.4f}"
        )
        print(
            f"    Macro F1 : "
            f"{result['macro_f1_mean']:.4f} "
            f"+/- {result['macro_f1_std']:.4f}"
        )
        print()

    results_df = pd.DataFrame(results)

    results_df = results_df.sort_values(
        by=["macro_f1_mean", "accuracy_mean"],
        ascending=False,
    ).reset_index(drop=True)

    print("=" * 78)
    print("FINAL CV RANKING")
    print("=" * 78)

    print(
        results_df[
            [
                "model",
                "C",
                "accuracy_mean",
                "macro_f1_mean",
                "accuracy_std",
                "macro_f1_std",
            ]
        ].to_string(index=False)
    )

    print()
    print("Best development configuration:")
    print(
        results_df.iloc[0][
            [
                "model",
                "C",
                "accuracy_mean",
                "macro_f1_mean",
            ]
        ].to_string()
    )

    output_path = (
        PROJECT_ROOT
        / "evaluation"
        / "results"
        / "classifier_cv_experiments.csv"
    )

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    results_df.to_csv(
        output_path,
        index=False,
    )

    print()
    print(f"Saved: {output_path}")


if __name__ == "__main__":
    main()