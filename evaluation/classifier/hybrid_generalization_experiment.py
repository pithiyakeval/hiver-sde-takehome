from pathlib import Path

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import StratifiedKFold

from src.intents.rules import classify_by_rule


PROJECT_ROOT = Path(__file__).resolve().parents[2]
TRAIN_PATH = PROJECT_ROOT / "golden" / "evaluation_train.csv"


def evaluate_model(
    name,
    vectorizer_config,
    classifier_config,
    texts,
    labels,
):
    splitter = StratifiedKFold(
        n_splits=2,
        shuffle=True,
        random_state=42,
    )

    accuracies = []
    macro_f1_scores = []

    for train_idx, valid_idx in splitter.split(texts, labels):
        train_texts = texts.iloc[train_idx]
        valid_texts = texts.iloc[valid_idx]

        train_labels = labels.iloc[train_idx]
        valid_labels = labels.iloc[valid_idx]

        vectorizer = TfidfVectorizer(**vectorizer_config)

        train_features = vectorizer.fit_transform(train_texts)
        valid_features = vectorizer.transform(valid_texts)

        classifier = LogisticRegression(**classifier_config)

        classifier.fit(
            train_features,
            train_labels,
        )

        predictions = []

        for text, features in zip(
            valid_texts,
            valid_features,
        ):
            rule_prediction = classify_by_rule(text)

            if rule_prediction is not None:
                predictions.append(rule_prediction)
            else:
                predictions.append(
                    classifier.predict(features)[0]
                )

        accuracies.append(
            accuracy_score(
                valid_labels,
                predictions,
            )
        )

        macro_f1_scores.append(
            f1_score(
                valid_labels,
                predictions,
                average="macro",
                zero_division=0,
            )
        )

    return {
        "model": name,
        "accuracy": sum(accuracies) / len(accuracies),
        "macro_f1": sum(macro_f1_scores) / len(macro_f1_scores),
    }


def main():
    df = pd.read_csv(TRAIN_PATH)

    texts = df["customer_text"].fillna("").astype(str)
    labels = df["Intent"].astype(str).str.strip()

    results = []

    configurations = [
        (
            "char_tfidf_plus_rules",
            {
                "analyzer": "char",
                "ngram_range": (3, 5),
                "min_df": 2,
                "max_features": 50000,
            },
            {
                "max_iter": 2000,
                "class_weight": "balanced",
            },
        ),
        (
            "word_tfidf_plus_rules",
            {
                "analyzer": "word",
                "ngram_range": (1, 2),
                "min_df": 1,
                "max_features": 50000,
                "sublinear_tf": True,
            },
            {
                "C": 0.25,
                "max_iter": 3000,
                "class_weight": "balanced",
                "solver": "lbfgs",
            },
        ),
    ]

    for name, vectorizer_config, classifier_config in configurations:
        result = evaluate_model(
            name=name,
            vectorizer_config=vectorizer_config,
            classifier_config=classifier_config,
            texts=texts,
            labels=labels,
        )

        results.append(result)

        print(
            f"{name}: "
            f"accuracy={result['accuracy']:.4f}, "
            f"macro_f1={result['macro_f1']:.4f}"
        )

    results_df = pd.DataFrame(results)

    output_path = (
        PROJECT_ROOT
        / "evaluation"
        / "results"
        / "hybrid_generalization_experiments.csv"
    )

    results_df.to_csv(
        output_path,
        index=False,
    )

    print(f"\nSaved results to: {output_path}")


if __name__ == "__main__":
    main()