from pathlib import Path

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import StratifiedKFold
from scipy.sparse import hstack


PROJECT_ROOT = Path(__file__).resolve().parents[2]
TRAIN_PATH = PROJECT_ROOT / "golden" / "evaluation_train.csv"


def evaluate_configuration(
    name,
    vectorizer_builder,
    classifier_builder,
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

        vectorizer = vectorizer_builder()

        train_features = vectorizer.fit_transform(train_texts)
        valid_features = vectorizer.transform(valid_texts)

        classifier = classifier_builder()

        classifier.fit(
            train_features,
            train_labels,
        )

        predictions = classifier.predict(valid_features)

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

    configurations = [
        (
            "char_tfidf",
            lambda: TfidfVectorizer(
                analyzer="char",
                ngram_range=(3, 5),
                min_df=2,
                max_features=50000,
            ),
            lambda: LogisticRegression(
                max_iter=2000,
                class_weight="balanced",
            ),
        ),
        (
            "word_tfidf",
            lambda: TfidfVectorizer(
                analyzer="word",
                ngram_range=(1, 2),
                min_df=1,
                max_features=50000,
                sublinear_tf=True,
            ),
            lambda: LogisticRegression(
                C=0.25,
                max_iter=3000,
                class_weight="balanced",
                solver="lbfgs",
            ),
        ),
    ]

    results = []

    for name, vectorizer_builder, classifier_builder in configurations:
        result = evaluate_configuration(
            name=name,
            vectorizer_builder=vectorizer_builder,
            classifier_builder=classifier_builder,
            texts=texts,
            labels=labels,
        )

        results.append(result)

        print(
            f"{name}: "
            f"accuracy={result['accuracy']:.4f}, "
            f"macro_f1={result['macro_f1']:.4f}"
        )

    # ---------------------------------------------------------------
    # Word + character TF-IDF
    # ---------------------------------------------------------------

    splitter = StratifiedKFold(
        n_splits=2,
        shuffle=True,
        random_state=42,
    )

    combined_accuracy = []
    combined_f1 = []

    for train_idx, valid_idx in splitter.split(texts, labels):
        train_texts = texts.iloc[train_idx]
        valid_texts = texts.iloc[valid_idx]

        train_labels = labels.iloc[train_idx]
        valid_labels = labels.iloc[valid_idx]

        word_vectorizer = TfidfVectorizer(
            analyzer="word",
            ngram_range=(1, 2),
            min_df=1,
            max_features=50000,
            sublinear_tf=True,
        )

        char_vectorizer = TfidfVectorizer(
            analyzer="char",
            ngram_range=(3, 5),
            min_df=2,
            max_features=50000,
            sublinear_tf=True,
        )

        train_word = word_vectorizer.fit_transform(train_texts)
        valid_word = word_vectorizer.transform(valid_texts)

        train_char = char_vectorizer.fit_transform(train_texts)
        valid_char = char_vectorizer.transform(valid_texts)

        train_features = hstack(
            [train_word, train_char]
        )

        valid_features = hstack(
            [valid_word, valid_char]
        )

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

        predictions = classifier.predict(valid_features)

        combined_accuracy.append(
            accuracy_score(
                valid_labels,
                predictions,
            )
        )

        combined_f1.append(
            f1_score(
                valid_labels,
                predictions,
                average="macro",
                zero_division=0,
            )
        )

    combined_result = {
        "model": "word_plus_char_tfidf",
        "accuracy": sum(combined_accuracy) / len(combined_accuracy),
        "macro_f1": sum(combined_f1) / len(combined_f1),
    }

    results.append(combined_result)

    print(
        f"word_plus_char_tfidf: "
        f"accuracy={combined_result['accuracy']:.4f}, "
        f"macro_f1={combined_result['macro_f1']:.4f}"
    )

    results_df = pd.DataFrame(results)

    output_path = (
        PROJECT_ROOT
        / "evaluation"
        / "results"
        / "generalization_experiments.csv"
    )

    results_df.to_csv(
        output_path,
        index=False,
    )

    print(f"\nSaved results to: {output_path}")


if __name__ == "__main__":
    main()