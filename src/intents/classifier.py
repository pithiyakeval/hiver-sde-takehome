from pathlib import Path

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from src.intents.models import IntentPrediction
from src.intents.rules import classify_by_rule


PROJECT_ROOT = Path(__file__).resolve().parents[2]
TRAIN_PATH = PROJECT_ROOT / "golden" / "evaluation_train.csv"


RULE_CONFIDENCE = 0.99


class IntentClassifier:
    """
    Hybrid intent classifier.

    High-precision rules handle distinctive intent patterns first.
    Messages without a strong rule match are classified using the
    character n-gram TF-IDF + Logistic Regression model.

    The ML configuration was selected during development and evaluated
    separately on the frozen test set.
    """

    def __init__(self):
        self.vectorizer = TfidfVectorizer(
            analyzer="char",
            ngram_range=(3, 5),
            min_df=2,
            max_features=50000,
        )

        self.classifier = LogisticRegression(
            max_iter=2000,
            class_weight="balanced",
        )

        self._fit()

    def _fit(self) -> None:
        """Fit the ML fallback classifier on the development set."""

        if not TRAIN_PATH.exists():
            raise FileNotFoundError(
                f"Training dataset not found: {TRAIN_PATH}"
            )

        df = pd.read_csv(TRAIN_PATH)

        required_columns = {"customer_text", "Intent"}
        missing_columns = required_columns - set(df.columns)

        if missing_columns:
            raise ValueError(
                "Training dataset is missing required columns: "
                f"{sorted(missing_columns)}"
            )

        texts = df["customer_text"].fillna("").astype(str)
        labels = df["Intent"].astype(str).str.strip()

        if texts.empty:
            raise ValueError("Training dataset contains no examples.")

        features = self.vectorizer.fit_transform(texts)
        self.classifier.fit(features, labels)

    def predict(self, text: str) -> dict:
        """
        Predict the customer's intent.

        High-precision rules are applied first. The ML classifier is used
        when no strong rule applies.
        """

        if not isinstance(text, str):
            raise TypeError("text must be a string")

        text = text.strip()

        if not text:
            raise ValueError("text must not be empty")

        # --------------------------------------------------------------
        # High-precision rule layer
        # --------------------------------------------------------------

        rule_intent = classify_by_rule(text)

        if rule_intent is not None:
            return IntentPrediction(
                intent=rule_intent,
                confidence=RULE_CONFIDENCE,
                reason="A high-precision deterministic rule matched the customer message.",
            )

        # --------------------------------------------------------------
        # ML fallback
        # --------------------------------------------------------------

        features = self.vectorizer.transform([text])
        probabilities = self.classifier.predict_proba(features)[0]

        predicted_index = probabilities.argmax()

        return IntentPrediction(
            intent=self.classifier.classes_[predicted_index],
            confidence=float(probabilities[predicted_index]),
            reason="Intent predicted by the TF-IDF Logistic Regression classifier.",
        )