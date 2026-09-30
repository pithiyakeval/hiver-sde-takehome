import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression


GOLDEN_PATH = "golden/evaluation_train.csv"


class IntentClassifier:
    def __init__(self, golden_path=GOLDEN_PATH):
        df = pd.read_csv(golden_path)

        df = df.dropna(
            subset=["customer_text", "Intent"]
        ).copy()

        df["customer_text"] = (
            df["customer_text"]
            .astype(str)
            .str.strip()
        )

        df["Intent"] = (
            df["Intent"]
            .astype(str)
            .str.strip()
        )

        self.vectorizer = TfidfVectorizer(
            analyzer="char",
            ngram_range=(3, 5),
            min_df=2,
            max_features=50000,
            # sublinear_tf=True,
        )

        X = self.vectorizer.fit_transform(
            df["customer_text"]
        )

        self.model = LogisticRegression(
            max_iter=2000,
            class_weight="balanced",
        )

        self.model.fit(X, df["Intent"])

    def predict(self, text):
        vector = self.vectorizer.transform([text])

        intent = self.model.predict(vector)[0]

        probabilities = self.model.predict_proba(vector)[0]

        confidence = probabilities.max()

        return {
            "intent": intent,
            "confidence": float(confidence),
        }