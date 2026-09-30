import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


DATA_PATH = "data/processed/retrieval_pairs.csv"


class TfidfRetriever:
    def __init__(self, data_path=DATA_PATH):
        self.df = pd.read_csv(data_path)

        self.vectorizer = TfidfVectorizer(
            lowercase=True,
            ngram_range=(1, 2),
            min_df=2,
            max_features=50000,
        )

        self.matrix = self.vectorizer.fit_transform(
            self.df["customer_text"]
        )

    def search(self, query, top_k=5):
        query_vector = self.vectorizer.transform([query])

        scores = cosine_similarity(
            query_vector,
            self.matrix,
        ).flatten()

        top_indices = scores.argsort()[-top_k:][::-1]

        results = self.df.iloc[top_indices].copy()
        results["similarity"] = scores[top_indices]

        return results[
            [
                "conversation_id",
                "customer_text",
                "amazonhelp_text",
                "similarity",
            ]
        ].reset_index(drop=True)