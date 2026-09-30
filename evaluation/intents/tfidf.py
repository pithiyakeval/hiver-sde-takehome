from sklearn.feature_extraction.text import TfidfVectorizer

from evaluation.intents.discovery_config import (
    CUSTOM_STOPWORDS,
    MIN_DOCUMENT_FREQUENCY,
    MAX_DOCUMENT_FREQUENCY,
    MAX_FEATURES,
)


def build_tfidf(sample):
    """
    Convert the discovery sample into a TF-IDF matrix.

    Unigrams + bigrams make the discovered topics easier
    to interpret.
    """
    print("\n" + "=" * 70)
    print("BUILDING TF-IDF REPRESENTATION")
    print("=" * 70)

    vectorizer = TfidfVectorizer(
        stop_words=list(CUSTOM_STOPWORDS),
        ngram_range=(1, 2),
        min_df=MIN_DOCUMENT_FREQUENCY,
        max_df=MAX_DOCUMENT_FREQUENCY,
        max_features=MAX_FEATURES,
        sublinear_tf=True,
        strip_accents="unicode",
    )

    matrix = vectorizer.fit_transform(sample["clean_text"])

    print(f"Documents: {matrix.shape[0]:,}")
    print(f"Vocabulary size: {matrix.shape[1]:,}")

    return vectorizer, matrix