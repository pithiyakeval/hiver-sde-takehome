import pytest

from src.retrieval.tfidf_retriever import TfidfRetriever


@pytest.fixture(scope="module")
def retriever():
    """Create one retriever instance for the test module."""
    return TfidfRetriever()


def test_retriever_returns_requested_number_of_results(retriever):
    """Retriever should return at most the requested number of results."""

    query = "My package says delivered but I have not received it."

    results = retriever.search(query, top_k=3)

    assert len(results) == 3


def test_retriever_returns_expected_columns(retriever):
    """Retrieved results should follow the expected output schema."""

    results = retriever.search(
        "My package says delivered but I have not received it.",
        top_k=3,
    )

    expected_columns = {
        "conversation_id",
        "customer_text",
        "amazonhelp_text",
        "similarity",
    }

    assert set(results.columns) == expected_columns


def test_retriever_returns_valid_similarity_scores(retriever):
    """Similarity scores should be valid cosine-similarity values."""

    results = retriever.search(
        "My package says delivered but I have not received it.",
        top_k=3,
    )

    assert results["similarity"].notna().all()
    assert results["similarity"].between(0.0, 1.0).all()


def test_retriever_results_are_ranked_by_similarity(retriever):
    """Results should be returned from highest to lowest similarity."""

    results = retriever.search(
        "My package says delivered but I have not received it.",
        top_k=5,
    )

    similarities = results["similarity"].tolist()

    assert similarities == sorted(similarities, reverse=True)


def test_retriever_handles_different_top_k_values(retriever):
    """Retriever should respect different top-k requests."""

    query = "I want a refund for my order."

    for top_k in [1, 3, 5]:
        results = retriever.search(query, top_k=top_k)

        assert len(results) == top_k