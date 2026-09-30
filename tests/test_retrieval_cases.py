import pytest

from src.retrieval.tfidf_retriever import TfidfRetriever


RETRIEVAL_CASES = [
    "My package says delivered but I have not received it.",
    "I want a refund for my returned item.",
    "My Amazon account is locked and I cannot log in.",
    "My payment was charged twice.",
    "My Echo device is not working.",
    "Hello, can someone help me?",
]


@pytest.fixture(scope="module")
def retriever():
    """Create one retriever instance for the test module."""
    return TfidfRetriever()


@pytest.mark.parametrize("query", RETRIEVAL_CASES)
def test_retrieval_returns_result_for_representative_queries(
    retriever,
    query,
):
    """Representative support queries should return retrieval evidence."""

    results = retriever.search(query, top_k=1)

    assert len(results) == 1

    result = results.iloc[0]

    assert isinstance(result["customer_text"], str)
    assert result["customer_text"].strip()

    assert isinstance(result["amazonhelp_text"], str)
    assert result["amazonhelp_text"].strip()

    assert 0.0 <= result["similarity"] <= 1.0


@pytest.mark.parametrize("query", RETRIEVAL_CASES)
def test_retrieval_does_not_return_empty_evidence(
    retriever,
    query,
):
    """Retrieved examples must contain both customer and support text."""

    result = retriever.search(query, top_k=1).iloc[0]

    assert result["customer_text"].strip() != ""
    assert result["amazonhelp_text"].strip() != ""


def test_retrieval_is_deterministic(retriever):
    """The same query should return the same top result."""

    query = "My payment was charged twice."

    first = retriever.search(query, top_k=1).iloc[0]
    second = retriever.search(query, top_k=1).iloc[0]

    assert first["conversation_id"] == second["conversation_id"]
    assert first["similarity"] == second["similarity"]