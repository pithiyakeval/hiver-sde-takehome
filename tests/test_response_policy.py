from src.generation.response_policy import ResponsePolicy


def test_high_confidence_order_status_uses_deterministic_response():
    policy = ResponsePolicy()

    decision = policy.decide(
        intent="order_status",
        confidence=0.99,
        retrieval_similarities=[0.84, 0.82, 0.80],
    )

    assert decision.mode == ResponsePolicy.DETERMINISTIC
    assert decision.reason


def test_delivered_not_received_is_escalated():
    policy = ResponsePolicy()

    decision = policy.decide(
        intent="delivered_not_received",
        confidence=0.99,
        retrieval_similarities=[0.90, 0.85],
    )

    assert decision.mode == ResponsePolicy.ESCALATE
    assert decision.reason


def test_low_retrieval_similarity_is_escalated():
    policy = ResponsePolicy()

    decision = policy.decide(
        intent="order_status",
        confidence=0.99,
        retrieval_similarities=[0.10, 0.08],
    )

    assert decision.mode == ResponsePolicy.ESCALATE
    assert decision.reason


def test_uncertain_supported_case_uses_llm():
    policy = ResponsePolicy()

    decision = policy.decide(
        intent="product_or_device_help",
        confidence=0.70,
        retrieval_similarities=[0.70, 0.65],
    )

    assert decision.mode == ResponsePolicy.LLM
    assert decision.reason