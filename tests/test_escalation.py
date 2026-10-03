from src.escalation.policy import EscalationPolicy


def test_unclear_intent_escalates():
    policy = EscalationPolicy()

    result = policy.decide(
        intent="other_or_unclear",
        intent_confidence=0.90,
        retrieval_similarities=[0.80, 0.70, 0.60],
    )

    assert result.should_escalate is True
    assert "could not be mapped" in result.reason


def test_low_intent_confidence_escalates():
    policy = EscalationPolicy()

    result = policy.decide(
        intent="delivery_late",
        intent_confidence=0.0,
        retrieval_similarities=[0.80, 0.70, 0.60],
    )

    assert result.should_escalate is True
    assert "confidence" in result.reason


def test_missing_retrieval_evidence_escalates():
    policy = EscalationPolicy()

    result = policy.decide(
        intent="delivery_late",
        intent_confidence=0.80,
        retrieval_similarities=[],
    )

    assert result.should_escalate is True
    assert "evidence" in result.reason


def test_weak_retrieval_evidence_escalates():
    policy = EscalationPolicy()

    result = policy.decide(
        intent="delivery_late",
        intent_confidence=0.80,
        retrieval_similarities=[0.15, 0.12, 0.10],
    )

    assert result.should_escalate is True
    assert "similar" in result.reason


def test_confident_supported_case_does_not_escalate():
    policy = EscalationPolicy()

    result = policy.decide(
        intent="delivery_late",
        intent_confidence=0.85,
        retrieval_similarities=[0.60, 0.45, 0.35],
    )

    assert result.should_escalate is False
    assert "meet" in result.reason

def test_financial_security_issue_escalates():
    policy = EscalationPolicy()

    result = policy.decide(
        intent="payment_or_charge",
        intent_confidence=0.90,
        retrieval_similarities=[0.80, 0.70, 0.60],
        customer_message="Someone used my card without authorization.",
    )

    assert result.should_escalate is True
    assert "Financial or security" in result.reason


def test_delivered_not_received_escalates():
    policy = EscalationPolicy()

    result = policy.decide(
        intent="delivered_not_received",
        intent_confidence=0.90,
        retrieval_similarities=[0.80, 0.70, 0.60],
        customer_message="Tracking says delivered but I never received it.",
    )

    assert result.should_escalate is True
    assert "delivered but not received" in result.reason


def test_repeated_unresolved_issue_escalates():
    policy = EscalationPolicy()

    result = policy.decide(
        intent="customer_service_followup",
        intent_confidence=0.90,
        retrieval_similarities=[0.80, 0.70, 0.60],
        customer_message="I contacted support again and still have no response after weeks.",
    )

    assert result.should_escalate is True
    assert "repeated or prolonged" in result.reason


def test_long_pending_refund_escalates():
    policy = EscalationPolicy()

    result = policy.decide(
        intent="return_or_refund",
        intent_confidence=0.90,
        retrieval_similarities=[0.80, 0.70, 0.60],
        customer_message="My refund is still pending and I have been waiting for weeks.",
    )

    assert result.should_escalate is True
    assert "prolonged" in result.reason


def test_normal_supported_case_does_not_escalate():
    policy = EscalationPolicy()

    result = policy.decide(
        intent="order_status",
        intent_confidence=0.90,
        retrieval_similarities=[0.80, 0.70, 0.60],
        customer_message="Where is my order?",
    )

    assert result.should_escalate is False