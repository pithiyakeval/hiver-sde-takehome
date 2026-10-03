from dataclasses import dataclass
import re


@dataclass(frozen=True)
class EscalationDecision:
    should_escalate: bool
    reason: str


class EscalationPolicy:
    def __init__(
        self,
        intent_confidence_threshold: float = 0.08,
        retrieval_similarity_threshold: float = 0.25,
    ):
        self.intent_confidence_threshold = intent_confidence_threshold
        self.retrieval_similarity_threshold = retrieval_similarity_threshold

    def decide(
        self,
        intent: str,
        intent_confidence: float,
        retrieval_similarities: list[float],
        customer_message: str = "",
    ) -> EscalationDecision:

        text = customer_message.lower()

        if intent == "other_or_unclear":
            return EscalationDecision(
                True,
                "Customer request could not be mapped to a supported intent.",
            )

        if intent_confidence < self.intent_confidence_threshold:
            return EscalationDecision(
                True,
                "Intent classification confidence is below the calibrated threshold.",
            )

        if not retrieval_similarities:
            return EscalationDecision(
                True,
                "No historical resolution evidence was retrieved.",
            )

        if max(retrieval_similarities) < self.retrieval_similarity_threshold:
            return EscalationDecision(
                True,
                "Retrieved historical examples are not sufficiently similar to the customer request.",
            )

        if self._contains_financial_or_security_risk(text):
            return EscalationDecision(
                True,
                "Financial or security-sensitive issue requires human review.",
            )

        if intent == "delivered_not_received":
            return EscalationDecision(
                True,
                "Customer reports a delivery marked delivered but not received.",
            )

        if self._contains_repeated_unresolved_signal(text):
            return EscalationDecision(
                True,
                "Customer reports a repeated or prolonged unresolved support issue.",
            )

        if self._contains_long_pending_refund(text):
            return EscalationDecision(
                True,
                "Refund issue is described as prolonged or still unresolved.",
            )

        return EscalationDecision(
            False,
            "Intent confidence, historical evidence, and escalation risk signals meet the configured policy.",
        )

    @staticmethod
    def _contains_financial_or_security_risk(text: str) -> bool:
        signals = [
            "unauthorized",
            "unauthorised",
            "authorization",
            "authorisation",
            "fraud",
            "stolen card",
            "card not mine",
            "emi",
            "charged",
        ]

        return any(signal in text for signal in signals)
    @staticmethod
    def _contains_repeated_unresolved_signal(text: str) -> bool:
        patterns = [
            r"\brepeated\b",
            r"\bagain\b",
            r"\bno response\b",
            r"\bno update\b",
            r"\bstill waiting\b",
            r"\bstill unresolved\b",
            r"\bfor weeks?\b",
            r"\bfor months?\b",
            r"\bfor a year\b",
            r"\byear now\b",
        ]
        return any(re.search(pattern, text) for pattern in patterns)

    @staticmethod
    def _contains_long_pending_refund(text: str) -> bool:
        refund_terms = [
            "refund",
            "money back",
            "reimbursement",
        ]

        pending_terms = [
            "pending",
            "still waiting",
            "not received",
            "no update",
            "long pending",
            "haven't got",
            "have not got",
        ]

        has_refund_term = any(term in text for term in refund_terms)
        has_pending_term = any(term in text for term in pending_terms)

        return has_refund_term and has_pending_term