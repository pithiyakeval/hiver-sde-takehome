from dataclasses import dataclass


@dataclass(frozen=True)
class ResponsePolicyDecision:
    """
    Determines how a support response should be produced.
    """

    mode: str
    reason: str


class ResponsePolicy:
    """
    Decides whether a case should use:
    - deterministic response generation
    - LLM generation
    - escalation
    """

    DETERMINISTIC = "deterministic"
    LLM = "llm"
    ESCALATE = "escalate"

    MIN_DETERMINISTIC_CONFIDENCE = 0.95
    MIN_LLM_CONFIDENCE = 0.45
    MIN_RETRIEVAL_SIMILARITY = 0.25

    SAFE_DETERMINISTIC_INTENTS = {
        "order_status",
        "delivery_late",
        "delivery_promise",
        "prime_membership",
    }

    ALWAYS_ESCALATE_INTENTS = {
        "delivered_not_received",
    }

    def decide(
        self,
        *,
        intent: str,
        confidence: float,
        retrieval_similarities: list[float],
    ) -> ResponsePolicyDecision:
        """
        Select the safest response-generation strategy.
        """

        if intent in self.ALWAYS_ESCALATE_INTENTS:
            return ResponsePolicyDecision(
                mode=self.ESCALATE,
                reason=(
                    "The intent requires case-specific verification "
                    "that is not available to the AI agent."
                ),
            )

        if not retrieval_similarities:
            return ResponsePolicyDecision(
                mode=self.ESCALATE,
                reason=(
                    "No historical support evidence was retrieved "
                    "for the customer request."
                ),
            )

        top_similarity = max(retrieval_similarities)

        if top_similarity < self.MIN_RETRIEVAL_SIMILARITY:
            return ResponsePolicyDecision(
                mode=self.ESCALATE,
                reason=(
                    "Retrieved historical support evidence is "
                    "below the minimum similarity threshold."
                ),
            )

        if (
            intent in self.SAFE_DETERMINISTIC_INTENTS
            and confidence >= self.MIN_DETERMINISTIC_CONFIDENCE
        ):
            return ResponsePolicyDecision(
                mode=self.DETERMINISTIC,
                reason=(
                    "High-confidence supported intent with sufficient "
                    "historical evidence can use a deterministic response."
                ),
            )

        if confidence < self.MIN_LLM_CONFIDENCE:
            return ResponsePolicyDecision(
                mode=self.ESCALATE,
                reason=(
                    "Intent confidence is too low for reliable "
                    "response generation."
                ),
            )

        return ResponsePolicyDecision(
            mode=self.LLM,
            reason=(
                "The case is suitable for controlled LLM response "
                "generation using retrieved historical evidence."
            ),
        )