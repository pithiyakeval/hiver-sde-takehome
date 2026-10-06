from dataclasses import dataclass
import time

from src.escalation.policy import EscalationDecision, EscalationPolicy
from src.generation.generator import ResponseGenerator
from src.generation.models import GenerationInput, RetrievedExample
from src.generation.response_policy import ResponsePolicy
from src.retrieval.tfidf_retriever import TfidfRetriever


@dataclass(frozen=True)
class AgentTiming:
    classification_ms: float
    retrieval_ms: float
    generation_ms: float
    escalation_ms: float
    total_ms: float


@dataclass(frozen=True)
class AgentResult:
    customer_message: str
    intent: str
    intent_confidence: float
    classification_reason: str
    retrieved_examples: list[RetrievedExample]
    draft_response: str
    generation_model: str
    grounded: bool
    escalation: EscalationDecision
    timing: AgentTiming


class SupportAgent:
    """Orchestrates classification, retrieval, generation, and escalation."""

    def __init__(
        self,
        classifier,
        retriever: TfidfRetriever,
        generator: ResponseGenerator,
        escalation_policy: EscalationPolicy,
        response_policy: ResponsePolicy,
        retrieval_top_k: int = 3,
    ):
        self.classifier = classifier
        self.retriever = retriever
        self.generator = generator
        self.escalation_policy = escalation_policy
        self.response_policy = response_policy
        self.retrieval_top_k = retrieval_top_k

    def handle(self, customer_message: str) -> AgentResult:
        total_start = time.perf_counter()

        # -------------------------
        # Classification
        # -------------------------
        classification_start = time.perf_counter()

        classification = self.classifier.predict(customer_message)

        classification_ms = (
            time.perf_counter() - classification_start
        ) * 1000

        # -------------------------
        # Retrieval
        # -------------------------
        retrieval_start = time.perf_counter()

        retrieved = self.retriever.search(
            customer_message,
            top_k=self.retrieval_top_k,
        )

        retrieval_ms = (
            time.perf_counter() - retrieval_start
        ) * 1000

        examples = [
            RetrievedExample(
                customer_text=row["customer_text"],
                historical_response=row["amazonhelp_text"],
                similarity=float(row["similarity"]),
            )
            for _, row in retrieved.iterrows()
        ]

        # -------------------------
        # Response policy
        # -------------------------
        response_policy_decision = self.response_policy.decide(
            intent=classification.intent,
            confidence=classification.confidence,
            retrieval_similarities=[
                example.similarity
                for example in examples
            ],
        )

        # -------------------------
        # Generation
        # -------------------------
        generation_start = time.perf_counter()

        if response_policy_decision.mode == ResponsePolicy.LLM:
            generation_input = GenerationInput(
                customer_message=customer_message,
                intent=classification.intent,
                retrieved_examples=examples[:1],
            )

            generation_result = self.generator.generate(
                generation_input
            )

        elif response_policy_decision.mode == ResponsePolicy.DETERMINISTIC:
            generation_result = self.generator.generate_deterministic(
                customer_message=customer_message,
                intent=classification.intent,
                historical_response=(
                    examples[0].historical_response
                    if examples
                    else ""
                ),
            )

        else:
            generation_result = self.generator.generate_fallback(
                customer_message=customer_message,
            )

        generation_ms = (
            time.perf_counter() - generation_start
        ) * 1000

        # -------------------------
        # Escalation
        # -------------------------
        escalation_start = time.perf_counter()

        escalation = self.escalation_policy.decide(
            intent=classification.intent,
            intent_confidence=classification.confidence,
            retrieval_similarities=[
                example.similarity
                for example in examples
            ],
            customer_message=customer_message,
        )

        escalation_ms = (
            time.perf_counter() - escalation_start
        ) * 1000

        # -------------------------
        # Total
        # -------------------------
        total_ms = (
            time.perf_counter() - total_start
        ) * 1000

        timing = AgentTiming(
            classification_ms=round(classification_ms, 2),
            retrieval_ms=round(retrieval_ms, 2),
            generation_ms=round(generation_ms, 2),
            escalation_ms=round(escalation_ms, 2),
            total_ms=round(total_ms, 2),
        )

        # -------------------------
        # Final result
        # -------------------------
        return AgentResult(
            customer_message=customer_message,
            intent=classification.intent,
            intent_confidence=classification.confidence,
            classification_reason=classification.reason,
            retrieved_examples=examples,
            draft_response=generation_result.draft_response,
            generation_model=generation_result.model,
            grounded=generation_result.grounded,
            escalation=escalation,
            timing=timing,
        )