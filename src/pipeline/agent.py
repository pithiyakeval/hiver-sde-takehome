from dataclasses import dataclass

from src.escalation.policy import EscalationDecision, EscalationPolicy
from src.generation.generator import ResponseGenerator
from src.generation.models import GenerationInput, RetrievedExample
from src.intents.classifier import IntentClassifier
from src.retrieval.tfidf_retriever import TfidfRetriever


@dataclass(frozen=True)
class AgentResult:
    customer_message: str
    intent: str
    intent_confidence: float
    retrieved_examples: list[RetrievedExample]
    draft_response: str
    generation_model: str
    grounded: bool
    escalation: EscalationDecision


class SupportAgent:
    """Orchestrates classification, retrieval, generation, and escalation."""

    def __init__(
        self,
        classifier: IntentClassifier,
        retriever: TfidfRetriever,
        generator: ResponseGenerator,
        escalation_policy: EscalationPolicy,
        retrieval_top_k: int = 3,
    ):
        self.classifier = classifier
        self.retriever = retriever
        self.generator = generator
        self.escalation_policy = escalation_policy
        self.retrieval_top_k = retrieval_top_k

    def handle(self, customer_message: str) -> AgentResult:
        classification = self.classifier.predict(customer_message)

        retrieved = self.retriever.search(
            customer_message,
            top_k=self.retrieval_top_k,
        )

        examples = [
            RetrievedExample(
                customer_text=row["customer_text"],
                historical_response=row["amazonhelp_text"],
                similarity=float(row["similarity"]),
            )
            for _, row in retrieved.iterrows()
        ]

        generation_input = GenerationInput(
            customer_message=customer_message,
            intent=classification["intent"],
            retrieved_examples=examples,
        )

        generation_result = self.generator.generate(generation_input)

        escalation = self.escalation_policy.decide(
            intent=classification["intent"],
            intent_confidence=classification["confidence"],
            retrieval_similarities=[
                example.similarity for example in examples
            ],
            customer_message=customer_message,
        )

        return AgentResult(
            customer_message=customer_message,
            intent=classification["intent"],
            intent_confidence=classification["confidence"],
            retrieved_examples=examples,
            draft_response=generation_result.draft_response,
            generation_model=generation_result.model,
            grounded=generation_result.grounded,
            escalation=escalation,
        )