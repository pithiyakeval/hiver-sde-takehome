from pathlib import Path

import pandas as pd

from src.escalation.policy import EscalationPolicy
from src.generation.generator import ResponseGenerator
from src.generation.ollama_provider import OllamaProvider
from src.intents.classifier import IntentClassifier
from src.pipeline.agent import SupportAgent
from src.retrieval.tfidf_retriever import TfidfRetriever


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[1]

INPUT_PATH = PROJECT_ROOT / "golden" / "evaluation_test.csv"
OUTPUT_PATH = (
    PROJECT_ROOT
    / "evaluation"
    / "results"
    / "agent_predictions.csv"
)


# ---------------------------------------------------------------------------
# Agent construction
# ---------------------------------------------------------------------------

def build_agent() -> SupportAgent:
    """
    Build the complete production support agent.

    Components:
    - IntentClassifier: predicts the customer intent.
    - TfidfRetriever: retrieves similar historical support cases.
    - OllamaProvider: generates the response using the local Ministral model.
    - EscalationPolicy: determines whether the case should be escalated.
    """

    classifier = IntentClassifier()
    retriever = TfidfRetriever()

    # Uses the configured local Ollama model.
    # Default model: ministral-3:3b
    generator = ResponseGenerator(
        OllamaProvider()
    )

    escalation_policy = EscalationPolicy()

    return SupportAgent(
        classifier=classifier,
        retriever=retriever,
        generator=generator,
        escalation_policy=escalation_policy,
    )


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def evaluate() -> None:
    """
    Run the support agent on the frozen evaluation test set.

    The evaluation set is never modified by this script. Predictions and
    supporting signals are written to a separate results file.
    """

    if not INPUT_PATH.exists():
        raise FileNotFoundError(
            f"Evaluation dataset not found: {INPUT_PATH}"
        )

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    test_df = pd.read_csv(INPUT_PATH)

    required_columns = {"ID", "customer_text", "Intent"}
    missing_columns = required_columns - set(test_df.columns)

    if missing_columns:
        raise ValueError(
            "Evaluation dataset is missing required columns: "
            f"{sorted(missing_columns)}"
        )

    agent = build_agent()

    rows = []

    for index, (_, row) in enumerate(test_df.iterrows(), start=1):
        print(
            f"[{index}/{len(test_df)}] "
            f"Running example {row['ID']}...",
            flush=True,
        )

        customer_text = str(row["customer_text"]).strip()

        if not customer_text:
            raise ValueError(
                f"Empty customer message found for example ID {row['ID']}"
            )

        result = agent.handle(customer_text)

        top_similarity = (
            result.retrieved_examples[0].similarity
            if result.retrieved_examples
            else 0.0
        )

        rows.append(
            {
                "example_id": row["ID"],
                "customer_text": customer_text,
                "gold_intent": row["Intent"],
                "predicted_intent": result.intent,
                "intent_confidence": result.intent_confidence,
                "retrieval_top1_similarity": top_similarity,
                "retrieval_count": len(result.retrieved_examples),
                "draft_response": result.draft_response,
                "grounded": result.grounded,
                "should_escalate": result.escalation.should_escalate,
                "escalation_reason": result.escalation.reason,
            }
        )

    predictions = pd.DataFrame(rows)

    predictions.to_csv(
        OUTPUT_PATH,
        index=False,
    )

    print("=" * 72)
    print("Support Agent Evaluation")
    print("=" * 72)
    print(f"Evaluation examples : {len(predictions)}")
    print(f"LLM provider        : Ollama / configured local model")
    print(f"Input               : {INPUT_PATH}")
    print(f"Output              : {OUTPUT_PATH}")
    print("=" * 72)


if __name__ == "__main__":
    evaluate()