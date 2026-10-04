from pathlib import Path

import pandas as pd

from src.escalation.policy import EscalationPolicy
from src.generation.generator import ResponseGenerator
from src.generation.ollama_provider import OllamaProvider
from src.intents.llm_classifier import LLMIntentClassifier
from src.pipeline.agent import SupportAgent
from src.retrieval.tfidf_retriever import TfidfRetriever


INPUT_PATH = Path("golden/human_agreement_sample.csv")
OUTPUT_PATH = Path(
    "evaluation/results/human_agreement_agent_predictions.csv"
)


def build_agent() -> SupportAgent:
    classifier = LLMIntentClassifier(
        model_name="ministral-3:3b",
    )

    retriever = TfidfRetriever()

    generator = ResponseGenerator(
        OllamaProvider(),
    )

    escalation_policy = EscalationPolicy()

    return SupportAgent(
        classifier=classifier,
        retriever=retriever,
        generator=generator,
        escalation_policy=escalation_policy,
    )


def main() -> None:
    df = pd.read_csv(INPUT_PATH)

    if len(df) != 50:
        raise ValueError(
            f"Expected 50 human-agreement examples, found {len(df)}"
        )

    agent = build_agent()

    rows = []

    for index, row in df.iterrows():
        customer_text = str(row["customer_text"])

        print(
            f"[{index + 1:02d}/50] "
            f"Processing {row['ID']}..."
        )

        result = agent.handle(customer_text)

        rows.append(
            {
                "ID": row["ID"],
                "customer_text": customer_text,
                "second_annotator_intent": row[
                    "second_annotator_intent"
                ],
                "second_annotator_confidence": row[
                    "second_annotator_confidence"
                ],
                "second_annotator_ambiguous": row[
                    "second_annotator_ambiguous"
                ],
                "predicted_intent": result.intent,
                "intent_confidence": result.intent_confidence,
                "classification_reason": result.classification_reason,
                "retrieval_top1_similarity": (
                    result.retrieved_examples[0].similarity
                    if result.retrieved_examples
                    else 0.0
                ),
                "retrieval_count": len(
                    result.retrieved_examples
                ),
                "draft_response": result.draft_response,
                "generation_model": result.generation_model,
                "grounded": result.grounded,
                "should_escalate": result.escalation.should_escalate,
                "escalation_reason": result.escalation.reason,
            }
        )

    output = pd.DataFrame(rows)

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    output.to_csv(
        OUTPUT_PATH,
        index=False,
        encoding="utf-8",
    )

    print()
    print(f"Saved: {OUTPUT_PATH}")
    print(f"Rows: {len(output)}")


if __name__ == "__main__":
    main()