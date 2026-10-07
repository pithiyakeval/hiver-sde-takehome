import time

from src.intents.llm_classifier import LLMIntentClassifier


MESSAGES = [
    "Where is my order?",
    "My package says delivered but I never received it.",
    "I was charged twice for the same order and need help.",
]


def main() -> None:
    classifier = LLMIntentClassifier()

    print("\n=== Intent Classifier Benchmark ===\n")

    for message in MESSAGES:
        start = time.perf_counter()

        result = classifier.predict(message)

        elapsed_ms = (time.perf_counter() - start) * 1000

        print(f"Message: {message}")
        print(f"Intent: {result.intent}")
        print(f"Confidence: {result.confidence:.4f}")
        print(f"Reason: {result.reason}")
        print(f"Latency: {elapsed_ms:.2f} ms")
        print("-" * 70)


if __name__ == "__main__":
    main()