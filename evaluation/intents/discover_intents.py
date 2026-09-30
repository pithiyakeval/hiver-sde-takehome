"""
AmazonHelp Intent Discovery Pipeline

Orchestrates:
1. Data loading
2. Customer-message preparation
3. Discovery sampling
4. TF-IDF representation
5. Global phrase discovery
6. NMF topic discovery
7. Human-review worksheet creation
8. Discovery validation
"""

from evaluation.intents.data_loader import (
    load_data,
    prepare_customer_messages,
)
from evaluation.intents.sampling import create_discovery_sample
from evaluation.intents.tfidf import build_tfidf
from evaluation.intents.phrases import discover_global_phrases
from evaluation.intents.topics import discover_topics
from evaluation.intents.review import create_review_file
from evaluation.intents.validation import print_validation

from evaluation.intents.discovery_config import (
    SAMPLE_PATH,
    PHRASES_PATH,
    TOPICS_PATH,
    EXAMPLES_PATH,
    REVIEW_PATH,
)


def main():
    """Run the complete intent-discovery pipeline."""

    # --------------------------------------------------------
    # 1. Load dataset
    # --------------------------------------------------------
    df = load_data()

    # --------------------------------------------------------
    # 2. Prepare unique customer messages
    # --------------------------------------------------------
    customer_df = prepare_customer_messages(df)

    # --------------------------------------------------------
    # 3. Create discovery sample
    # --------------------------------------------------------
    sample = create_discovery_sample(
        customer_df
    )

    # --------------------------------------------------------
    # 4. Build TF-IDF representation
    # --------------------------------------------------------
    vectorizer, matrix = build_tfidf(
        sample
    )

    # --------------------------------------------------------
    # 5. Discover global phrases
    # --------------------------------------------------------
    discover_global_phrases(
        sample,
        vectorizer,
        matrix,
    )

    # --------------------------------------------------------
    # 6. Discover candidate topics
    # --------------------------------------------------------
    topic_df, examples_df = discover_topics(
        sample,
        vectorizer,
        matrix,
    )

    # --------------------------------------------------------
    # 7. Create human-review worksheet
    # --------------------------------------------------------
    create_review_file(
        sample,
        topic_df,
        examples_df,
    )

    # --------------------------------------------------------
    # 8. Validate discovery results
    # --------------------------------------------------------
    print_validation(
        customer_df,
        sample,
        topic_df,
    )

    # --------------------------------------------------------
    # 9. Completion summary
    # --------------------------------------------------------
    print_completion_summary()


def print_completion_summary():
    """Print generated artifacts and next action."""

    print("\n" + "=" * 70)
    print("INTENT DISCOVERY COMPLETE")
    print("=" * 70)

    print("\nGenerated files:")

    generated_files = [
        SAMPLE_PATH,
        PHRASES_PATH,
        TOPICS_PATH,
        EXAMPLES_PATH,
        REVIEW_PATH,
        
    ]

    for index, path in enumerate(
        generated_files,
        start=1,
    ):
        print(f"{index}. {path}")

    print("\nNext:")
    print(
        "Review the candidate topics and examples, "
        "then define the final 8–12 human-labelled intents."
    )


if __name__ == "__main__":
    main()