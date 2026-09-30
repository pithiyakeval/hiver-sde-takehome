import pandas as pd

from evaluation.intents.discovery_config import REVIEW_PATH


def create_review_file(
    sample,
    topic_df,
    examples_df,
):
    """
    Create a human-review worksheet combining
    topic information with representative examples.
    """
    review_rows = []

    for _, topic in topic_df.iterrows():
        topic_name = topic["topic"]

        topic_examples = examples_df[
            examples_df["topic"] == topic_name
        ].head(20)

        for _, example in topic_examples.iterrows():
            review_rows.append(
                {
                    "topic": topic_name,
                    "discovery_rank": topic[
                        "discovery_rank"
                    ],
                    "topic_share": topic[
                        "topic_share"
                    ],
                    "top_terms": topic[
                        "top_terms"
                    ],
                    "example_rank": example[
                        "example_rank"
                    ],
                    "customer_tweet_id": example[
                        "customer_tweet_id"
                    ],
                    "customer_text": example[
                        "customer_text"
                    ],
                    "human_intent": "",
                    "notes": "",
                }
            )

    review_df = pd.DataFrame(review_rows)

    review_df.to_csv(
        REVIEW_PATH,
        index=False,
        encoding="utf-8",
    )

    print(
        f"\nReview worksheet:\n{REVIEW_PATH}"
    )

    return review_df