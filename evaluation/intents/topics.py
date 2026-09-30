import numpy as np
import pandas as pd
from sklearn.decomposition import NMF

from evaluation.intents.discovery_config import (
    N_TOPICS,
    RANDOM_STATE,
    TOP_TERMS_PER_TOPIC,
    EXAMPLES_PER_TOPIC,
    TOPICS_PATH,
    EXAMPLES_PATH,
)


def discover_topics(
    sample,
    vectorizer,
    matrix,
):
    """
    Discover interpretable candidate themes with NMF.

    IMPORTANT:

    NMF topics are NOT final support intents.

    They are evidence used to help a human define the final
    intent taxonomy.
    """
    print("\n" + "=" * 70)
    print(
        f"DISCOVERING {N_TOPICS} CANDIDATE TOPICS WITH NMF"
    )
    print("=" * 70)

    model = NMF(
        n_components=N_TOPICS,
        init="nndsvda",
        random_state=RANDOM_STATE,
        max_iter=500,
        l1_ratio=0.1,
    )

    document_topic_matrix = model.fit_transform(matrix)
    topic_term_matrix = model.components_
    terms = vectorizer.get_feature_names_out()

    topic_rows = []
    example_rows = []

    topic_strength = document_topic_matrix.sum(axis=0)

    topic_order = np.argsort(topic_strength)[::-1]

    for display_rank, topic_index in enumerate(
        topic_order,
        start=1,
    ):
        weights = topic_term_matrix[topic_index]

        top_indices = weights.argsort()[::-1][
            :TOP_TERMS_PER_TOPIC
        ]

        top_terms = [
            terms[index]
            for index in top_indices
        ]

        topic_scores = document_topic_matrix[
            :,
            topic_index,
        ]

        example_indices = topic_scores.argsort()[
            ::-1
        ][:EXAMPLES_PER_TOPIC]

        topic_mass = float(
            topic_strength[topic_index]
        )

        total_mass = float(
            topic_strength.sum()
        )

        topic_share = (
            topic_mass / total_mass
            if total_mass > 0
            else 0.0
        )

        topic_rows.append(
            {
                "topic": f"topic_{topic_index + 1}",
                "discovery_rank": display_rank,
                "topic_share": topic_share,
                "topic_mass": topic_mass,
                "top_terms": " | ".join(
                    top_terms
                ),
            }
        )

        for example_rank, row_index in enumerate(
            example_indices,
            start=1,
        ):
            row = sample.iloc[row_index]

            example_rows.append(
                {
                    "topic": f"topic_{topic_index + 1}",
                    "discovery_rank": display_rank,
                    "example_rank": example_rank,
                    "topic_score": float(
                        topic_scores[row_index]
                    ),
                    "customer_tweet_id": row[
                        "customer_tweet_id"
                    ],
                    "customer_text": row[
                        "customer_text"
                    ],
                }
            )

    topic_df = (
        pd.DataFrame(topic_rows)
        .sort_values("discovery_rank")
        .reset_index(drop=True)
    )

    examples_df = (
        pd.DataFrame(example_rows)
        .sort_values(
            [
                "discovery_rank",
                "example_rank",
            ]
        )
        .reset_index(drop=True)
    )

    topic_df.to_csv(
        TOPICS_PATH,
        index=False,
        encoding="utf-8",
    )

    examples_df.to_csv(
        EXAMPLES_PATH,
        index=False,
        encoding="utf-8",
    )

    print("\nCandidate topic summary:\n")

    for _, row in topic_df.iterrows():
        print(
            f"{row['topic']} "
            f"(share={row['topic_share']:.2%})"
        )

        print(
            f"  {row['top_terms']}"
        )

        topic_examples = examples_df[
            examples_df["topic"] == row["topic"]
        ].head(5)

        for _, example in topic_examples.iterrows():
            text = str(
                example["customer_text"]
            ).replace("\n", " ")

            print(
                f"  - {text}"
            )

        print()

    print(
        f"Saved topic summary:\n{TOPICS_PATH}"
    )

    print(
        f"Saved topic examples:\n{EXAMPLES_PATH}"
    )

    return topic_df, examples_df