import numpy as np
import pandas as pd

from evaluation.intents.discovery_config import PHRASES_PATH


def discover_global_phrases(
    sample,
    vectorizer,
    matrix,
):
    """
    Find phrases that are useful across the discovery sample.

    This is exploratory evidence only.
    """
    print("\n" + "=" * 70)
    print("DISCOVERING GLOBAL PHRASES")
    print("=" * 70)

    terms = vectorizer.get_feature_names_out()

    scores = np.asarray(
        matrix.mean(axis=0)
    ).ravel()

    phrase_df = pd.DataFrame(
        {
            "phrase": terms,
            "mean_tfidf": scores,
        }
    )

    phrase_df = (
        phrase_df
        .sort_values(
            "mean_tfidf",
            ascending=False,
        )
        .reset_index(drop=True)
    )

    phrase_df.insert(
        0,
        "rank",
        range(
            1,
            len(phrase_df) + 1,
        ),
    )

    phrase_df.head(500).to_csv(
        PHRASES_PATH,
        index=False,
        encoding="utf-8",
    )

    print("\nTop 50 phrases:\n")

    print(
        phrase_df
        .head(50)
        .to_string(index=False)
    )

    print(
        f"\nSaved:\n{PHRASES_PATH}"
    )

    return phrase_df