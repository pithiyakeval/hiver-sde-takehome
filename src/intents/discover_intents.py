import re
from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.decomposition import NMF
from sklearn.feature_extraction.text import TfidfVectorizer


# ============================================================
# Configuration
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

INPUT_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "amazonhelp_conversations.csv"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "intent_discovery"
)

SAMPLE_PATH = (
    OUTPUT_DIR
    / "intent_discovery_sample.csv"
)

PHRASES_PATH = (
    OUTPUT_DIR
    / "top_phrases.csv"
)

TOPICS_PATH = (
    OUTPUT_DIR
    / "topic_summary.csv"
)

EXAMPLES_PATH = (
    OUTPUT_DIR
    / "topic_examples.csv"
)

REVIEW_PATH = (
    OUTPUT_DIR
    / "intent_review.csv"
)


# ============================================================
# Reproducibility / discovery settings
# ============================================================

RANDOM_STATE = 42

# Number of unique customer messages used for discovery.
DISCOVERY_SAMPLE_SIZE = 5_000

# Candidate themes.
#
# IMPORTANT:
# These are NOT the final intents.
# They are exploratory topics that we will inspect.
N_TOPICS = 12

# Number of representative examples per topic.
EXAMPLES_PER_TOPIC = 20

# Number of phrases shown for each topic.
TOP_TERMS_PER_TOPIC = 15

# TF-IDF vocabulary limits.
MIN_DOCUMENT_FREQUENCY = 5
MAX_DOCUMENT_FREQUENCY = 0.85
MAX_FEATURES = 20_000


# ============================================================
# Text normalization
# ============================================================

# Generic Twitter/support words that are usually not useful
# for identifying the customer's actual problem.
CUSTOM_STOPWORDS = {
    "amazon",
    "amazonhelp",
    "help",
    "please",
    "thanks",
    "thank",
    "hi",
    "hello",
    "hey",
    "customer",
    "service",
    "support",
    "team",
    "guys",
    "need",
    "want",
    "just",
    "like",
    "really",
    "today",
    "now",
    "know",
    "got",
    "get",
    "getting",
    "did",
    "does",
    "dont",
    "didnt",
    "cant",
    "could",
    "would",
    "im",
    "ive",
    "id",
    "youre",
    "weve",
    "amp",
    "https",
    "http",
    "www",
}


# Common Twitter artifacts / contractions.
CONTRACTION_REPLACEMENTS = {
    "can't": "cannot",
    "won't": "willnot",
    "don't": "do not",
    "didn't": "did not",
    "doesn't": "does not",
    "isn't": "is not",
    "aren't": "are not",
    "wasn't": "was not",
    "weren't": "were not",
    "haven't": "have not",
    "hasn't": "has not",
    "hadn't": "had not",
    "wouldn't": "would not",
    "couldn't": "could not",
    "shouldn't": "should not",
    "can't": "cannot",
    "i'm": "i am",
    "i've": "i have",
    "i'll": "i will",
    "i'd": "i would",
    "you're": "you are",
    "you've": "you have",
    "you'll": "you will",
    "we're": "we are",
    "we've": "we have",
    "they're": "they are",
}


def clean_text(text):
    """
    Clean customer text for topic discovery.

    The original customer_text is never overwritten.
    This function creates only the analysis representation.
    """

    if pd.isna(text):
        return ""

    text = str(text)

    # Lowercase.
    text = text.lower()

    # Expand common contractions before removing punctuation.
    for old, new in CONTRACTION_REPLACEMENTS.items():
        text = text.replace(old, new)

    # Remove URLs.
    text = re.sub(
        r"https?://\S+|www\.\S+",
        " ",
        text,
    )

    # Remove Twitter mentions.
    text = re.sub(
        r"@\w+",
        " ",
        text,
    )

    # Remove hashtags but preserve the word.
    text = re.sub(
        r"#(\w+)",
        r"\1",
        text,
    )

    # Remove email addresses.
    text = re.sub(
        r"\S+@\S+",
        " ",
        text,
    )

    # Remove HTML-ish entities.
    text = re.sub(
        r"&\w+;",
        " ",
        text,
    )

    # Keep alphabetic characters and spaces.
    text = re.sub(
        r"[^a-z\s]",
        " ",
        text,
    )

    # Normalize whitespace.
    text = re.sub(
        r"\s+",
        " ",
        text,
    )

    return text.strip()


# ============================================================
# Load data
# ============================================================

def load_data():
    print("=" * 70)
    print("AMAZONHELP INTENT DISCOVERY")
    print("=" * 70)

    print("\nInput:")
    print(INPUT_PATH)

    if not INPUT_PATH.exists():
        raise FileNotFoundError(
            f"\nConversation file not found:\n{INPUT_PATH}\n\n"
            "Run this first:\n"
            "python -m src.data.conversations"
        )

    df = pd.read_csv(
        INPUT_PATH,
        low_memory=False,
    )

    print(
        f"\nConversation/reply rows loaded: "
        f"{len(df):,}"
    )

    required_columns = [
        "customer_tweet_id",
        "customer_author_id",
        "customer_created_at",
        "customer_text",
    ]

    missing = [
        column
        for column in required_columns
        if column not in df.columns
    ]

    if missing:
        raise ValueError(
            f"Missing required columns: {missing}"
        )

    return df


# ============================================================
# Prepare unique customer messages
# ============================================================

def prepare_customer_messages(df):
    """
    Keep one row per customer tweet.

    A customer tweet may have more than one AmazonHelp reply.
    For intent discovery, the customer tweet itself is the unit
    of analysis.
    """

    customer_df = (
        df[
            [
                "customer_tweet_id",
                "customer_author_id",
                "customer_created_at",
                "customer_text",
            ]
        ]
        .drop_duplicates(
            subset="customer_tweet_id"
        )
        .copy()
    )

    customer_df["customer_text"] = (
        customer_df["customer_text"]
        .fillna("")
        .astype(str)
        .str.strip()
    )

    # Remove empty customer messages.
    customer_df = customer_df[
        customer_df["customer_text"].str.len() > 0
    ].copy()

    # Analysis text.
    customer_df["clean_text"] = (
        customer_df["customer_text"]
        .map(clean_text)
    )

    # Remove empty cleaned messages.
    customer_df = customer_df[
        customer_df["clean_text"].str.len() > 0
    ].copy()

    # Very short messages are generally poor candidates
    # for intent discovery.
    customer_df = customer_df[
        customer_df["clean_text"].str.len() >= 10
    ].copy()

    customer_df["text_length"] = (
        customer_df["clean_text"].str.len()
    )

    customer_df = customer_df.reset_index(
        drop=True
    )

    print(
        f"Unique customer tweets available: "
        f"{len(customer_df):,}"
    )

    return customer_df


# ============================================================
# Stratified discovery sample
# ============================================================

def create_discovery_sample(customer_df):
    """
    Build a reproducible stratified sample.

    We stratify by:
        1. message length
        2. approximate message volume

    Length stratification prevents the discovery set from being
    dominated by one particular style of Twitter message.
    """

    print("\n" + "=" * 70)
    print("CREATING DISCOVERY SAMPLE")
    print("=" * 70)

    df = customer_df.copy()

    target_size = min(
        DISCOVERY_SAMPLE_SIZE,
        len(df),
    )

    # Five quantile-based length groups.
    df["length_bucket"] = pd.qcut(
        df["text_length"],
        q=5,
        labels=[
            "very_short",
            "short",
            "medium",
            "long",
            "very_long",
        ],
        duplicates="drop",
    )

    bucket_count = (
        df["length_bucket"]
        .nunique()
    )

    per_bucket = target_size // bucket_count

    sample_parts = []

    for bucket in sorted(
        df["length_bucket"]
        .dropna()
        .unique()
    ):

        bucket_df = df[
            df["length_bucket"] == bucket
        ]

        n = min(
            per_bucket,
            len(bucket_df),
        )

        sample_parts.append(
            bucket_df.sample(
                n=n,
                random_state=RANDOM_STATE,
            )
        )

    sample = pd.concat(
        sample_parts,
        ignore_index=True,
    )

    # Fill any remainder.
    remaining = target_size - len(sample)

    if remaining > 0:

        selected_ids = set(
            sample["customer_tweet_id"]
        )

        remaining_df = df[
            ~df["customer_tweet_id"].isin(
                selected_ids
            )
        ]

        if not remaining_df.empty:

            extra = remaining_df.sample(
                n=min(
                    remaining,
                    len(remaining_df),
                ),
                random_state=RANDOM_STATE,
            )

            sample = pd.concat(
                [
                    sample,
                    extra,
                ],
                ignore_index=True,
            )

    sample = sample.sample(
        frac=1,
        random_state=RANDOM_STATE,
    ).reset_index(drop=True)

    sample.insert(
        0,
        "discovery_id",
        range(
            1,
            len(sample) + 1,
        ),
    )

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    sample[
        [
            "discovery_id",
            "customer_tweet_id",
            "customer_author_id",
            "customer_created_at",
            "customer_text",
            "clean_text",
            "text_length",
        ]
    ].to_csv(
        SAMPLE_PATH,
        index=False,
        encoding="utf-8",
    )

    print(
        f"Discovery sample size: "
        f"{len(sample):,}"
    )

    print(
        f"Saved:\n{SAMPLE_PATH}"
    )

    return sample


# ============================================================
# TF-IDF representation
# ============================================================

def build_tfidf(sample):
    """
    Convert the discovery sample into a TF-IDF matrix.

    Unigrams + bigrams make the discovered topics easier
    to interpret.
    """

    print("\n" + "=" * 70)
    print("BUILDING TF-IDF REPRESENTATION")
    print("=" * 70)

    vectorizer = TfidfVectorizer(
        stop_words=list(CUSTOM_STOPWORDS),
        ngram_range=(1, 2),
        min_df=MIN_DOCUMENT_FREQUENCY,
        max_df=MAX_DOCUMENT_FREQUENCY,
        max_features=MAX_FEATURES,
        sublinear_tf=True,
        strip_accents="unicode",
    )

    matrix = vectorizer.fit_transform(
        sample["clean_text"]
    )

    print(
        f"Documents: {matrix.shape[0]:,}"
    )

    print(
        f"Vocabulary size: "
        f"{matrix.shape[1]:,}"
    )

    return vectorizer, matrix


# ============================================================
# Global phrase discovery
# ============================================================

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
        .to_string(
            index=False
        )
    )

    print(
        f"\nSaved:\n{PHRASES_PATH}"
    )

    return phrase_df


# ============================================================
# NMF topic discovery
# ============================================================

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

    document_topic_matrix = (
        model.fit_transform(matrix)
    )

    topic_term_matrix = model.components_

    terms = vectorizer.get_feature_names_out()

    topic_rows = []
    example_rows = []

    # --------------------------------------------------------
    # Topic-level information
    # --------------------------------------------------------

    topic_strength = (
        document_topic_matrix.sum(
            axis=0
        )
    )

    topic_order = np.argsort(
        topic_strength
    )[::-1]

    for display_rank, topic_index in enumerate(
        topic_order,
        start=1,
    ):

        weights = topic_term_matrix[
            topic_index
        ]

        top_indices = weights.argsort()[
            ::-1
        ][:TOP_TERMS_PER_TOPIC]

        top_terms = [
            terms[index]
            for index in top_indices
        ]

        # Documents most strongly associated
        # with this topic.
        topic_scores = (
            document_topic_matrix[
                :,
                topic_index,
            ]
        )

        example_indices = topic_scores.argsort()[
            ::-1
        ][:EXAMPLES_PER_TOPIC]

        topic_mass = float(
            topic_strength[
                topic_index
            ]
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
        .sort_values(
            "discovery_rank"
        )
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

    # --------------------------------------------------------
    # Console output
    # --------------------------------------------------------

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
            examples_df["topic"]
            == row["topic"]
        ].head(5)

        for _, example in topic_examples.iterrows():

            text = str(
                example["customer_text"]
            ).replace(
                "\n",
                " ",
            )

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


# ============================================================
# Create human-review worksheet
# ============================================================

def create_review_file(
    sample,
    examples_df,
    topic_df,
):
    """
    Create a practical worksheet for human review.

    We intentionally leave the final intent fields blank.

    This file becomes the bridge between unsupervised discovery
    and the final human-defined taxonomy.
    """

    print("\n" + "=" * 70)
    print("CREATING HUMAN REVIEW WORKSHEET")
    print("=" * 70)

    # Use all representative examples.
    review_df = examples_df.copy()

    topic_lookup = topic_df[
        [
            "topic",
            "top_terms",
            "topic_share",
        ]
    ]

    review_df = review_df.merge(
        topic_lookup,
        on="topic",
        how="left",
        suffixes=(
            "",
            "_summary",
        ),
    )

    # Fields we will fill after manual inspection.
    review_df["candidate_intent"] = ""
    review_df["confidence"] = ""
    review_df["notes"] = ""

    review_df = review_df[
        [
            "topic",
            "discovery_rank",
            "example_rank",
            "topic_score",
            "topic_share",
            "top_terms",
            "customer_tweet_id",
            "customer_text",
            "candidate_intent",
            "confidence",
            "notes",
        ]
    ]

    review_df.to_csv(
        REVIEW_PATH,
        index=False,
        encoding="utf-8",
    )

    print(
        f"Review worksheet:\n{REVIEW_PATH}"
    )

    return review_df


# ============================================================
# Validation
# ============================================================

def print_validation(
    customer_df,
    sample,
    topic_df,
):
    """
    Print final discovery statistics.
    """

    print("\n" + "=" * 70)
    print("DISCOVERY VALIDATION")
    print("=" * 70)

    print(
        f"\nUsable unique customer messages: "
        f"{len(customer_df):,}"
    )

    print(
        f"Discovery sample: "
        f"{len(sample):,}"
    )

    print(
        f"Candidate topics: "
        f"{len(topic_df):,}"
    )

    print(
        "\nSample coverage: "
        f"{len(sample) / len(customer_df):.2%}"
    )

    print(
        "\nTopic shares:"
    )

    for _, row in topic_df.iterrows():

        print(
            f"  {row['topic']}: "
            f"{row['topic_share']:.2%}"
        )


# ============================================================
# Main
# ============================================================

def main():

    # --------------------------------------------------------
    # 1. Load
    # --------------------------------------------------------

    df = load_data()

    # --------------------------------------------------------
    # 2. Prepare unique customer messages
    # --------------------------------------------------------

    customer_df = (
        prepare_customer_messages(df)
    )

    # --------------------------------------------------------
    # 3. Create discovery sample
    # --------------------------------------------------------

    sample = (
        create_discovery_sample(
            customer_df
        )
    )

    # --------------------------------------------------------
    # 4. TF-IDF
    # --------------------------------------------------------

    vectorizer, matrix = (
        build_tfidf(sample)
    )

    # --------------------------------------------------------
    # 5. Global phrase analysis
    # --------------------------------------------------------

    discover_global_phrases(
        sample,
        vectorizer,
        matrix,
    )

    # --------------------------------------------------------
    # 6. Topic discovery
    # --------------------------------------------------------

    topic_df, examples_df = (
        discover_topics(
            sample,
            vectorizer,
            matrix,
        )
    )

    # --------------------------------------------------------
    # 7. Human review worksheet
    # --------------------------------------------------------

    create_review_file(
        sample,
        examples_df,
        topic_df,
    )

    # --------------------------------------------------------
    # 8. Validation
    # --------------------------------------------------------

    print_validation(
        customer_df,
        sample,
        topic_df,
    )

    # --------------------------------------------------------
    # Done
    # --------------------------------------------------------

    print("\n" + "=" * 70)
    print("INTENT DISCOVERY COMPLETE")
    print("=" * 70)

    print("\nGenerated files:")

    print(
        f"1. {SAMPLE_PATH}"
    )

    print(
        f"2. {PHRASES_PATH}"
    )

    print(
        f"3. {TOPICS_PATH}"
    )

    print(
        f"4. {EXAMPLES_PATH}"
    )

    print(
        f"5. {REVIEW_PATH}"
    )

    print("\nNext:")
    print(
        "Review the candidate topics and examples, "
        "then define the final 8–12 human-labelled intents."
    )


if __name__ == "__main__":
    main()