from pathlib import Path
import pandas as pd


# ============================================================
# Configuration
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATA_PATH = PROJECT_ROOT / "data" / "raw" / "twcs" / "twcs.csv"

PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"

AMAZON_CUSTOMER_SAMPLE = (
    PROCESSED_DIR / "amazonhelp_customer_sample.csv"
)

CHUNK_SIZE = 100_000
SAMPLE_SIZE = 2_000

SUPPORT_ACCOUNT = "AmazonHelp"


# ============================================================
# Helpers
# ============================================================

def print_section(title: str) -> None:
    print()
    print("=" * 70)
    print(title)
    print("=" * 70)


# ============================================================
# Main audit
# ============================================================

def run_audit() -> None:

    print_section("DATASET AUDIT")

    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Dataset not found:\n{DATA_PATH}\n\n"
            "Expected location:\n"
            "data/raw/twcs/twcs.csv"
        )

    print(f"Dataset: {DATA_PATH}")
    print(f"Chunk size: {CHUNK_SIZE:,}")

    total_rows = 0
    inbound_rows = 0
    outbound_rows = 0

    amazon_support_rows = 0
    amazon_response_parent_ids = set()

    customer_messages = []

    required_columns = {
        "tweet_id",
        "author_id",
        "inbound",
        "created_at",
        "text",
        "response_tweet_id",
        "in_response_to_tweet_id",
    }

    # --------------------------------------------------------
    # First pass
    # --------------------------------------------------------

    print_section("PASS 1 — DATASET COUNTS")

    for chunk_number, chunk in enumerate(
        pd.read_csv(
            DATA_PATH,
            chunksize=CHUNK_SIZE,
            low_memory=False,
        ),
        start=1,
    ):

        if chunk_number == 1:

            missing = required_columns - set(chunk.columns)

            if missing:
                raise ValueError(
                    f"Missing expected columns: {sorted(missing)}"
                )

        total_rows += len(chunk)

        inbound_mask = chunk["inbound"].astype(bool)

        inbound_rows += inbound_mask.sum()
        outbound_rows += (~inbound_mask).sum()

        # AmazonHelp support tweets
        amazon_mask = (
            (~inbound_mask)
            & (chunk["author_id"] == SUPPORT_ACCOUNT)
        )

        amazon_rows = chunk.loc[amazon_mask]

        amazon_support_rows += len(amazon_rows)

        # Customer tweet IDs directly answered by AmazonHelp
        parent_ids = amazon_rows["in_response_to_tweet_id"].dropna()

        amazon_response_parent_ids.update(
            parent_ids.astype("int64").tolist()
        )

        # Progress
        if chunk_number % 5 == 0:
            print(
                f"Processed {total_rows:,} tweets..."
            )

    # --------------------------------------------------------
    # Second pass — find customers answered by AmazonHelp
    # --------------------------------------------------------

    print_section("PASS 2 — AMAZONHELP CUSTOMER INTERACTIONS")

    direct_customer_count = 0

    for chunk_number, chunk in enumerate(
        pd.read_csv(
            DATA_PATH,
            chunksize=CHUNK_SIZE,
            low_memory=False,
            usecols=[
                "tweet_id",
                "author_id",
                "inbound",
                "created_at",
                "text",
                "response_tweet_id",
                "in_response_to_tweet_id",
            ],
        ),
        start=1,
    ):

        inbound_mask = chunk["inbound"].astype(bool)

        matched = chunk[
            inbound_mask
            & chunk["tweet_id"].isin(amazon_response_parent_ids)
        ].copy()

        if not matched.empty:

            direct_customer_count += len(matched)

            if len(customer_messages) < SAMPLE_SIZE:

                remaining = SAMPLE_SIZE - len(customer_messages)

                customer_messages.extend(
                    matched.head(remaining).to_dict("records")
                )

        if chunk_number % 5 == 0:
            print(
                f"Processed {chunk_number * CHUNK_SIZE:,} tweets..."
            )

    # --------------------------------------------------------
    # Results
    # --------------------------------------------------------

    print_section("RESULTS")

    print(f"Total tweets:                 {total_rows:,}")
    print(f"Customer tweets:              {inbound_rows:,}")
    print(f"Support tweets:               {outbound_rows:,}")
    print(
        f"AmazonHelp support tweets:    "
        f"{amazon_support_rows:,}"
    )
    print(
        f"Unique customer tweets "
        f"answered by AmazonHelp:      "
        f"{direct_customer_count:,}"
    )

    print()
    print(
        "AmazonHelp response links:    "
        f"{len(amazon_response_parent_ids):,}"
    )

    # --------------------------------------------------------
    # Percentages
    # --------------------------------------------------------

    print_section("DATASET DISTRIBUTION")

    if total_rows:
        print(
            f"Customer share:               "
            f"{inbound_rows / total_rows:.2%}"
        )

        print(
            f"Support share:                "
            f"{outbound_rows / total_rows:.2%}"
        )

    if outbound_rows:
        print(
            f"AmazonHelp share of support:  "
            f"{amazon_support_rows / outbound_rows:.2%}"
        )

    # --------------------------------------------------------
    # Save sample
    # --------------------------------------------------------

    PROCESSED_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    if customer_messages:

        sample_df = pd.DataFrame(customer_messages)

        sample_df.to_csv(
            AMAZON_CUSTOMER_SAMPLE,
            index=False,
        )

        print_section("SAVED EXPLORATION SAMPLE")

        print(
            f"File: {AMAZON_CUSTOMER_SAMPLE}"
        )

        print(
            f"Rows: {len(sample_df):,}"
        )

    # --------------------------------------------------------
    # Message statistics
    # --------------------------------------------------------

    if customer_messages:

        sample_df = pd.DataFrame(customer_messages)

        sample_df["text_length"] = (
            sample_df["text"]
            .fillna("")
            .astype(str)
            .str.len()
        )

        print_section(
            "AMAZONHELP CUSTOMER MESSAGE LENGTH"
        )

        print(
            sample_df["text_length"].describe()
        )

    print_section("AUDIT COMPLETE")


if __name__ == "__main__":
    run_audit()