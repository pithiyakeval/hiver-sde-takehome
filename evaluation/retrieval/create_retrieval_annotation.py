import pandas as pd

INPUT_PATH = "evaluation/results/retrieval_candidates_safe.csv"
OUTPUT_PATH = "evaluation/results/retrieval_annotation.csv"


def main():
    df = pd.read_csv(INPUT_PATH)

    # Use the first 50 golden queries.
    selected_ids = sorted(df["golden_id"].unique())[:50]

    sample = df[
        df["golden_id"].isin(selected_ids) &
        df["rank"].isin([1, 2, 3])
    ].copy()

    sample["relevant"] = ""
    sample["useful"] = ""
    sample["notes"] = ""

    sample = sample[
        [
            "golden_id",
            "query_intent",
            "query",
            "rank",
            "retrieved_customer_text",
            "historical_response",
            "similarity",
            "relevant",
            "useful",
            "notes",
        ]
    ]

    sample.to_csv(OUTPUT_PATH, index=False)

    print(f"Created: {OUTPUT_PATH}")
    print(f"Rows: {len(sample)}")
    print(f"Queries: {sample['golden_id'].nunique()}")


if __name__ == "__main__":
    main()