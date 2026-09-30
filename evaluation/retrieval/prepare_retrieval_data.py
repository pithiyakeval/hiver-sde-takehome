import pandas as pd


INPUT_PATH = "data/processed/amazonhelp_conversations.csv"
OUTPUT_PATH = "data/processed/retrieval_pairs.csv"


def main():
    df = pd.read_csv(INPUT_PATH)

    retrieval_df = df[
        [
            "conversation_id",
            "customer_text",
            "amazonhelp_text",
        ]
    ].copy()

    retrieval_df = retrieval_df.drop_duplicates(
        subset=["customer_text", "amazonhelp_text"]
    )

    retrieval_df["customer_text"] = (
        retrieval_df["customer_text"]
        .astype(str)
        .str.strip()
    )

    retrieval_df["amazonhelp_text"] = (
        retrieval_df["amazonhelp_text"]
        .astype(str)
        .str.strip()
    )

    retrieval_df = retrieval_df[
        (retrieval_df["customer_text"] != "")
        & (retrieval_df["amazonhelp_text"] != "")
    ]

    retrieval_df.to_csv(
        OUTPUT_PATH,
        index=False,
    )

    print("=== Retrieval Dataset ===")
    print(f"Original rows: {len(df)}")
    print(f"Retrieval rows: {len(retrieval_df)}")
    print(f"Saved to: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()