import pandas as pd


RETRIEVAL_PATH = "data/processed/retrieval_pairs.csv"
GOLDEN_PATH = "golden/golden_evaluation.csv"
OUTPUT_PATH = "data/processed/retrieval_pairs_safe.csv"


def main():
    retrieval = pd.read_csv(RETRIEVAL_PATH)
    golden = pd.read_csv(GOLDEN_PATH)

    golden_texts = set(
        golden["customer_text"]
        .fillna("")
        .astype(str)
        .str.strip()
    )

    retrieval_text = (
        retrieval["customer_text"]
        .fillna("")
        .astype(str)
        .str.strip()
    )

    keep_mask = ~retrieval_text.isin(golden_texts)

    safe = retrieval[keep_mask].copy()

    safe.to_csv(
        OUTPUT_PATH,
        index=False,
    )

    removed = len(retrieval) - len(safe)

    print("Leakage-Safe Retrieval Corpus")
    print("-----------------------------")
    print(f"Original rows: {len(retrieval)}")
    print(f"Removed rows: {removed}")
    print(f"Safe rows: {len(safe)}")
    print(f"Saved: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()