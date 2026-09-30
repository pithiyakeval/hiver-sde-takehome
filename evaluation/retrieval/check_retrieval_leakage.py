import pandas as pd


GOLDEN_PATH = "golden/golden_evaluation.csv"
RETRIEVAL_PATH = "data/processed/retrieval_pairs.csv"


def main():
    golden = pd.read_csv(GOLDEN_PATH)
    retrieval = pd.read_csv(RETRIEVAL_PATH)

    golden_text = (
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

    retrieval_text_set = set(retrieval_text)

    overlap = golden_text.isin(retrieval_text_set)

    print("Retrieval Leakage Check")
    print("-----------------------")
    print(f"Golden examples: {len(golden)}")
    print(f"Golden examples found in retrieval corpus: {overlap.sum()}")
    print(
        f"Overlap rate: {overlap.mean():.2%}"
    )


if __name__ == "__main__":
    main()