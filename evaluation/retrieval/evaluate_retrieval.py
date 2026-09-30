import pandas as pd

from src.retrieval.tfidf_retriever import TfidfRetriever


GOLDEN_PATH = "golden/golden_evaluation.csv"
RETRIEVAL_PATH = "data/processed/retrieval_pairs_safe.csv"
OUTPUT_PATH = "evaluation/retrieval_candidates_safe.csv"


def main():
    golden = pd.read_csv(GOLDEN_PATH)

    retriever = TfidfRetriever(
        data_path=RETRIEVAL_PATH
    )

    rows = []

    for _, example in golden.iterrows():
        query = str(example["customer_text"])

        results = retriever.search(
            query,
            top_k=5,
        )

        for rank, (_, result) in enumerate(
            results.iterrows(),
            start=1,
        ):
            rows.append({
                "golden_id": example["ID"],
                "query": query,
                "query_intent": example["Intent"],
                "rank": rank,
                "conversation_id": result["conversation_id"],
                "retrieved_customer_text": result["customer_text"],
                "historical_response": result["amazonhelp_text"],
                "similarity": result["similarity"],
            })

    output = pd.DataFrame(rows)

    output.to_csv(
        OUTPUT_PATH,
        index=False,
    )

    print("Leakage-Safe Retrieval Evaluation")
    print("---------------------------------")
    print(f"Queries: {len(golden)}")
    print(f"Retrieved rows: {len(output)}")
    print(f"Saved: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()