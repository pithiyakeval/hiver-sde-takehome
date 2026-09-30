import pandas as pd

INPUT_PATH = "evaluation/results/retrieval_annotation.csv"


def main():
    df = pd.read_csv(INPUT_PATH)

    df["relevant"] = pd.to_numeric(df["relevant"])
    df["useful"] = pd.to_numeric(df["useful"])
    df["rank"] = pd.to_numeric(df["rank"])

    print("Human Retrieval Evaluation")
    print("==========================")

    print(f"Queries: {df['golden_id'].nunique()}")
    print(f"Retrieved candidates: {len(df)}")

    print("\nOverall:")
    print(f"Relevant: {df['relevant'].mean():.3f}")
    print(f"Useful:   {df['useful'].mean():.3f}")

    print("\nBy rank:")
    print(
        df.groupby("rank")[["relevant", "useful"]]
        .mean()
        .round(3)
    )

    print("\nTop-k:")
    for k in [1, 2, 3]:
        subset = df[df["rank"] <= k]

        print(
            f"Top-{k}: "
            f"relevant={subset['relevant'].mean():.3f}, "
            f"useful={subset['useful'].mean():.3f}"
        )

    query_groups = df.groupby("golden_id")

    recall_relevant = query_groups["relevant"].max().mean()
    recall_useful = query_groups["useful"].max().mean()

    print("\nQueries with at least one positive result:")
    print(f"Relevant: {recall_relevant:.3f}")
    print(f"Useful:   {recall_useful:.3f}")


if __name__ == "__main__":
    main()