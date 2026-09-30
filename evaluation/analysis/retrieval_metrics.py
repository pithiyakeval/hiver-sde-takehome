import pandas as pd


INPUT_PATH = "evaluation/results/retrieval_candidates_safe.csv"


def main():
    df = pd.read_csv(INPUT_PATH)

    print("Retrieval Similarity Summary")
    print("---------------------------")

    summary = (
        df.groupby("rank")["similarity"]
        .agg(["mean", "median", "min", "max"])
        .round(4)
    )

    print(summary)

    print("\nTop-1 similarity quantiles:")
    print(
        df[df["rank"] == 1]["similarity"]
        .quantile([0.25, 0.50, 0.75, 0.90])
        .round(4)
    )


if __name__ == "__main__":
    main()