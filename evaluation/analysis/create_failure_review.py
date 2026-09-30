import pandas as pd


INPUT_PATH = "evaluation/results/tfidf_predictions.csv"
OUTPUT_PATH = "evaluation/results/failure_review_template.csv"


def main():
    df = pd.read_csv(INPUT_PATH)

    errors = df[df["correct"] == False].copy()

    review = errors[
        [
            "customer_text",
            "Intent",
            "predicted_intent",
        ]
    ].copy()

    review["failure_mode"] = ""
    review["hypothesis"] = ""

    review.to_csv(OUTPUT_PATH, index=False)

    print(f"Review rows: {len(review)}")
    print(f"Saved: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()