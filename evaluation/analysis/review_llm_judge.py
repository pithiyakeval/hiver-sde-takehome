from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]

RESULTS_PATH = (
    PROJECT_ROOT
    / "evaluation"
    / "results"
    / "llm_judge_results.csv"
)


def main() -> None:
    if not RESULTS_PATH.exists():
        raise FileNotFoundError(
            f"Missing judge results: {RESULTS_PATH}"
        )

    df = pd.read_csv(RESULTS_PATH)

    required = [
        "example_id",
        "relevance",
        "correctness",
        "grounding",
        "helpfulness",
        "safety",
        "overall",
        "reason",
    ]

    missing = [column for column in required if column not in df.columns]

    if missing:
        raise ValueError(f"Missing columns: {missing}")

    worst = (
        df.sort_values(
            ["overall", "correctness", "helpfulness"],
            ascending=[True, True, True],
        )
        .head(5)
    )

    print("=" * 90)
    print("WORST 5 LLM-JUDGE RESPONSES")
    print("=" * 90)

    for _, row in worst.iterrows():
        print("\n" + "-" * 90)
        print(f"Example     : {row['example_id']}")
        print(f"Relevance   : {row['relevance']}/5")
        print(f"Correctness : {row['correctness']}/5")
        print(f"Grounding   : {row['grounding']}/5")
        print(f"Helpfulness : {row['helpfulness']}/5")
        print(f"Safety      : {row['safety']}/5")
        print(f"Overall     : {row['overall']}/5")
        print(f"Reason      : {row['reason']}")

    print("\n" + "=" * 90)


if __name__ == "__main__":
    main()