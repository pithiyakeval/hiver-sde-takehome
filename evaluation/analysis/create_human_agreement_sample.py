from pathlib import Path

import pandas as pd


SOURCE = Path("golden/golden_evaluation.csv")
OUTPUT = Path("golden/human_agreement_sample.csv")

SAMPLE_SIZE = 50
RANDOM_STATE = 42


def main() -> None:
    df = pd.read_csv(SOURCE)

    required_columns = {"ID", "customer_text"}
    missing = required_columns - set(df.columns)

    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    if len(df) < SAMPLE_SIZE:
        raise ValueError(
            f"Need at least {SAMPLE_SIZE} golden examples, found {len(df)}"
        )

    # Stratify using the existing gold labels internally so that
    # less-common intents have a chance to appear in the agreement sample.
    # The labels themselves are never written to the annotator worksheet.
    intent_counts = df["Intent"].value_counts()

    base_per_intent = 2
    selected_parts = []

    for intent in intent_counts.index:
        intent_df = df[df["Intent"] == intent]

        n = min(base_per_intent, len(intent_df))

        selected_parts.append(
            intent_df.sample(
                n=n,
                random_state=RANDOM_STATE + hash(intent) % 10000,
            )
        )

    selected = pd.concat(selected_parts, ignore_index=True)

    remaining_needed = SAMPLE_SIZE - len(selected)

    if remaining_needed > 0:
        remaining = df[~df["ID"].isin(selected["ID"])]

        extra = remaining.sample(
            n=remaining_needed,
            random_state=RANDOM_STATE,
        )

        selected = pd.concat([selected, extra], ignore_index=True)

    selected = selected.sample(
        frac=1,
        random_state=RANDOM_STATE,
    ).reset_index(drop=True)

    worksheet = pd.DataFrame(
        {
            "ID": selected["ID"],
            "customer_text": selected["customer_text"],
            "second_annotator_intent": "",
            "second_annotator_confidence": "",
            "second_annotator_ambiguous": "",
            "second_annotator_notes": "",
        }
    )

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    worksheet.to_csv(OUTPUT, index=False)

    print(f"Created: {OUTPUT}")
    print(f"Examples: {len(worksheet)}")
    print("Original gold labels were not included in the worksheet.")


if __name__ == "__main__":
    main()