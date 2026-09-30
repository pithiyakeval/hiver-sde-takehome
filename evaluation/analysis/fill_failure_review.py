import pandas as pd


INPUT_PATH = "evaluation/results/failure_review_template.csv"
OUTPUT_PATH = "evaluation/results/failure_review_derived.csv"


FAILURE_MODES = {
    1: "rare_specific_intent",
    2: "noisy_multilingual_indirect_language",
    3: "noisy_multilingual_indirect_language",
    4: "underlying_issue_vs_requested_resolution",
    5: "followup_language_masking_intent",
    6: "noisy_multilingual_indirect_language",
    7: "rare_specific_intent",
    8: "noisy_multilingual_indirect_language",
    9: "rare_specific_intent",
    10: "rare_specific_intent",
    11: "delivery_state_boundary_confusion",
    12: "rare_specific_intent",
    13: "rare_specific_intent",
    14: "noisy_multilingual_indirect_language",
    15: "delivery_state_boundary_confusion",
    16: "rare_specific_intent",
    17: "followup_language_masking_intent",
    18: "noisy_multilingual_indirect_language",
    19: "delivery_state_boundary_confusion",
    20: "underlying_issue_vs_requested_resolution",
    21: "delivery_state_boundary_confusion",
    22: "underlying_issue_vs_requested_resolution",
    23: "underlying_issue_vs_requested_resolution",
    24: "underlying_issue_vs_requested_resolution",
    25: "followup_language_masking_intent",
    26: "underlying_issue_vs_requested_resolution",
    27: "delivery_state_boundary_confusion",
    28: "underlying_issue_vs_requested_resolution",
}


HYPOTHESES = {
    "rare_specific_intent":
        "Low-frequency intents provide too few examples for reliable lexical decision boundaries.",

    "noisy_multilingual_indirect_language":
        "Sparse lexical features are less robust to multilingual, abbreviated, indirect, and social-media-style messages.",

    "underlying_issue_vs_requested_resolution":
        "The classifier does not explicitly distinguish the underlying customer problem from the requested remedy.",

    "followup_language_masking_intent":
        "Follow-up phrases such as no response, please help, and repeated contact overlap with many underlying support intents.",

    "delivery_state_boundary_confusion":
        "Delivery intents share vocabulary, while their distinction depends on the relationship between tracking status and the customer's experience.",
}


def main():
    df = pd.read_csv(INPUT_PATH)

    if len(df) != len(FAILURE_MODES):
        raise ValueError(
            f"Expected {len(FAILURE_MODES)} review rows, found {len(df)}."
        )

    df["failure_mode"] = [
        FAILURE_MODES[i]
        for i in range(1, len(df) + 1)
    ]

    df["hypothesis"] = df["failure_mode"].map(HYPOTHESES)

    df.to_csv(OUTPUT_PATH, index=False)

    print(f"Derived review rows: {len(df)}")
    print(f"Saved: {OUTPUT_PATH}")
    print("\nCounts:")
    print(df["failure_mode"].value_counts())


if __name__ == "__main__":
    main()