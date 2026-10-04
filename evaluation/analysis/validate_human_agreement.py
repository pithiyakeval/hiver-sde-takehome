from pathlib import Path

import pandas as pd


INPUT_PATH = Path("golden/human_agreement_sample.csv")

VALID_INTENTS = {
    "order_status",
    "delivery_late",
    "delivered_not_received",
    "delivery_promise",
    "missing_or_wrong_item",
    "damaged_item_or_package",
    "return_or_refund",
    "payment_or_charge",
    "prime_membership",
    "account_or_access",
    "product_or_device_help",
    "customer_service_followup",
    "other_or_unclear",
}

VALID_CONFIDENCE = {"high", "medium", "low"}
VALID_AMBIGUOUS = {"yes", "no"}


def main() -> None:
    df = pd.read_csv(INPUT_PATH)

    required_columns = {
        "ID",
        "customer_text",
        "second_annotator_intent",
        "second_annotator_confidence",
        "second_annotator_ambiguous",
        "second_annotator_notes",
    }

    missing_columns = required_columns - set(df.columns)

    if missing_columns:
        raise ValueError(
            f"Missing columns: {sorted(missing_columns)}"
        )

    print(f"Rows: {len(df)}")

    if len(df) != 50:
        raise ValueError(f"Expected exactly 50 rows, found {len(df)}")

    duplicate_ids = df[df["ID"].duplicated()]["ID"].tolist()

    if duplicate_ids:
        raise ValueError(
            f"Duplicate IDs found: {duplicate_ids}"
        )

    print("✓ No duplicate IDs")

    missing_intents = df[
        df["second_annotator_intent"].isna()
        | (df["second_annotator_intent"].astype(str).str.strip() == "")
    ]

    if not missing_intents.empty:
        raise ValueError(
            f"Missing intent labels for: {missing_intents['ID'].tolist()}"
        )

    invalid_intents = sorted(
        set(df["second_annotator_intent"].astype(str).str.strip())
        - VALID_INTENTS
    )

    if invalid_intents:
        print("\nInvalid intents:")
        for value in invalid_intents:
            print(f"  - {value}")
    else:
        print("✓ All intent labels are valid")

    invalid_confidence = sorted(
        set(df["second_annotator_confidence"].astype(str).str.strip())
        - VALID_CONFIDENCE
    )

    if invalid_confidence:
        print("\nInvalid confidence values:")
        for value in invalid_confidence:
            print(f"  - {value}")
    else:
        print("✓ All confidence values are valid")

    invalid_ambiguous = sorted(
        set(df["second_annotator_ambiguous"].astype(str).str.strip())
        - VALID_AMBIGUOUS
    )

    if invalid_ambiguous:
        print("\nInvalid ambiguity values:")
        for value in invalid_ambiguous:
            print(f"  - {value}")
    else:
        print("✓ All ambiguity values are valid")

    print("\nValidation complete.")


if __name__ == "__main__":
    main()