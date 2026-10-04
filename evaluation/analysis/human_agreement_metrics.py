from pathlib import Path

import pandas as pd
from sklearn.metrics import accuracy_score, cohen_kappa_score


GOLD_PATH = Path("golden/golden_evaluation.csv")
SECOND_ANNOTATOR_PATH = Path("golden/human_agreement_sample.csv")

INTENT_CODE_MAP = {
    "OS": "order_status",
    "DL": "delivery_late",
    "DNR": "delivered_not_received",
    "DP": "delivery_promise",
    "MWI": "missing_or_wrong_item",
    "DIP": "damaged_item_or_package",
    "RR": "return_or_refund",
    "PC": "payment_or_charge",
    "PM": "prime_membership",
    "AA": "account_or_access",
    "PDH": "product_or_device_help",
    "CSF": "customer_service_followup",
    "OU": "other_or_unclear",
}


def main() -> None:
    gold_df = pd.read_csv(GOLD_PATH)
    second_df = pd.read_csv(SECOND_ANNOTATOR_PATH)

    gold_labels = gold_df[["ID", "Intent"]].copy()
    gold_labels["Intent"] = (
        gold_labels["Intent"]
        .astype(str)
        .str.strip()
        .map(INTENT_CODE_MAP)
    )

    second_labels = second_df[
        [
            "ID",
            "second_annotator_intent",
            "second_annotator_confidence",
            "second_annotator_ambiguous",
        ]
    ].copy()

    second_labels["second_annotator_intent"] = (
        second_labels["second_annotator_intent"]
        .astype(str)
        .str.strip()
    )

    df = gold_labels.merge(
        second_labels,
        on="ID",
        how="inner",
        validate="one_to_one",
    )

    if len(df) != 50:
        raise ValueError(
            f"Expected 50 matched examples, found {len(df)}."
        )

    if df["Intent"].isna().any():
        raise ValueError("Unknown gold intent code found.")

    gold = df["Intent"]
    second = df["second_annotator_intent"]

    agreement = accuracy_score(gold, second)
    kappa = cohen_kappa_score(gold, second)

    disagreements = df[gold != second].copy()

    print("=" * 60)
    print("HUMAN-HUMAN AGREEMENT")
    print("=" * 60)

    print(f"Examples: {len(df)}")
    print(
        f"Exact agreement: {agreement:.4f} "
        f"({agreement * 100:.2f}%)"
    )
    print(f"Cohen's kappa: {kappa:.4f}")
    print(f"Disagreements: {len(disagreements)}")

    print("\nAgreement counts:")
    print(f"Agreements: {len(df) - len(disagreements)}")
    print(f"Disagreements: {len(disagreements)}")

    print("\nDisagreement pairs:")

    if disagreements.empty:
        print("None")
    else:
        print(
            pd.crosstab(
                disagreements["Intent"],
                disagreements["second_annotator_intent"],
                rownames=["Gold"],
                colnames=["Second annotator"],
            )
        )

    print("\nIndividual disagreements:")

    if disagreements.empty:
        print("None")
    else:
        print(
            disagreements[
                [
                    "ID",
                    "Intent",
                    "second_annotator_intent",
                    "second_annotator_confidence",
                    "second_annotator_ambiguous",
                ]
            ].to_string(index=False)
        )


if __name__ == "__main__":
    main()