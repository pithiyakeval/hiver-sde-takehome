from pathlib import Path

import pandas as pd
from sklearn.metrics import accuracy_score, f1_score


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[2]

TEST_PATH = PROJECT_ROOT / "golden" / "evaluation_test.csv"
PREDICTIONS_PATH = (
    PROJECT_ROOT
    / "evaluation"
    / "results"
    / "agent_predictions.csv"
)


# ---------------------------------------------------------------------------
# Canonical intent mapping
# ---------------------------------------------------------------------------
#
# The frozen golden set uses compact intent codes.
# The LLM classifier uses descriptive intent names.
#
# Evaluation must normalize both representations before calculating metrics.
# ---------------------------------------------------------------------------

INTENT_CODE_MAP = {
    "order_status": "OS",
    "delivery_late": "DL",
    "delivered_not_received": "DNR",
    "delivery_promise": "DP",
    "missing_or_wrong_item": "MWI",
    "damaged_item_or_package": "DIP",
    "return_or_refund": "RR",
    "payment_or_charge": "PC",
    "prime_membership": "PM",
    "account_or_access": "AA",
    "product_or_device_help": "PDH",
    "customer_service_followup": "CSF",
    "other_or_unclear": "OU",
}


# ---------------------------------------------------------------------------
# Validation helpers
# ---------------------------------------------------------------------------

def parse_bool(value):
    """Normalize common human-annotation boolean representations."""

    if pd.isna(value):
        return None

    normalized = str(value).strip().lower()

    if normalized in {"true", "yes", "y", "1"}:
        return True

    if normalized in {"false", "no", "n", "0"}:
        return False

    raise ValueError(
        f"Unsupported boolean value: {value!r}"
    )


def normalize_intent(value):
    """
    Normalize an intent prediction into the canonical evaluation code.

    The LLM uses descriptive names such as ``delivery_late`` while the
    golden set uses compact codes such as ``DL``.
    """

    if pd.isna(value):
        raise ValueError("Encountered missing intent prediction.")

    normalized = str(value).strip()

    if not normalized:
        raise ValueError("Encountered empty intent prediction.")

    # LLM descriptive intent → canonical code.
    if normalized in INTENT_CODE_MAP:
        return INTENT_CODE_MAP[normalized]

    # Already canonical.
    canonical_codes = set(INTENT_CODE_MAP.values())

    if normalized in canonical_codes:
        return normalized

    raise ValueError(
        f"Unknown intent prediction: {normalized!r}. "
        f"Expected one of: "
        f"{sorted(INTENT_CODE_MAP)} "
        f"or canonical codes {sorted(canonical_codes)}."
    )


# ---------------------------------------------------------------------------
# Data loading and validation
# ---------------------------------------------------------------------------

def load_evaluation_data():
    """Load and validate the frozen gold set and agent predictions."""

    if not TEST_PATH.exists():
        raise FileNotFoundError(
            f"Evaluation dataset not found: {TEST_PATH}"
        )

    if not PREDICTIONS_PATH.exists():
        raise FileNotFoundError(
            f"Agent predictions not found: {PREDICTIONS_PATH}"
        )

    gold = pd.read_csv(TEST_PATH)
    predictions = pd.read_csv(PREDICTIONS_PATH)

    required_gold_columns = {
        "ID",
        "Intent",
        "Escalate",
    }

    required_prediction_columns = {
        "example_id",
        "predicted_intent",
        "intent_confidence",
        "should_escalate",
        "retrieval_top1_similarity",
    }

    missing_gold = required_gold_columns - set(gold.columns)

    if missing_gold:
        raise ValueError(
            "Gold evaluation dataset is missing required columns: "
            f"{sorted(missing_gold)}"
        )

    missing_predictions = (
        required_prediction_columns - set(predictions.columns)
    )

    if missing_predictions:
        raise ValueError(
            "Agent predictions are missing required columns: "
            f"{sorted(missing_predictions)}"
        )

    # Keep only the fields required for this evaluation.
    gold = gold[
        [
            "ID",
            "Intent",
            "Escalate",
        ]
    ].copy()

    predictions = predictions[
        [
            "example_id",
            "predicted_intent",
            "intent_confidence",
            "should_escalate",
            "retrieval_top1_similarity",
        ]
    ].copy()

    # Check for duplicate IDs before joining.
    if gold["ID"].duplicated().any():
        duplicates = gold.loc[
            gold["ID"].duplicated(),
            "ID",
        ].tolist()

        raise ValueError(
            f"Duplicate IDs found in gold evaluation set: {duplicates}"
        )

    if predictions["example_id"].duplicated().any():
        duplicates = predictions.loc[
            predictions["example_id"].duplicated(),
            "example_id",
        ].tolist()

        raise ValueError(
            f"Duplicate example IDs found in agent predictions: {duplicates}"
        )

    merged = gold.merge(
        predictions,
        left_on="ID",
        right_on="example_id",
        how="inner",
    )

    if len(merged) != len(gold):
        raise ValueError(
            "Evaluation join lost rows: "
            f"gold={len(gold)}, "
            f"predictions={len(predictions)}, "
            f"merged={len(merged)}"
        )

    if len(predictions) != len(gold):
        raise ValueError(
            "Prediction count does not match the frozen test set: "
            f"gold={len(gold)}, "
            f"predictions={len(predictions)}"
        )

    return merged


# ---------------------------------------------------------------------------
# Metric calculation
# ---------------------------------------------------------------------------

def calculate_metrics(merged):
    """Calculate intent, escalation, confidence, and retrieval metrics."""

    merged = merged.copy()

    # ---------------------------------------------------------------
    # Intent normalization
    # ---------------------------------------------------------------

    merged["gold_intent"] = (
        merged["Intent"]
        .astype(str)
        .str.strip()
    )

    merged["predicted_intent_code"] = (
        merged["predicted_intent"]
        .apply(normalize_intent)
    )

    # ---------------------------------------------------------------
    # Escalation normalization
    # ---------------------------------------------------------------

    merged["gold_escalate"] = (
        merged["Escalate"]
        .apply(parse_bool)
    )

    if merged["gold_escalate"].isna().any():
        raise ValueError(
            "Gold escalation labels contain missing values."
        )

    merged["predicted_escalate"] = (
        merged["should_escalate"]
        .apply(parse_bool)
    )

    if merged["predicted_escalate"].isna().any():
        raise ValueError(
            "Predicted escalation labels contain missing values."
        )

    # ---------------------------------------------------------------
    # Intent metrics
    # ---------------------------------------------------------------

    intent_accuracy = accuracy_score(
        merged["gold_intent"],
        merged["predicted_intent_code"],
    )

    intent_macro_f1 = f1_score(
        merged["gold_intent"],
        merged["predicted_intent_code"],
        average="macro",
        zero_division=0,
    )

    # ---------------------------------------------------------------
    # Escalation metrics
    # ---------------------------------------------------------------

    escalation_accuracy = accuracy_score(
        merged["gold_escalate"],
        merged["predicted_escalate"],
    )

    # ---------------------------------------------------------------
    # Supporting metrics
    # ---------------------------------------------------------------

    mean_confidence = (
        pd.to_numeric(
            merged["intent_confidence"],
            errors="coerce",
        )
        .mean()
    )

    mean_top1_similarity = (
        pd.to_numeric(
            merged["retrieval_top1_similarity"],
            errors="coerce",
        )
        .mean()
    )

    return {
        "intent_accuracy": intent_accuracy,
        "intent_macro_f1": intent_macro_f1,
        "escalation_accuracy": escalation_accuracy,
        "mean_confidence": mean_confidence,
        "mean_top1_similarity": mean_top1_similarity,
    }, merged


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def print_results(metrics, merged):
    """Print the evaluation summary."""

    print()
    print("=" * 72)
    print("LLM-Powered Support Agent Evaluation")
    print("=" * 72)

    print(f"Examples: {len(merged)}")

    print()
    print("Intent")
    print("-" * 72)
    print(
        f"Accuracy:          "
        f"{metrics['intent_accuracy']:.4f}"
    )
    print(
        f"Macro F1:          "
        f"{metrics['intent_macro_f1']:.4f}"
    )

    print()
    print("Escalation")
    print("-" * 72)
    print(
        f"Accuracy:          "
        f"{metrics['escalation_accuracy']:.4f}"
    )

    print()
    print("Supporting Signals")
    print("-" * 72)
    print(
        f"Mean LLM confidence: "
        f"{metrics['mean_confidence']:.4f}"
    )
    print(
        f"Mean top-1 retrieval similarity: "
        f"{metrics['mean_top1_similarity']:.4f}"
    )

    print()
    print("Intent Distribution")
    print("-" * 72)

    distribution = (
        merged[
            [
                "gold_intent",
                "predicted_intent_code",
            ]
        ]
        .value_counts()
        .sort_index()
    )

    for (gold_intent, predicted_intent), count in distribution.items():
        print(
            f"Gold={gold_intent:<4} "
            f"Predicted={predicted_intent:<4} "
            f"Count={count}"
        )

    print("=" * 72)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    """Run the complete end-to-end agent evaluation."""

    merged = load_evaluation_data()

    metrics, evaluated = calculate_metrics(merged)

    print_results(
        metrics,
        evaluated,
    )


if __name__ == "__main__":
    main()