import pandas as pd

from evaluation.golden.config import (
    OUTPUT_DIR,
    SEED,
    SAMPLE_SIZE,
    SAMPLING_TARGETS,
)


def build_sampling_report(
    candidates: dict[str, pd.DataFrame],
    final_df: pd.DataFrame,
) -> pd.DataFrame:
    rows = []

    for bucket, target in SAMPLING_TARGETS.items():
        available = len(
            candidates.get(
                bucket,
                pd.DataFrame(),
            )
        )

        selected = int(
            (
                final_df["sampling_bucket"]
                == bucket
            ).sum()
        )

        rows.append(
            {
                "sampling_bucket": bucket,
                "target_count": target,
                "available_candidates": available,
                "selected_count": selected,
                "target_met": selected >= target,
            }
        )

    # Add overall row.
    rows.append(
        {
            "sampling_bucket": "TOTAL",
            "target_count": SAMPLE_SIZE,
            "available_candidates": sum(
                len(v)
                for v in candidates.values()
                if v is not None
            ),
            "selected_count": len(final_df),
            "target_met": len(final_df) >= SAMPLE_SIZE,
        }
    )

    return pd.DataFrame(rows)


def write_labeling_notes(
    report: pd.DataFrame,
    final_df: pd.DataFrame,
):
    output_path = OUTPUT_DIR / "labeling_notes.md"

    lines = []

    lines.append("# Golden Set Sampling Notes")
    lines.append("")

    lines.append("## Dataset")
    lines.append("")

    lines.append(
        "The golden-set candidates were sampled from "
        "`data/processed/amazonhelp_conversations.csv`."
    )
    lines.append("")

    lines.append(
        "The source contains AmazonHelp customer-support conversation pairs."
    )
    lines.append("")

    lines.append(
        "The sampling process first deduplicates customer messages "
        "and then selects examples using deterministic sampling rules."
    )
    lines.append("")

    lines.append("## Important distinction")
    lines.append("")

    lines.append(
        "This file contains **candidate examples for human annotation**."
    )
    lines.append("")

    lines.append(
        "The sampling buckets are **not intent labels**."
    )
    lines.append("")

    lines.append(
        "Final intent labels must be assigned by a human using "
        "`golden/intent_taxonomy.md`."
    )
    lines.append("")

    lines.append(
        "The model must not be allowed to determine the golden labels."
    )
    lines.append("")

    lines.append("## Reproducibility")
    lines.append("")

    lines.append("Random seed:")
    lines.append("")
    lines.append(f"`{SEED}`")
    lines.append("")

    lines.append("Target golden-set size:")
    lines.append("")
    lines.append(f"`{SAMPLE_SIZE}`")
    lines.append("")

    lines.append("Sampling script:")
    lines.append("")
    lines.append("`src/intents/build_golden_sample.py`")
    lines.append("")

    lines.append("Run:")
    lines.append("")
    lines.append(
        "`python -m evaluation.intents.build_golden_sample`"
    )
    lines.append("")

    lines.append("## Sampling strategy")
    lines.append("")

    lines.append(
        "The candidate set deliberately contains:"
    )
    lines.append("")

    lines.append("1. Common, clearly expressed customer problems")
    lines.append("2. Boundary cases between similar intents")
    lines.append("3. Short/noisy Twitter messages")
    lines.append("4. Multilingual examples")
    lines.append("5. Multi-intent messages")
    lines.append("6. Likely escalation cases")
    lines.append("7. Ambiguous/unclear examples")
    lines.append("")

    lines.append(
        "This is preferable to a purely random sample because "
        "a golden evaluation set should test both ordinary behavior "
        "and failure-prone cases."
    )
    lines.append("")

    lines.append("## Sampling distribution")
    lines.append("")

    lines.append(
        "| Bucket | Target | Available candidates | Selected | Target met |"
    )
    lines.append(
        "|---|---:|---:|---:|---|"
    )

    for _, row in report.iterrows():
        bucket = row["sampling_bucket"]
        target = int(row["target_count"])
        available = int(row["available_candidates"])
        selected = int(row["selected_count"])
        target_met = "Yes" if row["target_met"] else "No"

        lines.append(
            f"| `{bucket}` | {target} | {available} | "
            f"{selected} | {target_met} |"
        )

    lines.append("")

    lines.append("## Human annotation")
    lines.append("")

    lines.append(
        "Annotators should use the taxonomy and boundary rules in:"
    )
    lines.append("")

    lines.append("`golden/intent_taxonomy.md`")
    lines.append("")

    lines.append(
        "The following fields must be completed manually:"
    )
    lines.append("")

    lines.append("- `intent`")
    lines.append("- `confidence`")
    lines.append("- `is_ambiguous`")
    lines.append("- `ambiguity_reason`")
    lines.append("- `expected_action`")
    lines.append("- `should_escalate`")
    lines.append("- `escalation_reason`")
    lines.append("- `annotator`")
    lines.append("")

    lines.append(
        "The original `customer_text` must not be rewritten."
    )
    lines.append("")

    lines.append("## Golden-set integrity")
    lines.append("")

    lines.append(
        "After annotation, the golden set should be frozen and "
        "kept separate from model training data."
    )
    lines.append("")

    lines.append(
        "No golden example should be used to train or tune the "
        "final classifier after it becomes part of the held-out "
        "evaluation set."
    )
    lines.append("")

    lines.append(
        "If taxonomy definitions change, affected examples should "
        "be re-reviewed and the taxonomy version recorded in the "
        "evaluation results."
    )
    lines.append("")

    output_path.write_text(
        "\n".join(lines),
        encoding="utf-8",
    )