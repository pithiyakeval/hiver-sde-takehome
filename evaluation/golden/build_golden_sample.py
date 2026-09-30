"""
Build a reproducible, annotation-ready golden-set candidate sample
for the AmazonHelp customer-support agent.

The script intentionally does NOT assign final intent labels.

It samples real AmazonHelp customer messages into several useful
evaluation buckets so that the final human-labelled golden set is
not dominated by easy/common examples.

Input:
    data/processed/amazonhelp_conversations.csv

Outputs:
    golden/golden_candidates.csv
    golden/golden_sampling_report.csv
    golden/labeling_notes.md

Design goals:
    - deterministic sampling
    - no duplicate customer tweets
    - preserve original customer text
    - include difficult/boundary cases
    - include multilingual examples
    - include short/noisy Twitter messages
    - include likely multi-intent messages
    - provide transparent sampling metadata
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from evaluation.golden.config import (
    INPUT_PATH,
    OUTPUT_DIR,
    SEED,
    SAMPLE_SIZE,
    SAMPLING_TARGETS,
)

from evaluation.golden.deduplication import deduplicate_messages

from evaluation.golden.sampling import (
    build_bucket_candidates,
    select_final_candidates,
)

from evaluation.golden.annotations import add_annotation_columns

from evaluation.golden.validation import validate_final_set

from evaluation.golden.report import (
    build_sampling_report,
    write_labeling_notes,
)

def main():

    np.random.seed(SEED)

    print("=" * 70)
    print("AMAZONHELP GOLDEN-SET CANDIDATE BUILDER")
    print("=" * 70)

    print(f"\nInput: {INPUT_PATH}")
    print(f"Output directory: {OUTPUT_DIR}")
    print(f"Random seed: {SEED}")
    print(f"Target size: {SAMPLE_SIZE}")

    if not INPUT_PATH.exists():
        raise FileNotFoundError(
            f"Input file not found: {INPUT_PATH}"
        )

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    # --------------------------------------------------------
    # Load data
    # --------------------------------------------------------
                                       

    print("\nLoading conversation data...")

    df = pd.read_csv(
        INPUT_PATH,
        low_memory=False,
    )

    print(
        f"Loaded {len(df):,} conversation rows."
    )

    # --------------------------------------------------------
    # Deduplicate
    # --------------------------------------------------------

    print("\nDeduplicating customer messages...")

    df =  deduplicate_messages(df)

    print(
        f"Usable unique customer messages: "
        f"{len(df):,}"
    )

    # --------------------------------------------------------
    # Build candidate buckets
    # --------------------------------------------------------

    candidates = build_bucket_candidates(df)

    # --------------------------------------------------------
    # Candidate availability
    # --------------------------------------------------------

    print("\nCandidate availability:")

    for bucket in SAMPLING_TARGETS:

        count = len(
            candidates.get(
                bucket,
                pd.DataFrame(),
            )
        )

        target = SAMPLING_TARGETS[bucket]

        print(
            f"{bucket:25s}: "
            f"{count:4d} available "
            f"(target {target})"
        )

    # --------------------------------------------------------
    # Select final candidates
    # --------------------------------------------------------

    print("\nSelecting final candidate set...")

    final_df = select_final_candidates(
        candidates
    )

    # --------------------------------------------------------
    # Add human annotation columns
    # --------------------------------------------------------

    final_df = add_annotation_columns(
        final_df
    )

    # --------------------------------------------------------
    # Keep clean output columns
    # --------------------------------------------------------

    preferred_columns = [
        "example_id",
        "customer_tweet_id",
        "customer_text",

        # Human annotation fields.
        "intent",
        "confidence",
        "is_ambiguous",
        "ambiguity_reason",
        "language",
        "expected_action",
        "should_escalate",
        "escalation_reason",
        "annotator",

        # Sampling audit fields.
        "sampling_bucket",
        "sampling_score",
    ]

    final_df = final_df[
        [
            col
            for col in preferred_columns
            if col in final_df.columns
        ]
    ]

    # --------------------------------------------------------
    # Validate
    # --------------------------------------------------------

    validate_final_set(
        final_df
    )

    # --------------------------------------------------------
    # Save candidate CSV
    # --------------------------------------------------------

    candidate_path = (
        OUTPUT_DIR
        / "golden_candidates.csv"
    )

    final_df.to_csv(
        candidate_path,
        index=False,
        encoding="utf-8-sig",
    )

    print(
        f"\nSaved candidate set:\n"
        f"{candidate_path.resolve()}"
    )

    # --------------------------------------------------------
    # Sampling report
    # --------------------------------------------------------

    report = build_sampling_report(
        candidates,
        final_df,
    )

    report_path = (
        OUTPUT_DIR
        / "golden_sampling_report.csv"
    )

    report.to_csv(
        report_path,
        index=False,
        encoding="utf-8-sig",
    )

    print(
        f"Saved sampling report:\n"
        f"{report_path.resolve()}"
    )

    # --------------------------------------------------------
    # Labeling notes
    # --------------------------------------------------------

    write_labeling_notes(
        report,
        final_df,
    )

    notes_path = (
        OUTPUT_DIR
        / "labeling_notes.md"
    )

    print(
        f"Saved labeling notes:\n"
        f"{notes_path.resolve()}"
    )

    # --------------------------------------------------------
    # Final summary
    # --------------------------------------------------------

    print("\n" + "=" * 70)
    print("GOLDEN CANDIDATE SET READY")
    print("=" * 70)
    
    print(
        f"\n{len(final_df)} examples ready "
        "for human annotation."
    )

    print(
        "\nNext step:"
        "\nOpen golden/golden_candidates.csv "
        "and label the examples using "
        "golden/intent_taxonomy.md."
    )

    print(
        "\nIMPORTANT:"
        "\nDo not train the classifier on this file."
        "\nThis will become our held-out golden evaluation set."
    )


if __name__ == "__main__":
    main()