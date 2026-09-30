import pandas as pd

from evaluation.golden.config import(SAMPLING_TARGETS,SAMPLE_SIZE,MAX_BUCKET_CANDIDATES)
from evaluation.golden.bucket import classify_sampling_buckets
from evaluation.golden.scoring import bucket_priority_score


def build_bucket_candidates(
    df: pd.DataFrame,
) -> dict[str, pd.DataFrame]:
    """
    Build ranked candidate pools for every sampling bucket.
    """
    bucket_rows = {
        bucket: []
        for bucket in SAMPLING_TARGETS
    }

    print("\nDetecting sampling buckets...")

    for idx, row in df.iterrows():
        buckets = classify_sampling_buckets(row)

        for bucket in buckets:
            bucket_rows[bucket].append(idx)

    candidates = {}

    for bucket, indices in bucket_rows.items():
        if not indices:
            candidates[bucket] = df.iloc[0:0].copy()
            continue

        bucket_df = df.loc[indices].copy()

        bucket_df["sampling_score"] = bucket_df.apply(
            lambda row: bucket_priority_score(
                row,
                bucket,
            ),
            axis=1,
        )

        bucket_df = (
            bucket_df
            .sort_values(
                [
                    "sampling_score",
                    "customer_tweet_id",
                ],
                ascending=[False, True],
            )
            .head(MAX_BUCKET_CANDIDATES)
            .copy()
        )

        bucket_df["sampling_bucket"] = bucket
        candidates[bucket] = bucket_df

    return candidates


def select_final_candidates(
    candidates: dict[str, pd.DataFrame],
) -> pd.DataFrame:
    """
    Select the final 200 candidate examples while avoiding duplicate
    customer tweets.

    We use the requested bucket quotas in priority order.
    Hard/difficult buckets are selected first so that common examples
    cannot consume the entire sample.
    """

    # Hard buckets first.
    selection_order = [
        "boundary_case",
        "multilingual",
        "multi_intent",
        "likely_escalation",
        "short_noisy",
        "other_unclear_candidate",
        "common_normal",
    ]

    selected_rows = []
    selected_ids = set()
    remaining_target = SAMPLE_SIZE

    # Track actual requested quotas.
    quotas = dict(SAMPLING_TARGETS)

    for bucket in selection_order:
        if remaining_target <= 0:
            break

        bucket_df = candidates.get(bucket)

        if bucket_df is None or bucket_df.empty:
            print(
                f"WARNING: no candidates available for "
                f"'{bucket}'."
            )
            continue

        target = min(
            quotas[bucket],
            remaining_target,
        )

        count = 0

        for _, row in bucket_df.iterrows():
            tweet_id = str(
                row["customer_tweet_id"]
            )

            if tweet_id in selected_ids:
                continue

            selected_ids.add(tweet_id)
            selected_rows.append(row)
            count += 1

            if count >= target:
                break

        print(
            f"{bucket:25s}: "
            f"selected {count:3d} / {target:3d}"
        )

        remaining_target -= count

    # --------------------------------------------------------
    # Fill any remaining slots using unused candidates.
    # --------------------------------------------------------

    if remaining_target > 0:
        print(
            f"\nFilling remaining {remaining_target} "
            "slots from unused candidates..."
        )

        all_candidates = []

        for bucket_df in candidates.values():
            if bucket_df is None or bucket_df.empty:
                continue

            all_candidates.append(bucket_df)

        if all_candidates:
            combined = pd.concat(
                all_candidates,
                ignore_index=True,
            )

            combined = (
                combined
                .sort_values(
                    [
                        "sampling_score",
                        "customer_tweet_id",
                    ],
                    ascending=[False, True],
                )
            )

            for _, row in combined.iterrows():
                tweet_id = str(
                    row["customer_tweet_id"]
                )

                if tweet_id in selected_ids:
                    continue

                selected_ids.add(tweet_id)

                # Preserve the bucket in which this row was
                # originally selected.
                row = row.copy()
                selected_rows.append(row)

                remaining_target -= 1

                if remaining_target <= 0:
                    break

    if not selected_rows:
        raise RuntimeError(
            "No golden candidates could be selected."
        )

    result = pd.DataFrame(selected_rows)

    # --------------------------------------------------------
    # Final deduplication.
    # --------------------------------------------------------

    result = (
        result
        .drop_duplicates(
            subset=["customer_tweet_id"],
            keep="first",
        )
    )

    # --------------------------------------------------------
    # Deterministic ordering.
    # --------------------------------------------------------

    result = (
        result
        .sort_values(
            "customer_tweet_id"
        )
        .reset_index(drop=True)
    )

    # --------------------------------------------------------
    # Assign stable example IDs.
    # --------------------------------------------------------

    result.insert(
        0,
        "example_id",
        [
            f"G{i:03d}"
            for i in range(
                1,
                len(result) + 1,
            )
        ],
    )

    return result