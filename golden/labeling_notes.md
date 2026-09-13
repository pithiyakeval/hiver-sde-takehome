# Golden Set Sampling Notes

## Dataset

The golden-set candidates were sampled from `data/processed/amazonhelp_conversations.csv`.

The source contains AmazonHelp customer-support conversation pairs.

The sampling process first deduplicates customer messages and then selects examples using deterministic sampling rules.

## Important distinction

This file contains **candidate examples for human annotation**.

The sampling buckets are **not intent labels**.

Final intent labels must be assigned by a human using `golden/intent_taxonomy.md`.

The model must not be allowed to determine the golden labels.

## Reproducibility

Random seed:

`42`

Target golden-set size:

`200`

Sampling script:

`src/intents/build_golden_sample.py`

Run:

`python -m src.intents.build_golden_sample`

## Sampling strategy

The candidate set deliberately contains:

1. Common, clearly expressed customer problems
2. Boundary cases between similar intents
3. Short/noisy Twitter messages
4. Multilingual examples
5. Multi-intent messages
6. Likely escalation cases
7. Ambiguous/unclear examples

This is preferable to a purely random sample because a golden evaluation set should test both ordinary behavior and failure-prone cases.

## Sampling distribution

| Bucket | Target | Available candidates | Selected | Target met |
|---|---:|---:|---:|---|
| `common_normal` | 100 | 350 | 100 | Yes |
| `boundary_case` | 40 | 350 | 40 | Yes |
| `short_noisy` | 20 | 350 | 20 | Yes |
| `multilingual` | 15 | 350 | 15 | Yes |
| `multi_intent` | 10 | 350 | 10 | Yes |
| `likely_escalation` | 10 | 350 | 10 | Yes |
| `other_unclear_candidate` | 5 | 350 | 5 | Yes |
| `TOTAL` | 200 | 2450 | 200 | Yes |

## Human annotation

Annotators should use the taxonomy and boundary rules in:

`golden/intent_taxonomy.md`

The following fields must be completed manually:

- `intent`
- `confidence`
- `is_ambiguous`
- `ambiguity_reason`
- `expected_action`
- `should_escalate`
- `escalation_reason`
- `annotator`

The original `customer_text` must not be rewritten.

## Golden-set integrity

After annotation, the golden set should be frozen and kept separate from model training data.

No golden example should be used to train or tune the final classifier after it becomes part of the held-out evaluation set.

If taxonomy definitions change, affected examples should be re-reviewed and the taxonomy version recorded in the evaluation results.
