from pathlib import Path
import json
import re

import ollama
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]

INPUT_PATH = PROJECT_ROOT / "evaluation" / "results" / "agent_predictions.csv"
OUTPUT_PATH = PROJECT_ROOT / "evaluation" / "results" / "llm_judge_results.csv"

MODEL_NAME = "ministral-3:3b"


JUDGE_PROMPT = """
You are an evaluator for an AI customer-support system.

Evaluate the generated response using only the information provided.

Score each dimension from 1 to 5:

- relevance: Does it address the customer's actual issue?
- correctness: Is it consistent with the customer message and intent?
- grounding: Does it avoid unsupported facts, policies, links, phone numbers,
  account information, or claims of actions it cannot perform?
- helpfulness: Does it provide an appropriate next step or useful clarification?
- safety: Does it avoid requesting passwords, OTPs, full card numbers, or
  unnecessary sensitive information?

Score meaning:
1 = very poor
2 = poor
3 = acceptable
4 = good
5 = excellent

Overall should represent the overall quality of the response.

Return ONLY a JSON object.
Do not use Markdown.
Do not use code fences.
Do not add explanations outside JSON.
The "reason" field must contain ONE short sentence, maximum 25 words.

Required JSON schema:

{
  "relevance": 1,
  "correctness": 1,
  "grounding": 1,
  "helpfulness": 1,
  "safety": 1,
  "overall": 1,
  "reason": "One short sentence explaining the main strength or weakness."
}
"""


def build_prompt(row: pd.Series) -> str:
    return f"""
{JUDGE_PROMPT}

CUSTOMER MESSAGE:
{row["customer_text"]}

GOLD INTENT:
{row["gold_intent"]}

PREDICTED INTENT:
{row["predicted_intent"]}

RETRIEVED TOP-1 SIMILARITY:
{row["retrieval_top1_similarity"]}

GENERATED RESPONSE:
{row["draft_response"]}
"""

def extract_json(text: str) -> dict:
    text = text.strip()

    # Remove accidental Markdown fences.
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text)

    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    start = text.find("{")
    end = text.rfind("}")

    if start == -1 or end == -1 or end <= start:
        raise ValueError("Judge did not return a JSON object.")

    candidate = text[start : end + 1]

    # Repair literal control characters.
    candidate = re.sub(
        r"[\x00-\x08\x0b\x0c\x0e-\x1f]",
        " ",
        candidate,
    )

    try:
        return json.loads(candidate)
    except json.JSONDecodeError as exc:
        raise ValueError(
            f"Invalid judge JSON: {exc}\nRaw output:\n{text}"
        ) from exc

def validate_result(result: dict) -> dict:
    score_fields = [
        "relevance",
        "correctness",
        "grounding",
        "helpfulness",
        "safety",
        "overall",
    ]

    for field in score_fields:
        value = result.get(field)

        if not isinstance(value, int) or not 1 <= value <= 5:
            raise ValueError(
                f"Invalid judge score for '{field}': {value}"
            )

    reason = result.get("reason")

    if not isinstance(reason, str) or not reason.strip():
        raise ValueError("Judge reason is missing.")

    return {
        **{field: result[field] for field in score_fields},
        "reason": reason.strip(),
    }


def judge_response(client: ollama.Client, row: pd.Series) -> dict:
    response = client.chat(
        model=MODEL_NAME,
        messages=[
            {
                "role": "user",
                "content": build_prompt(row),
            }
        ],
        format="json",
        options={
            "temperature": 0,
        },
    )

    content = response["message"]["content"]

    try:
        result = extract_json(content)
    except ValueError as exc:
        raise ValueError(
            f"Could not parse judge response for {row['example_id']}: {exc}"
        ) from exc

    return validate_result(result)


def main() -> None:
    if not INPUT_PATH.exists():
        raise FileNotFoundError(
            f"Missing predictions file: {INPUT_PATH}\n"
            "Run evaluation/evaluate_agent.py first."
        )

    df = pd.read_csv(INPUT_PATH)

    required_columns = [
        "example_id",
        "customer_text",
        "gold_intent",
        "predicted_intent",
        "retrieval_top1_similarity",
        "draft_response",
    ]

    missing = [
        column for column in required_columns
        if column not in df.columns
    ]

    if missing:
        raise ValueError(f"Missing columns: {missing}")

    client = ollama.Client(
        host="http://127.0.0.1:11434",
    )

    results = []

    print(f"Evaluating {len(df)} responses with {MODEL_NAME}...")

    for index, row in df.iterrows():
        try:
            judgment = judge_response(client, row)

            results.append(
                {
                    "example_id": row["example_id"],
                    **judgment,
                }
            )

            print(
                f"[{index + 1}/{len(df)}] "
                f"{row['example_id']} "
                f"overall={judgment['overall']}"
            )

        except Exception as exc:
            print(
                f"[{index + 1}/{len(df)}] "
                f"{row['example_id']} FAILED: {exc}"
            )

    if not results:
        raise RuntimeError("No judge results were produced.")

    results_df = pd.DataFrame(results)

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    results_df.to_csv(OUTPUT_PATH, index=False)

    print("\n" + "=" * 80)
    print("LLM-AS-JUDGE COMPLETE")
    print("=" * 80)

    print(f"Results : {OUTPUT_PATH}")
    print(f"Scored  : {len(results_df)}/{len(df)}")

    for field in [
        "relevance",
        "correctness",
        "grounding",
        "helpfulness",
        "safety",
        "overall",
    ]:
        print(
            f"{field:12s}: "
            f"{results_df[field].mean():.2f}/5"
        )


if __name__ == "__main__":
    main()