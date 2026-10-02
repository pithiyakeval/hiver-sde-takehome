import json
import re
from pathlib import Path

import ollama
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score


PROJECT_ROOT = Path(__file__).resolve().parents[2]

TRAIN_PATH = PROJECT_ROOT / "golden" / "evaluation_train.csv"
TEST_PATH = PROJECT_ROOT / "golden" / "evaluation_test.csv"
OUTPUT_PATH = (
    PROJECT_ROOT
    / "evaluation"
    / "results"
    / "llm_classifier_predictions.csv"
)

MODEL_NAME = "ministral-3:3b"


INTENT_DEFINITIONS = {
    "OS": "Order status: asking where an order is or what its current status is.",
    "DL": "Delivery late: the expected delivery date/time has passed and the order is delayed.",
    "DNR": "Delivered not received: tracking or status says delivered, but the customer did not receive the package.",
    "DP": "Delivery promise: a promised, guaranteed, Prime, same-day, or specific delivery commitment was not met.",
    "MWI": "Missing or wrong item: an item is missing from the order or the customer received the wrong item/product.",
    "DIP": "Damaged item or package: an item or package arrived damaged, broken, defective, cracked, or otherwise physically damaged.",
    "RR": "Return or refund: the customer primarily wants a return, refund, money back, or is asking about a return/refund.",
    "PC": "Payment or charge: payment, billing, card charges, duplicate charges, unauthorized charges, or other payment-related issues.",
    "PM": "Prime membership: questions or problems specifically about Amazon Prime membership, subscription, renewal, trial, or benefits.",
    "AA": "Account or access: login, password, account access, locked account, or account-related access problem.",
    "PDH": "Product or device help: questions about a product, device, product functionality, setup, or product information.",
    "CSF": "Customer service follow-up: following up on a previous support request, waiting for a response, or asking for contact/support assistance.",
    "OU": "Other or unclear: the message does not clearly fit another supported intent.",
}


BOUNDARIES = """
Important intent boundaries:

OS = asking where the order is or its current status.
DL = the expected delivery date/time has already passed.
DNR = tracking explicitly says delivered but the customer did not receive it.
DP = a specific delivery promise, guarantee, Prime delivery, or same-day commitment failed.
RR = return/refund/money-back is the primary request.
CSF = following up on previous support/contact is the primary purpose.
PM = Prime membership/subscription itself, not a Prime delivery failure.
PC = payment, billing, card, or financial transaction issue.
OU = genuinely unclear or unsupported.

If multiple issues appear, classify the customer's PRIMARY issue.
"""


def build_system_prompt() -> str:
    definitions = "\n".join(
        f"{code}: {description}"
        for code, description in INTENT_DEFINITIONS.items()
    )

    return f"""
You classify Amazon customer support messages.

Choose exactly ONE intent code from the allowed list.

Intent definitions:

{definitions}

{BOUNDARIES}

Return ONLY valid JSON in exactly this format:

{{"intent":"CODE"}}

Do not explain your answer.
Do not use markdown.
Do not add any other fields.
""".strip()


def parse_intent(response: str) -> str:
    """
    Parse the model response safely.

    Returns OU if the model produces an invalid response.
    """

    response = response.strip()

    # First try strict JSON.
    try:
        data = json.loads(response)

        intent = str(
            data.get("intent", "")
        ).strip().upper()

        if intent in INTENT_DEFINITIONS:
            return intent

    except json.JSONDecodeError:
        pass

    # Fallback for slightly malformed model output.
    match = re.search(
        r'"intent"\s*:\s*"([A-Z]+)"',
        response,
        re.IGNORECASE,
    )

    if match:
        intent = match.group(1).upper()

        if intent in INTENT_DEFINITIONS:
            return intent

    return "OU"


def build_examples(train: pd.DataFrame):
    """
    Select representative labelled examples from the training split.

    One example is used for most intents.
    Two examples are used for the most easily-confused delivery,
    refund, and follow-up intents.

    The frozen evaluation set is never used.
    """

    examples = []

    extra_example_intents = {
        "OS",
        "DL",
        "DNR",
        "DP",
        "RR",
        "CSF",
    }

    for intent in INTENT_DEFINITIONS:

        rows = train[
            train["Intent"].astype(str).str.strip() == intent
        ].copy()

        if rows.empty:
            continue

        rows["text_length"] = (
            rows["customer_text"]
            .fillna("")
            .astype(str)
            .str.len()
        )

        # Prefer a representative medium-length example.
        rows = rows.sort_values("text_length")

        middle = len(rows) // 2

        selected = [rows.iloc[middle]]

        if intent in extra_example_intents and len(rows) > 1:
            second_index = max(0, middle - 1)

            if second_index != middle:
                selected.append(rows.iloc[second_index])

        for row in selected:
            examples.append(
                (
                    str(row["customer_text"]).strip(),
                    intent,
                )
            )

    return examples

def build_user_prompt(
    customer_text: str,
    examples,
) -> str:

    example_blocks = []

    for index, (text, intent) in enumerate(
        examples,
        start=1,
    ):
        example_blocks.append(
            f"""Example {index}
Customer: {text}
Intent: {intent}"""
        )

    examples_text = "\n\n".join(
        example_blocks
    )

    return f"""
Here are representative labelled examples from the training set:

{examples_text}

Now classify this customer message.

Customer:
{customer_text}

Apply the intent definitions and boundary rules.

Classify the customer's PRIMARY issue.

Return only:

{{"intent":"CODE"}}
""".strip()


def main():

    train = pd.read_csv(TRAIN_PATH)
    test = pd.read_csv(TEST_PATH)

    train["Intent"] = (
        train["Intent"]
        .astype(str)
        .str.strip()
    )

    test["Intent"] = (
        test["Intent"]
        .astype(str)
        .str.strip()
    )

    # Build demonstrations once.
    examples = build_examples(train)

    system_prompt = build_system_prompt()

    client = ollama.Client(
        host="http://localhost:11434"
    )

    predictions = []

    total = len(test)

    for index, row in test.iterrows():

        customer_text = str(
            row["customer_text"]
        ).strip()

        user_prompt = build_user_prompt(
            customer_text=customer_text,
            examples=examples,
        )

        response = client.chat(
            model=MODEL_NAME,
            messages=[
                {
                    "role": "system",
                    "content": system_prompt,
                },
                {
                    "role": "user",
                    "content": user_prompt,
                },
            ],
            options={
                "temperature": 0,
                "num_predict": 30,
            },
        )

        raw_output = response["message"]["content"]

        predicted_intent = parse_intent(
            raw_output
        )

        predictions.append(
            predicted_intent
        )

        print(
            f"[{index + 1}/{total}] "
            f"gold={row['Intent']} "
            f"predicted={predicted_intent}"
        )

    accuracy = accuracy_score(
        test["Intent"],
        predictions,
    )

    macro_f1 = f1_score(
        test["Intent"],
        predictions,
        average="macro",
        zero_division=0,
    )

    results = test.copy()

    results["predicted_intent"] = predictions

    results["correct"] = (
        results["Intent"]
        == results["predicted_intent"]
    )

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    results.to_csv(
        OUTPUT_PATH,
        index=False,
    )

    print()
    print("=== LLM Intent Classifier ===")
    print(f"Model: {MODEL_NAME}")
    print(f"Test examples: {len(test)}")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Macro F1: {macro_f1:.4f}")
    print(
        f"Saved predictions to: {OUTPUT_PATH}"
    )


if __name__ == "__main__":
    main()