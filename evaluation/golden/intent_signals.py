INTENT_SIGNALS = {
    "order_status": [
        "where is my order",
        "where is my package",
        "where's my package",
        "where is my parcel",
        "track my order",
        "tracking",
        "order status",
        "package status",
        "when will my order",
        "when will my package",
    ],

    "delivery_late": [
        "late",
        "delayed",
        "still hasn't arrived",
        "still hasnt arrived",
        "hasn't arrived",
        "hasnt arrived",
        "supposed to arrive",
        "supposed to be delivered",
        "was due",
        "overdue",
        "days late",
        "delivery date passed",
        "didn't arrive",
        "didnt arrive",
    ],

    "delivered_not_received": [
        "says delivered",
        "marked delivered",
        "shows delivered",
        "was delivered",
        "package was delivered",
        "order was delivered",
        "but i didn't receive",
        "but i did not receive",
        "never received",
        "not received",
        "nothing arrived",
        "not here",
    ],

    "delivery_promise": [
        "next day",
        "one day delivery",
        "same day",
        "two day delivery",
        "2 day delivery",
        "guaranteed delivery",
        "delivery guarantee",
        "prime delivery",
        "promised delivery",
        "guaranteed date",
    ],

    "missing_or_wrong_item": [
        "wrong item",
        "wrong product",
        "incorrect item",
        "missing item",
        "item missing",
        "missing product",
        "not what i ordered",
        "different product",
        "received the wrong",
    ],

    "damaged_item_or_package": [
        "damaged",
        "broken",
        "cracked",
        "damaged package",
        "damaged box",
        "broken item",
        "arrived broken",
        "packaging",
        "opened package",
        "crushed",
    ],

    "return_or_refund": [
        "refund",
        "refunded",
        "return",
        "returning",
        "send it back",
        "money back",
        "reimbursement",
        "cancelled and refund",
        "cancelled but no refund",
    ],

    "payment_or_charge": [
        "charged",
        "charge",
        "payment",
        "credit card",
        "debit card",
        "card",
        "billing",
        "charged twice",
        "don't recognize this charge",
        "do not recognize this charge",
        "payment failed",
        "payment verification",
    ],

    "prime_membership": [
        "prime membership",
        "prime member",
        "amazon prime",
        "prime subscription",
        "prime renewal",
        "prime fee",
        "prime charge",
        "cancel prime",
        "prime student",
        "prime music",
        "prime video",
    ],

    "account_or_access": [
        "account",
        "password",
        "login",
        "log in",
        "logged out",
        "account blocked",
        "account locked",
        "hacked",
        "email changed",
        "someone changed",
        "can't access",
        "cannot access",
    ],

    "product_or_device_help": [
        "echo",
        "alexa",
        "fire tablet",
        "fire tv",
        "kindle",
        "device",
        "not working",
        "doesn't work",
        "doesnt work",
        "compatible",
        "compatibility",
        "how do i use",
        "how does this work",
    ],

    "customer_service_followup": [
        "still waiting",
        "waiting for a reply",
        "waiting for response",
        "no response",
        "no reply",
        "nobody replied",
        "haven't heard back",
        "havent heard back",
        "call me",
        "called me",
        "get back to me",
        "follow up",
        "follow-up",
        "update",
    ],
}


def signal_intents(text: str) -> list[str]:
    """Return all intent areas with at least one lexical signal."""
    lower = text.lower()
    matched = []

    for intent, phrases in INTENT_SIGNALS.items():
        for phrase in phrases:
            if phrase in lower:
                matched.append(intent)
                break

    return matched