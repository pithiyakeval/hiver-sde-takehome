import re
from typing import Optional


def normalize_text(text: str) -> str:
    """Normalize customer text for deterministic rule matching."""

    return re.sub(
        r"\s+",
        " ",
        text.lower().strip(),
    )


def classify_by_rule(text: str) -> Optional[str]:
    """
    Apply only high-precision intent rules.

    Returns one of the project's canonical intent codes when a strong
    deterministic pattern is found. Otherwise returns None so the
    ML
    classifier can handle the message.
    """

    text = normalize_text(text)

    if not text:
        return None

    # ------------------------------------------------------------------
    # Delivered but not received -> DNR
    # ------------------------------------------------------------------

    delivered_signal = re.search(
        r"\b("
        r"delivered|marked delivered|shows delivered|"
        r"delivery says delivered|tracking says delivered"
        r")\b",
        text,
    )

    not_received_signal = re.search(
        r"\b("
        r"not received|didn't receive|did not receive|"
        r"never received|haven't received|have not received|"
        r"not got|didn't get|did not get|"
        r"never got|haven't got|have not got|"
        r"can't find|cannot find"
        r")\b",
        text,
    )

    if delivered_signal and not_received_signal:
        return "DNR"

        # ------------------------------------------------------------------
    # General order status / location -> OS
    # ------------------------------------------------------------------

    order_status_signal = re.search(
        r"\b("
        r"where is my order|"
        r"where's my order|"
        r"where is the order|"
        r"where's the order|"
        r"where is my package|"
        r"where's my package|"
        r"where is my parcel|"
        r"where's my parcel|"
        r"order status|"
        r"track my order|"
        r"tracking my order|"
        r"track the order|"
        r"order tracking|"
        r"where can i track my order"
        r")\b",
        text,
    )

    if order_status_signal:
        return "OS"

    # ------------------------------------------------------------------
    # Damaged item / package -> DIP
    # ------------------------------------------------------------------

    damaged_signal = re.search(
        r"\b("
        r"damaged|damage|broken|cracked|"
        r"destroyed|defective|dented|"
        r"arrived broken|arrived damaged"
        r")\b",
        text,
    )

    if damaged_signal:
        return "DIP"

    # ------------------------------------------------------------------
    # Missing or wrong item -> MWI
    # ------------------------------------------------------------------

    missing_item_signal = re.search(
        r"\b("
        r"missing item|item missing|"
        r"missing from my order|"
        r"wrong item|wrong product|"
        r"received the wrong|"
        r"sent the wrong|"
        r"something is missing"
        r")\b",
        text,
    )

    if missing_item_signal:
        return "MWI"

    # ------------------------------------------------------------------
    # Payment / charge -> PC
    # ------------------------------------------------------------------

    payment_signal = re.search(
        r"\b("
        r"charged|charge|payment|"
        r"billing|bill|invoice|"
        r"credit card|debit card|"
        r"card charged|"
        r"charged twice|double charged|"
        r"unauthorized|unauthorised|"
        r"fraud|"
        r"emi"
        r")\b",
        text,
    )

    if payment_signal:
        return "PC"

    # ------------------------------------------------------------------
    # Prime membership -> PM
    # ------------------------------------------------------------------

    prime_membership_signal = re.search(
        r"\b("
        r"prime membership|"
        r"prime member|"
        r"amazon prime|"
        r"prime subscription|"
        r"prime renewal|"
        r"cancel prime|"
        r"prime benefits|"
        r"prime trial"
        r")\b",
        text,
    )

    prime_delivery_signal = re.search(
        r"\b("
        r"delivery|"
        r"delivered|"
        r"deliver|"
        r"late|"
        r"on time|"
        r"not arrived|"
        r"arrived|"
        r"shipping|"
        r"shipment|"
        r"order"
        r")\b",
        text,
    )



    if prime_membership_signal:
        return "PM"

    # ------------------------------------------------------------------
    # Account / access -> AA
    # ------------------------------------------------------------------

    account_signal = re.search(
        r"\b("
        r"account locked|"
        r"account blocked|"
        r"can't log in|"
        r"cannot log in|"
        r"can't login|"
        r"cannot login|"
        r"unable to log in|"
        r"unable to login|"
        r"forgot password|"
        r"password reset|"
        r"account access|"
        r"locked out"
        r")\b",
        text,
    )

    if account_signal:
        return "AA"

    # ------------------------------------------------------------------
    # Return / refund -> RR
    # ------------------------------------------------------------------

    refund_signal = re.search(
        r"\b("
        r"refund|"
        r"refunded|"
        r"money back|"
        r"return my|"
        r"return an item|"
        r"return this|"
        r"send it back|"
        r"refund status"
        r")\b",
        text,
    )

    if refund_signal:
        return "RR"

    # ------------------------------------------------------------------
    # No high-confidence rule match
    # ------------------------------------------------------------------

    return None