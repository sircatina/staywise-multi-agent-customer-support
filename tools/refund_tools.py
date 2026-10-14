from datetime import date, datetime


HUMAN_APPROVAL_THRESHOLD_EUR = 200.0


def calculate_days_before_check_in(
    check_in: str,
    request_date: date | None = None
) -> int:
    """
    Calculate the number of full days between the cancellation
    request and the check-in date.
    """
    if request_date is None:
        request_date = date.today()

    check_in_date = datetime.strptime(
        check_in,
        "%Y-%m-%d"
    ).date()

    return (check_in_date - request_date).days


def get_refund_percentage(days_before_check_in: int) -> int:
    """
    Determine the refund percentage according to
    the StayWise cancellation policy.
    """
    if days_before_check_in > 7:
        return 100

    if 3 <= days_before_check_in <= 7:
        return 50

    return 0


def calculate_refund(
    total_eur: float,
    refund_percentage: int
) -> float:
    """Calculate the refund amount."""

    refund = total_eur * refund_percentage / 100

    return round(refund, 2)


def requires_human_approval(refund_amount: float) -> bool:
    """
    Refunds strictly greater than 200 EUR
    require human approval.
    """
    return refund_amount > HUMAN_APPROVAL_THRESHOLD_EUR


def analyze_cancellation(
    reservation: dict,
    request_date: date | None = None
) -> dict:
    """
    Analyze cancellation eligibility and calculate the refund
    according to the StayWise cancellation policy.
    """

    days_before = calculate_days_before_check_in(
        reservation["check_in"],
        request_date
    )

    percentage = get_refund_percentage(days_before)

    refund_amount = calculate_refund(
        float(reservation["total_eur"]),
        percentage
    )

    return {
        "type": "cancellation_refund",
        "days_before_check_in": days_before,
        "refund_percentage": percentage,
        "refund_amount_eur": refund_amount,
        "requires_human_approval": (
            requires_human_approval(refund_amount)
        )
    }


def analyze_duplicate_payment(invoice: dict) -> dict:
    """
    Determine the refund required for a duplicate payment.
    """

    payment_count = int(invoice["payment_count"])

    if payment_count <= 1:
        return {
            "type": "no_duplicate_payment",
            "refund_amount_eur": 0.0,
            "requires_human_approval": False
        }

    extra_payments = payment_count - 1

    refund_amount = round(
        float(invoice["amount_eur"]) * extra_payments,
        2
    )

    return {
        "type": "duplicate_payment_refund",
        "extra_payments": extra_payments,
        "refund_amount_eur": refund_amount,
        "requires_human_approval": (
            requires_human_approval(refund_amount)
        )
    }