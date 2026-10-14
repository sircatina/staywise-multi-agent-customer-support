def analyze_payment(invoice: dict) -> dict:
    """
    Analyze verified invoice data and determine
    the payment situation.
    """

    payment_status = invoice["payment_status"]
    payment_count = int(invoice["payment_count"])

    if payment_count > 1:
        return {
            "type": "duplicate_payment",
            "payment_status": payment_status,
            "payment_count": payment_count,
            "duplicate_payment": True
        }

    if payment_status == "failed":
        return {
            "type": "failed_payment",
            "payment_status": payment_status,
            "payment_count": payment_count,
            "duplicate_payment": False
        }

    if payment_status == "refunded":
        return {
            "type": "refunded",
            "payment_status": payment_status,
            "payment_count": payment_count,
            "duplicate_payment": False
        }

    return {
        "type": "successful_payment",
        "payment_status": payment_status,
        "payment_count": payment_count,
        "duplicate_payment": False
    }