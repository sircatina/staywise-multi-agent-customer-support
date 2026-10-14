
from datetime import datetime
from uuid import uuid4


def create_escalation_ticket(
    customer_message: str,
    reason: str,
    urgency: str,
    reservation: dict | None = None
) -> dict:
    """
    Create an escalation ticket for a human support operator.
    """

    ticket_id = f"ESC-{uuid4().hex[:8].upper()}"

    return {
        "ticket_id": ticket_id,
        "created_at": datetime.now().isoformat(
            timespec="seconds"
        ),
        "status": "open",
        "urgency": urgency,
        "reason": reason,
        "reservation_id": (
            reservation.get("reservation_id")
            if reservation
            else None
        ),
        "customer_message": customer_message
    }