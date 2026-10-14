import operator

from typing import TypedDict, Any, Annotated


class SupportState(TypedDict, total=False):
    # Customer input
    customer_message: str
    request_date: str

    # Triage result
    category: str
    urgency: str
    reservation_id: str | None
    summary: str

    # Verified data
    reservation: dict | None
    invoice: dict | None

    # Business decision
    decision: dict[str, Any] | None

    # Human approval
    requires_human_approval: bool
    approval_status: str | None

    # Escalation
    ticket: dict | None

    # System execution trace
    trace: Annotated[list[str], operator.add]

    # Final customer response
    final_response: str