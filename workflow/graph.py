from datetime import date, datetime

from langgraph.graph import StateGraph, START, END
from langgraph.types import interrupt
from langgraph.checkpoint.memory import InMemorySaver

from workflow.state import SupportState

from agents.triage_agent import TriageAgent
from agents.reservation_agent import ReservationAgent
from agents.payment_cancellation_agent import PaymentCancellationAgent
from agents.escalation_agent import EscalationAgent

from tools.customer_tools import extract_email
from tools.reservation_tools import get_reservation, get_reservation_by_email
from tools.invoice_tools import get_invoice
from tools.payment_tools import analyze_payment
from tools.refund_tools import analyze_cancellation, analyze_duplicate_payment
from tools.ticket_tools import create_escalation_ticket


def parse_request_date(value: str | None) -> date:
    if not value:
        return date.today()

    value = str(value).strip()

    for date_format in ["%Y-%m-%d", "%m/%d/%Y", "%d/%m/%Y"]:
        try:
            return datetime.strptime(value, date_format).date()
        except ValueError:
            continue

    raise ValueError(f"Unsupported request date format: {value}")


def is_romanian(text: str) -> bool:
    text = text.lower()

    romanian_markers = {
        "am", "este", "sunt", "vreau", "pot", "nu", "rezervarea",
        "rezervare", "plata", "anulare", "problema", "ajutor",
        "pentru", "care", "cand", "când", "unde", "buna", "bună",
        "taxat", "factura", "factură", "emailul", "meu", "mea",
        "primesc", "banii", "intra", "gazda", "răspunde", "raspunde"
    }

    words = set(
        text.replace(".", " ")
        .replace(",", " ")
        .replace("?", " ")
        .replace("!", " ")
        .split()
    )

    return len(words & romanian_markers) >= 1


def triage_node(state: SupportState) -> dict:
    agent = TriageAgent()
    result = agent.classify(state["customer_message"])

    return {
        "category": result.category,
        "urgency": result.urgency,
        "reservation_id": result.reservation_id,
        "summary": result.summary,
        "trace": [f"Triage → {result.category}"]
    }


def identify_reservation_node(state: SupportState) -> dict:
    reservation_id = state.get("reservation_id")
    reservation = None
    identification_method = None

    if reservation_id:
        reservation = get_reservation(reservation_id)
        identification_method = "reservation ID"
    else:
        email = extract_email(state["customer_message"])

        if email:
            reservation = get_reservation_by_email(email)
            identification_method = "email"

    if reservation:
        resolved_id = reservation["reservation_id"]

        return {
            "reservation": reservation,
            "reservation_id": resolved_id,
            "trace": [
                f"Reservation identified → {resolved_id} "
                f"(via {identification_method})"
            ]
        }

    return {
        "reservation": None,
        "reservation_id": reservation_id,
        "trace": ["Reservation identification failed"]
    }


def reservation_node(state: SupportState) -> dict:
    agent = ReservationAgent()

    response = agent.answer(
        customer_message=state["customer_message"],
        reservation=state["reservation"]
    )

    return {
        "final_response": response,
        "trace": ["Reservation Agent → response generated"]
    }


def payment_cancellation_node(state: SupportState) -> dict:
    reservation = state["reservation"]
    reservation_id = reservation["reservation_id"]
    invoice = get_invoice(reservation_id)
    romanian = is_romanian(state["customer_message"])

    if invoice is None:
        response = (
            "Nu am putut identifica factura asociată rezervării. "
            "Este necesară verificarea de către suport."
            if romanian
            else
            "I could not identify the invoice associated with the reservation. "
            "The case requires review by support."
        )

        return {
            "invoice": None,
            "final_response": response,
            "trace": [f"Invoice lookup failed → {reservation_id}"]
        }

    category = state["category"]

    if category == "Payment":
        decision = analyze_payment(invoice)

        if decision["type"] == "duplicate_payment":
            decision = analyze_duplicate_payment(invoice)

            trace = [
                "Payment analysis → duplicate payment detected",
                f"Refund calculated → {decision['refund_amount_eur']:.2f} EUR",
                f"Human approval required → {decision['requires_human_approval']}"
            ]
        else:
            trace = [
                f"Payment analysis → {decision['type']}"
            ]

    elif category == "Cancellation":
        request_date = parse_request_date(state.get("request_date"))

        if reservation["status"] == "cancelled":
            response = (
                "Rezervarea este deja anulată. "
                "Pentru verificarea rambursării, te rugăm să contactezi suportul."
                if romanian
                else
                "The reservation is already cancelled. "
                "Please contact support to verify the refund."
            )

            return {
                "invoice": invoice,
                "final_response": response,
                "trace": [
                    f"Cancellation check → {reservation_id} already cancelled"
                ]
            }

        decision = analyze_cancellation(
            reservation,
            request_date=request_date
        )

        trace = [
            (
                f"Cancellation analysis → "
                f"{decision['days_before_check_in']} days before check-in"
            ),
            f"Refund percentage → {decision['refund_percentage']}%",
            f"Refund calculated → {decision['refund_amount_eur']:.2f} EUR",
            f"Human approval required → {decision['requires_human_approval']}"
        ]

    else:
        raise ValueError(f"Unsupported category: {category}")

    approval_required = decision.get(
        "requires_human_approval",
        False
    )

    return {
        "invoice": invoice,
        "decision": decision,
        "requires_human_approval": approval_required,
        "trace": trace
    }


def human_approval_node(state: SupportState) -> dict:
    decision = state["decision"]

    human_decision = interrupt({
        "reservation_id": state["reservation_id"],
        "category": state["category"],
        "refund_amount_eur": decision["refund_amount_eur"],
        "reason": decision["type"],
        "message": "This refund requires human approval."
    })

    approved = human_decision.get("approved", False)
    approval_status = "approved" if approved else "rejected"

    return {
        "approval_status": approval_status,
        "trace": [f"Human approval → {approval_status}"]
    }


def payment_response_node(state: SupportState) -> dict:
    agent = PaymentCancellationAgent()
    decision = dict(state["decision"])

    if state.get("requires_human_approval", False):
        decision["approval_status"] = state.get("approval_status")

    response = agent.answer(
        customer_message=state["customer_message"],
        reservation=state["reservation"],
        invoice=state["invoice"],
        decision=decision
    )

    return {
        "final_response": response,
        "trace": [
            "Payment & Cancellation Agent → response generated"
        ]
    }


def escalation_node(state: SupportState) -> dict:
    ticket = create_escalation_ticket(
        customer_message=state["customer_message"],
        reason=state["summary"],
        urgency=state["urgency"],
        reservation=state.get("reservation")
    )

    agent = EscalationAgent()

    response = agent.answer(
        customer_message=state["customer_message"],
        ticket=ticket,
        reservation=state.get("reservation")
    )

    return {
        "ticket": ticket,
        "final_response": response,
        "trace": [
            f"Escalation ticket created → {ticket['ticket_id']}",
            "Escalation Agent → case sent to human support"
        ]
    }


def unclear_node(state: SupportState) -> dict:
    if is_romanian(state["customer_message"]):
        response = (
            "Nu am înțeles exact solicitarea. "
            "Te rog să oferi mai multe detalii despre problema întâmpinată."
        )
    else:
        response = (
            "I could not clearly understand your request. "
            "Please provide more details about the issue you are experiencing."
        )

    return {
        "final_response": response,
        "trace": [
            "Unclear request → additional details requested"
        ]
    }


def missing_reservation_node(state: SupportState) -> dict:
    if is_romanian(state["customer_message"]):
        response = (
            "Nu am putut identifica rezervarea. "
            "Te rog să furnizezi ID-ul rezervării sau adresa de email "
            "asociată acesteia."
        )
    else:
        response = (
            "I could not identify the reservation. "
            "Please provide the reservation ID or the email address "
            "associated with it."
        )

    return {
        "final_response": response,
        "trace": [
            "Missing reservation → reservation ID or email requested"
        ]
    }


def route_after_triage(state: SupportState) -> str:
    if state["category"] == "Unclear":
        return "unclear"

    return "identify_reservation"


def route_after_identification(state: SupportState) -> str:
    category = state["category"]

    if category == "Urgent Problem":
        return "escalation"

    if state.get("reservation") is None:
        return "missing_reservation"

    if category == "Reservation":
        return "reservation"

    if category in ("Payment", "Cancellation"):
        return "payment_cancellation"

    return "unclear"


def route_after_payment(state: SupportState) -> str:
    if state.get("final_response"):
        return "end"

    if state.get("requires_human_approval", False):
        return "human_approval"

    return "payment_response"


def build_graph():
    workflow = StateGraph(SupportState)

    workflow.add_node("triage", triage_node)
    workflow.add_node("identify_reservation", identify_reservation_node)
    workflow.add_node("reservation", reservation_node)
    workflow.add_node("payment_cancellation", payment_cancellation_node)
    workflow.add_node("payment_response", payment_response_node)
    workflow.add_node("human_approval", human_approval_node)
    workflow.add_node("escalation", escalation_node)
    workflow.add_node("unclear", unclear_node)
    workflow.add_node("missing_reservation", missing_reservation_node)

    workflow.add_edge(START, "triage")

    workflow.add_conditional_edges(
        "triage",
        route_after_triage,
        {
            "unclear": "unclear",
            "identify_reservation": "identify_reservation"
        }
    )

    workflow.add_conditional_edges(
        "identify_reservation",
        route_after_identification,
        {
            "reservation": "reservation",
            "payment_cancellation": "payment_cancellation",
            "escalation": "escalation",
            "missing_reservation": "missing_reservation",
            "unclear": "unclear"
        }
    )

    workflow.add_conditional_edges(
        "payment_cancellation",
        route_after_payment,
        {
            "human_approval": "human_approval",
            "payment_response": "payment_response",
            "end": END
        }
    )

    workflow.add_edge(
        "human_approval",
        "payment_response"
    )

    for node in [
        "reservation",
        "payment_response",
        "escalation",
        "unclear",
        "missing_reservation"
    ]:
        workflow.add_edge(node, END)

    memory = InMemorySaver()

    return workflow.compile(
        checkpointer=memory
    )