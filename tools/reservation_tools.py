from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parent.parent
RESERVATIONS_PATH = PROJECT_ROOT / "data" / "reservations.csv"


def _load_reservations() -> pd.DataFrame:
    """Load the reservations dataset."""
    return pd.read_csv(RESERVATIONS_PATH)


def get_reservation(reservation_id: str) -> dict | None:
    """
    Find a reservation by reservation ID.

    Returns the reservation as a dictionary,
    or None if it does not exist.
    """
    reservations = _load_reservations()

    match = reservations[
        reservations["reservation_id"].str.upper()
        == reservation_id.strip().upper()
    ]

    if match.empty:
        return None

    return match.iloc[0].to_dict()


def get_reservation_by_email(email: str) -> dict | None:
    """
    Find a reservation using the customer's email address.

    Returns the reservation as a dictionary,
    or None if no reservation is found.
    """
    reservations = _load_reservations()

    match = reservations[
        reservations["email"].str.lower()
        == email.strip().lower()
    ]

    if match.empty:
        return None

    return match.iloc[0].to_dict()