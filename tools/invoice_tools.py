from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parent.parent
INVOICES_PATH = PROJECT_ROOT / "data" / "invoices.csv"


def _load_invoices() -> pd.DataFrame:
    """Load the invoices dataset."""
    return pd.read_csv(INVOICES_PATH)


def get_invoice(reservation_id: str) -> dict | None:
    """
    Find the invoice associated with a reservation.

    Returns the invoice as a dictionary,
    or None if no invoice exists.
    """
    invoices = _load_invoices()

    match = invoices[
        invoices["reservation_id"].str.upper()
        == reservation_id.strip().upper()
    ]

    if match.empty:
        return None

    return match.iloc[0].to_dict()