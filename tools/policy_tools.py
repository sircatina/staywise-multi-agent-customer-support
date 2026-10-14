from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent
POLICIES_DIR = PROJECT_ROOT / "policies"


def _load_policy(filename: str) -> str:
    path = POLICIES_DIR / filename

    if not path.exists():
        raise FileNotFoundError(
            f"Policy file not found: {path}"
        )

    return path.read_text(encoding="utf-8")


def get_cancellation_policy() -> str:
    """Return the StayWise cancellation policy."""
    return _load_policy("cancellation_policy.txt")


def get_payment_policy() -> str:
    """Return the StayWise payment policy."""
    return _load_policy("payment_policy.txt")