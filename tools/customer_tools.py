import re


EMAIL_PATTERN = r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b"


def extract_email(text: str) -> str | None:
    """
    Extract the first email address found in a customer message.
    """
    match = re.search(EMAIL_PATTERN, text)

    if match:
        return match.group(0).lower()

    return None