import json
from pathlib import Path

from llm.factory import get_llm_provider


class ReservationAgent:

    def __init__(self):
        self.llm = get_llm_provider()
        self.system_prompt = self._load_prompt()

    @staticmethod
    def _load_prompt() -> str:
        project_root = Path(__file__).resolve().parent.parent

        prompt_path = (
            project_root
            / "prompts"
            / "agents"
            / "reservation_prompt.txt"
        )

        return prompt_path.read_text(encoding="utf-8")

    def answer(
        self,
        customer_message: str,
        reservation: dict
    ) -> str:

        reservation_data = json.dumps(
            reservation,
            ensure_ascii=False,
            indent=2
        )

        user_prompt = f"""
CUSTOMER MESSAGE:
{customer_message}

RESERVATION DATA:
{reservation_data}

Answer the customer's question using only the reservation data above.
"""

        return self.llm.generate(
            system_prompt=self.system_prompt,
            user_prompt=user_prompt
        )