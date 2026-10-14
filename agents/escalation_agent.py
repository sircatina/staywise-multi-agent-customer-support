import json
from pathlib import Path

from llm.factory import get_llm_provider


class EscalationAgent:

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
            / "escalation_prompt.txt"
        )

        return prompt_path.read_text(encoding="utf-8")

    def answer(
        self,
        customer_message: str,
        ticket: dict,
        reservation: dict | None = None
    ) -> str:

        context = {
            "ticket": ticket,
            "reservation": reservation
        }

        user_prompt = f"""
CUSTOMER MESSAGE:
{customer_message}

VERIFIED SYSTEM DATA:
{json.dumps(context, ensure_ascii=False, indent=2)}

Inform the customer about the escalation using only the
verified information above.
"""

        return self.llm.generate(
            system_prompt=self.system_prompt,
            user_prompt=user_prompt
        )