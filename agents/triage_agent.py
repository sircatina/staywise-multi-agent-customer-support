import json
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ValidationError

from llm.factory import get_llm_provider


class TriageResult(BaseModel):
    category: Literal[
        "Reservation",
        "Payment",
        "Cancellation",
        "Urgent Problem",
        "Unclear"
    ]

    urgency: Literal[
        "normal",
        "high",
        "unknown"
    ]

    reservation_id: str | None
    summary: str


class TriageAgent:

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
            / "triage_prompt.txt"
        )

        return prompt_path.read_text(encoding="utf-8")

    def classify(self, customer_message: str) -> TriageResult:
        response = self.llm.generate(
            system_prompt=self.system_prompt,
            user_prompt=customer_message
        )

        try:
            data = json.loads(response)
            return TriageResult.model_validate(data)

        except (json.JSONDecodeError, ValidationError) as error:
            raise ValueError(
                f"Invalid response received from Triage Agent:\n{response}"
            ) from error