import os

from dotenv import load_dotenv

from llm.base import LLMProvider
from llm.groq_provider import GroqProvider


load_dotenv()


def get_llm_provider() -> LLMProvider:
    provider = os.getenv("LLM_PROVIDER", "groq").lower()
    model = os.getenv(
        "LLM_MODEL",
        "llama-3.3-70b-versatile"
    )

    if provider == "groq":
        return GroqProvider(model=model)

    raise ValueError(
        f"Unsupported LLM provider: {provider}"
    )