from abc import ABC, abstractmethod
from collections.abc import Iterator


class LLMProvider(ABC):

    @abstractmethod
    def generate(
        self,
        system_prompt: str,
        user_prompt: str
    ) -> str:
        """Generate a response from the LLM."""
        pass

    @abstractmethod
    def stream(
        self,
        system_prompt: str,
        user_prompt: str
    ) -> Iterator[str]:
        """Generate a streaming response from the LLM."""
        pass