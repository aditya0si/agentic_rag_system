"""Deterministic stand-ins for the hosted LLM used by the agent modules.

The agent modules import ``get_llm`` at module level, so tests can replace that
attribute with ``fake_llm_factory(<stub>)`` and exercise the real prompt/chain
code without network access or API credentials.
"""

from typing import Any

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from langchain_core.runnables import RunnableLambda
from pydantic import Field


class RecordingFakeChatModel(BaseChatModel):
    """Answers with a fixed string and records the prompts it received."""

    response: str
    seen: list[list[BaseMessage]] = Field(default_factory=list)

    @property
    def _llm_type(self) -> str:
        return "recording-fake-chat-model"

    def _generate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: Any = None,
        **kwargs: Any,
    ) -> ChatResult:
        self.seen.append(list(messages))
        return ChatResult(generations=[ChatGeneration(message=AIMessage(content=self.response))])

    def rendered_prompts(self) -> list[str]:
        """Flatten every recorded prompt into searchable strings."""
        return [
            "\n".join(str(message.content) for message in prompt_messages)
            for prompt_messages in self.seen
        ]


class StructuredFakeChatModel:
    """Chat model stub supporting ``with_structured_output``.

    ``RecordingFakeChatModel`` cannot answer structured (pydantic) calls, so the
    graders that rely on ``with_structured_output`` get this stub instead.
    """

    def __init__(self, binary_score: str) -> None:
        self._binary_score = binary_score

    def with_structured_output(self, schema: type) -> RunnableLambda:
        score = self._binary_score
        return RunnableLambda(lambda _input: schema(binary_score=score))


def fake_llm_factory(model: Any) -> Any:
    """Return a ``get_llm``-compatible factory that always returns ``model``."""

    def factory(temperature: float = 0.0) -> Any:
        return model

    return factory
