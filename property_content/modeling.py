"""Small model abstraction used for dependency injection and mocking."""

from __future__ import annotations

from typing import Any, Protocol


class AsyncTextModel(Protocol):
    async def complete(
        self,
        prompt: str,
        *,
        response_schema: dict[str, Any] | None = None,
    ) -> str:
        """Return a text completion for ``prompt``."""


class StaticTextModel:
    """Network-free model used by examples and tests."""

    def __init__(self, completions: list[str]) -> None:
        self._completions = iter(completions)
        self.prompts: list[str] = []

    async def complete(
        self,
        prompt: str,
        *,
        response_schema: dict[str, Any] | None = None,
    ) -> str:
        self.prompts.append(prompt)
        try:
            return next(self._completions)
        except StopIteration as exc:
            raise RuntimeError("StaticTextModel has no completion remaining") from exc
