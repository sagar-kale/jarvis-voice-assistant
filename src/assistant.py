from __future__ import annotations

import anthropic

from config.settings import ANTHROPIC_API_KEY, CLAUDE_MODEL, SYSTEM_PROMPT


class ClaudeAssistant:
    """Stateful Claude conversation manager."""

    def __init__(
        self,
        api_key: str | None = None,
        model: str | None = None,
        system_prompt: str | None = None,
    ) -> None:
        self._client = anthropic.Anthropic(api_key=api_key or ANTHROPIC_API_KEY)
        self._model = model or CLAUDE_MODEL
        self._system = system_prompt or SYSTEM_PROMPT
        self._history: list[dict] = []

    # ── Public API ─────────────────────────────────────────────────────────────

    def chat(self, user_message: str) -> str:
        """Send a user message and return the assistant reply."""
        self._history.append({"role": "user", "content": user_message})

        response = self._client.messages.create(
            model=self._model,
            max_tokens=1024,
            system=self._system,
            messages=self._history,
        )

        reply = response.content[0].text
        self._history.append({"role": "assistant", "content": reply})
        return reply

    def reset(self) -> None:
        """Clear conversation history."""
        self._history.clear()

    @property
    def history(self) -> list[dict]:
        return list(self._history)
