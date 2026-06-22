"""LLM providers — Runtime.

A thin provider abstraction so the framework's core never hard-depends on a
model SDK. ``EchoProvider`` is a deterministic offline stand-in (the demo and
tests run with zero credentials). ``AnthropicProvider`` calls Claude via the
official ``anthropic`` SDK when ``ANTHROPIC_API_KEY`` is set.
"""

from __future__ import annotations

import os
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

DEFAULT_MODEL = "claude-opus-4-8"


@dataclass
class Message:
    role: str  # "user" | "assistant"
    content: str


@dataclass
class LLMResponse:
    text: str
    model: str
    input_tokens: int = 0
    output_tokens: int = 0
    raw: Any = field(default=None, repr=False)


def approx_tokens(text: str) -> int:
    """Rough token estimate (~4 chars/token) for the offline provider."""
    return max(1, len(text) // 4)


class LLMProvider(ABC):
    name: str = "base"

    @abstractmethod
    def complete(
        self,
        *,
        system: str,
        messages: list[Message],
        model: str,
        max_tokens: int = 1024,
    ) -> LLMResponse: ...


class EchoProvider(LLMProvider):
    """Deterministic, offline provider.

    It returns a structured, role-aware echo of the input so a full pipeline
    runs — and demonstrates tracing, cost, redaction, and learning — without
    any API key. Output is stable across runs, which keeps tests reproducible.
    """

    name = "echo"

    def complete(self, *, system, messages, model="echo", max_tokens=1024) -> LLMResponse:
        role = system.strip().splitlines()[0][:80] if system.strip() else "assistant"
        user_text = messages[-1].content if messages else ""
        body = " ".join(user_text.split())
        if len(body) > 240:
            body = body[:240] + "…"
        text = f"[{role}] Processed request: {body}"
        return LLMResponse(
            text=text,
            model="echo",
            input_tokens=approx_tokens(system + user_text),
            output_tokens=approx_tokens(text),
        )


class AnthropicProvider(LLMProvider):
    """Calls Claude through the official ``anthropic`` Python SDK.

    Install with ``pip install 'erdos-fai[anthropic]'`` and set
    ``ANTHROPIC_API_KEY``. Defaults to the most capable Claude model.
    """

    name = "anthropic"

    def __init__(self, api_key: str | None = None, default_model: str = DEFAULT_MODEL) -> None:
        try:
            import anthropic  # noqa: F401
        except ImportError as exc:  # pragma: no cover - exercised only without the dep
            raise ImportError(
                "AnthropicProvider needs the 'anthropic' package. "
                "Install it with: pip install 'erdos-fai[anthropic]'"
            ) from exc
        import anthropic

        self._client = anthropic.Anthropic(api_key=api_key or os.environ.get("ANTHROPIC_API_KEY"))
        self.default_model = default_model

    def complete(self, *, system, messages, model=None, max_tokens=1024) -> LLMResponse:
        model = model or self.default_model
        resp = self._client.messages.create(
            model=model,
            max_tokens=max_tokens,
            system=system or None,
            messages=[{"role": m.role, "content": m.content} for m in messages],
        )
        text = "".join(b.text for b in resp.content if getattr(b, "type", None) == "text")
        return LLMResponse(
            text=text,
            model=resp.model,
            input_tokens=resp.usage.input_tokens,
            output_tokens=resp.usage.output_tokens,
            raw=resp,
        )


def default_provider() -> LLMProvider:
    """AnthropicProvider when a key + SDK are available, else EchoProvider."""
    if os.environ.get("ANTHROPIC_API_KEY"):
        try:
            return AnthropicProvider()
        except ImportError:
            pass
    return EchoProvider()
