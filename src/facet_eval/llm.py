"""Thin wrapper around the local Ollama server with structured (JSON) output.

This is the single choke-point through which every LLM call passes — Phase 1
enrichment and Phase 3 scoring both use `chat_structured`. Centralizing it means
retries, JSON-mode, and schema validation are implemented once.

Why structured output (not "parse the prose"): we pass Ollama the pydantic JSON
schema via `format=`, which constrains decoding so the model returns valid JSON
matching our shape. We then validate with pydantic. Two layers of safety against
the classic "the model added a chatty preamble and broke json.loads" failure.
"""

from __future__ import annotations

import json
from typing import TypeVar

import ollama
from pydantic import BaseModel, ValidationError

from . import config

T = TypeVar("T", bound=BaseModel)

# One reused client pointed at the local server.
_client = ollama.Client(host=config.OLLAMA_HOST)


class LLMError(RuntimeError):
    """Raised when the model cannot produce schema-valid output after retries."""


def chat_structured(
    messages: list[dict[str, str]],
    schema: type[T],
    *,
    model: str | None = None,
    temperature: float = 0.0,
    retries: int = 2,
) -> T:
    """Call the model and return a validated instance of `schema`.

    Args:
        messages: standard chat messages [{"role": ..., "content": ...}].
        schema: a pydantic model class; its JSON schema constrains the output.
        model: override the configured model.
        temperature: 0.0 for deterministic enrichment/scoring.
        retries: extra attempts if validation fails (model nudged each time).

    Raises:
        LLMError: if no attempt yields schema-valid JSON.
    """
    model = model or config.OLLAMA_MODEL
    fmt = schema.model_json_schema()
    last_err: Exception | None = None
    msgs = list(messages)

    for attempt in range(retries + 1):
        try:
            resp = _client.chat(
                model=model,
                messages=msgs,
                format=fmt,
                options={"temperature": temperature},
            )
            content = resp["message"]["content"]
            return schema.model_validate_json(content)
        except (ValidationError, json.JSONDecodeError, KeyError) as e:
            last_err = e
            # Nudge the model with the error so the retry can self-correct.
            msgs = list(messages) + [
                {
                    "role": "user",
                    "content": (
                        "Your previous reply was not valid for the required schema "
                        f"({type(e).__name__}). Return ONLY a JSON object that "
                        "matches the schema exactly, with no extra text."
                    ),
                }
            ]

    raise LLMError(
        f"model {model!r} failed to produce schema-valid output after "
        f"{retries + 1} attempts: {last_err}"
    )


def ping() -> bool:
    """Return True if the configured model is available on the server."""
    try:
        names = {m.get("model", "") for m in _client.list().get("models", [])}
        return any(config.OLLAMA_MODEL.split(":")[0] in n for n in names)
    except Exception:
        return False
