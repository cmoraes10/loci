"""Reference ModelExtractor implementation.

Calls any OpenAI-compatible endpoint and asks for up to four facts as JSON.
No SDK dependency — raw urllib so this module adds nothing to the caller's
dependency tree.

Usage:

    from loci.extractors import build_extractor, ModelExtractorConfig

    extractor = build_extractor([
        ModelExtractorConfig(
            base_url="https://api.openai.com/v1",
            api_key=os.environ["OPENAI_API_KEY"],
            model="gpt-4o-mini",
        ),
        ModelExtractorConfig(
            base_url="https://api.anthropic.com/v1",
            api_key=os.environ["ANTHROPIC_API_KEY"],
            model="claude-haiku-4-5-20251001",
            extra_headers={"anthropic-version": "2023-06-01"},
        ),
    ])

    memories = extractor(user_text, assistant_text)
"""

from __future__ import annotations

import json
import time
import urllib.error
import urllib.request
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from loci.core.models import Category, Importance, Memory, Source, Status

_SYSTEM = """\
You are a memory extraction assistant. Given a conversation exchange, identify
facts worth remembering long-term about the user — preferences, decisions,
constraints, goals, relationships, routines. Skip passing mood and small talk.

Respond with valid JSON only, no markdown fences, using this exact shape:
{"facts": [{"key": "unique_slug_under_48_chars", "title": "Short label (max 80 chars)", "content": "Full statement as a sentence (max 220 chars)", "category": "preferences", "importance": "medium"}]}

Valid category values: routine, study, preferences, finance, goals, relationships, constraints, ephemeral
Valid importance values: critical, high, medium, low

Return at most 4 facts. If nothing is worth remembering, return {"facts": []}.
"""

_VALID_CATEGORIES = {c.value for c in Category}
_VALID_IMPORTANCE = {i.value for i in Importance}


@dataclass
class ModelExtractorConfig:
    """One provider endpoint. Supply multiple to get ordered fallback on failure."""

    base_url: str
    api_key: str
    model: str
    extra_headers: dict[str, str] = field(default_factory=dict)
    timeout: int = 20


def _call(config: ModelExtractorConfig, user_text: str, assistant_text: str) -> dict[str, Any]:
    body = json.dumps({
        "model": config.model,
        "messages": [
            {"role": "system", "content": _SYSTEM},
            {"role": "user", "content": f"User: {user_text}\n\nAssistant: {assistant_text}"},
        ],
        "temperature": 0,
        "max_tokens": 512,
    }).encode()

    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {config.api_key}",
        **config.extra_headers,
    }

    req = urllib.request.Request(
        f"{config.base_url.rstrip('/')}/chat/completions",
        data=body,
        headers=headers,
        method="POST",
    )

    with urllib.request.urlopen(req, timeout=config.timeout) as resp:
        return json.loads(resp.read())


def _parse(raw: dict[str, Any]) -> list[Memory]:
    """Turn the model's JSON payload into Memory objects, dropping malformed entries."""
    try:
        text = raw["choices"][0]["message"]["content"]
        facts = json.loads(text).get("facts", [])
    except (KeyError, IndexError, json.JSONDecodeError):
        return []

    memories: list[Memory] = []
    for f in facts[:4]:
        try:
            category = f["category"] if f.get("category") in _VALID_CATEGORIES else "preferences"
            importance = f["importance"] if f.get("importance") in _VALID_IMPORTANCE else "medium"
            memories.append(
                Memory(
                    key=str(f.get("key", ""))[:48] or "model_fact",
                    title=str(f.get("title", ""))[:80],
                    content=str(f.get("content", ""))[:220],
                    category=Category(category),
                    importance=Importance(importance),
                    source=Source.ASSISTANT_INFERENCE,
                    status=Status.HYPOTHESIS,
                    confidence=0.8,
                )
            )
        except (KeyError, ValueError):
            continue
    return memories


def _with_retry(
    config: ModelExtractorConfig,
    user_text: str,
    assistant_text: str,
    max_attempts: int = 3,
) -> list[Memory]:
    """Exponential backoff on transient failures (429, 5xx). Fail fast on other 4xx."""
    for attempt in range(max(max_attempts, 1)):
        try:
            return _parse(_call(config, user_text, assistant_text))
        except urllib.error.HTTPError as exc:
            if exc.code < 500 and exc.code != 429:
                return []
            if attempt < max_attempts - 1:
                time.sleep(2 ** attempt)
        except (urllib.error.URLError, TimeoutError, json.JSONDecodeError):
            if attempt < max_attempts - 1:
                time.sleep(2 ** attempt)
    return []


def build_extractor(
    providers: list[ModelExtractorConfig],
) -> Callable[[str, str], list[Memory]]:
    """Return a ModelExtractor that tries each provider in order.

    The first provider to return a non-empty list wins. If every provider
    fails, the callable returns [] — the heuristic path still runs.
    """
    def extractor(user_text: str, assistant_text: str) -> list[Memory]:
        for config in providers:
            result = _with_retry(config, user_text, assistant_text)
            if result:
                return result
        return []

    return extractor
