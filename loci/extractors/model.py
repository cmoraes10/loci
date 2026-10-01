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
from datetime import date
from typing import Any

from loci.core.models import Category, Importance, Memory, Source, Status

_SYSTEM = """\
You are a memory extraction assistant. Given a conversation exchange, identify
facts worth remembering long-term about the user.

Respond with valid JSON only, no markdown fences, using this exact shape:
{"facts": [{"key": "unique_slug_under_48_chars", "title": "Short label (max 80 chars)", "content": "Full statement as a sentence (max 220 chars)", "category": "preferences", "importance": "medium", "review_after": null}]}

review_after: ISO date string (YYYY-MM-DD) if the fact has a deadline or should be revisited, otherwise null.

Valid category values: routine, study, preferences, finance, goals, relationships, constraints, ephemeral
Valid importance values: critical, high, medium, low

Importance levels:
  critical — health constraints (allergies, medical conditions, dietary restrictions), safety-critical facts
  high — active goals, financial rules, relationships, declared routines; every RELATIONSHIPS or CONSTRAINTS item is at least high
  medium — general preferences, context details, supporting facts
  low — background details unlikely to affect future responses

Category-specific guidance:

finance — store the TYPE of fact, never exact monetary amounts:
  monthly_income: user has regular income (salary, freelance, pension)
  recurring_expense: fixed monthly cost (rent, phone plan, subscription)
  budget_preference: declared spending limit, savings target, or budget rule

relationships — use stable gender-neutral keys so updates overwrite prior versions:
  partner_romantic: romantic partner (girlfriend or boyfriend)
  spouse: husband or wife
  child_mentioned: son or daughter
  parent_mentioned: mother or father
  friend_close: named close friend explicitly mentioned
  colleague_key: named important colleague or manager

constraints — health facts are always critical importance:
  health_allergy: declared food or medication allergy
  health_condition: chronic condition (diabetes, hypertension, anxiety, depression)
  dietary_restriction: vegan, vegetarian, gluten-free, lactose-free

preferences:
  work_context: job, profession, work modality (remote/on-site)
  hobby_interest: declared hobbies or leisure activities
  communication_style: how this user writes — tone, slang, casing, abbreviations.
    Extract only when you observe three or more clear, consistent markers in the same turn.

goals:
  active_goal: immediate goal the user declared
  career_aspiration: desired future job, career change, or entrepreneurship

ephemeral — passing states only, always low importance, TTL 24h.
  If the same state repeats for 5+ consecutive days, reclassify as routine.

General rules:
  Return at most 4 facts. If nothing durable exists, return {"facts": []}.
  Skip passing mood, small talk, one-off logistics, and temporary requests.
  If the user asks to forget or delete something, do NOT extract that fact.
  Never store passwords, tokens, access codes, full card or account numbers, or sensitive PII.
  For the same category and key, write updated content — do not create a parallel entry.
"""

_VALID_CATEGORIES = {c.value for c in Category}
_VALID_IMPORTANCE = {i.value for i in Importance}

# Minimum importance floors by category and by special key.
# RELATIONSHIPS and CONSTRAINTS carry facts people act on — dropping them to
# medium/low means the context cap can silently push them out of the block.
_CATEGORY_FLOOR: dict[str, Importance] = {
    Category.RELATIONSHIPS.value: Importance.HIGH,
    Category.CONSTRAINTS.value: Importance.HIGH,
}
# communication_style guides every turn's tone — it must reach the context block.
_KEY_FLOOR: dict[str, Importance] = {
    "communication_style": Importance.HIGH,
}


def _enforce_min_importance(category: Category, importance: Importance, key: str) -> Importance:
    floor = _CATEGORY_FLOOR.get(category.value)
    if floor and importance.rank > floor.rank:
        importance = floor
    key_floor = _KEY_FLOOR.get(key)
    if key_floor and importance.rank > key_floor.rank:
        importance = key_floor
    return importance


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
            cat = Category(f["category"] if f.get("category") in _VALID_CATEGORIES else "preferences")
            imp = Importance(f["importance"] if f.get("importance") in _VALID_IMPORTANCE else "medium")
            key = str(f.get("key", ""))[:48] or "model_fact"
            imp = _enforce_min_importance(cat, imp, key)
            parsed_date: date | None = None
            raw_date = f.get("review_after")
            if isinstance(raw_date, str):
                try:
                    parsed_date = date.fromisoformat(raw_date)
                except ValueError:
                    pass
            memories.append(
                Memory(
                    key=key,
                    title=str(f.get("title", ""))[:80],
                    content=str(f.get("content", ""))[:220],
                    category=cat,
                    importance=imp,
                    source=Source.ASSISTANT_INFERENCE,
                    status=Status.HYPOTHESIS,
                    confidence=0.8,
                    review_after=parsed_date,
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
    """Exponential backoff on transient failures (429, 5xx). Fail fast on other 4xx.

    max_attempts=0 still runs once — extraction is best-effort, not optional.
    """
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
