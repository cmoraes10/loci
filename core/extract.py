"""Turning a finished exchange into durable facts.

Two paths run over the same text and are merged, because each fails where the
other holds: a model reads meaning but can be down, rate limited or vague;
regex never misses the phrasings you wrote it for and never costs a token.
The model wins on conflict, since it is the one that read the sentence.
"""

from __future__ import annotations

import re
from collections.abc import Callable

from .models import Category, Importance, Memory, Source, Status

#: A model extractor takes the exchange and returns candidate memories.
#: Injected, never imported: the core holds no API key and no provider SDK.
ModelExtractor = Callable[[str, str], list[Memory]]

HEURISTICS: list[tuple[str, Category, Importance]] = [
    (r"\b(?:eu )?(?:prefiro|gosto de|odeio|detesto)\s+(.{4,80})", Category.PREFERENCES, Importance.MEDIUM),
    (r"\b(?:sempre|nunca)\s+(.{4,80})", Category.ROUTINE, Importance.MEDIUM),
    (r"\b(?:decidi|escolhi|vamos de)\s+(.{4,80})", Category.GOALS, Importance.HIGH),
    (r"\b(?:n[ãa]o posso|estou impedido de|tenho que)\s+(.{4,80})", Category.CONSTRAINTS, Importance.HIGH),
    (r"\b(?:meu prazo|deadline|entrego)\s+(.{4,80})", Category.GOALS, Importance.HIGH),
]


def _slug(text: str, limit: int = 48) -> str:
    cleaned = re.sub(r"[^a-z0-9]+", "_", text.lower()).strip("_")
    return cleaned[:limit] or "fact"


def heuristic_extract(user_text: str) -> list[Memory]:
    """Cheap, deterministic, offline. Runs even when the provider is down."""
    found: list[Memory] = []
    for pattern, category, importance in HEURISTICS:
        for match in re.finditer(pattern, user_text, flags=re.IGNORECASE):
            phrase = match.group(0).strip().rstrip(".,;")
            found.append(
                Memory(
                    key=f"{category.value}_{_slug(match.group(1))}",
                    title=phrase[:80],
                    content=phrase,
                    category=category,
                    importance=importance,
                    source=Source.USER_MESSAGE,
                    status=Status.HYPOTHESIS,
                    confidence=0.6,
                )
            )
    return found


def merge_paths(model_side: list[Memory], heuristic_side: list[Memory]) -> list[Memory]:
    """Model precedence on a shared key; heuristics fill what it missed."""
    merged: dict[str, Memory] = {m.key: m for m in heuristic_side}
    for memory in model_side:
        merged[memory.key] = memory
    return list(merged.values())


def extract(
    user_text: str,
    assistant_text: str = "",
    model_extractor: ModelExtractor | None = None,
) -> list[Memory]:
    """Run both paths. Never raises: a failed extraction must not fail the turn.

    Called after the reply is already on its way to the person, so nothing here
    is on the path they wait on.
    """
    model_side: list[Memory] = []
    if model_extractor:
        try:
            model_side = model_extractor(user_text, assistant_text)
        except Exception:  # noqa: BLE001 - extraction is best effort by design
            model_side = []
    return merge_paths(model_side, heuristic_extract(user_text))
