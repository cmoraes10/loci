"""The jobs that keep the list from rotting. Run daily.

Without these a memory layer only grows: duplicates pile up, guesses made in
March are still asserted in September, and the cap starts dropping good facts
to make room for stale ones.
"""

from __future__ import annotations

from collections.abc import Callable
from datetime import date, timedelta

from .models import Category, Memory, Source, Status
from .store import Store

EPHEMERAL_TTL = timedelta(hours=24)
HYPOTHESIS_REVIEW = timedelta(days=30)


def _normalise(text: str) -> str:
    return " ".join(text.lower().split())


def consolidate(store: Store, merge_similar: Callable[[list[Memory]], list[Memory]] | None = None) -> int:
    """Exact dedupe always; model-assisted merge only if a merger is supplied.

    Exact dedupe is safe and free. Merging things that merely look alike is a
    judgement call, so it stays injectable and off by default.
    """
    seen: dict[tuple[str, str], Memory] = {}
    removed = 0
    for memory in store.active():
        fingerprint = (memory.category.value, _normalise(memory.content))
        if fingerprint in seen:
            kept = seen[fingerprint]
            store.upsert(kept.merge(memory))
            store.forget(memory.key)
            removed += 1
        else:
            seen[fingerprint] = memory

    if merge_similar:
        for category in Category:
            group = store.active(category)
            if len(group) > 1:
                for merged in merge_similar(group):
                    merged.source = Source.CONSOLIDATION
                    store.upsert(merged)
    return removed


def decay(store: Store, today: date | None = None) -> int:
    """Expire ephemeral facts past their TTL and hypotheses nobody confirmed."""
    today = today or date.today()
    expired = 0
    for memory in store.active():
        if memory.category is Category.EPHEMERAL:
            if memory.created_at.date() + EPHEMERAL_TTL < today:
                store.set_status(memory.key, Status.EXPIRED)
                expired += 1
        elif memory.status is Status.HYPOTHESIS and memory.review_after:
            if memory.review_after < today:
                store.set_status(memory.key, Status.EXPIRED)
                expired += 1
    return expired
