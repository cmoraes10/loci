"""Which memories reach the model, and in what order.

The cap is the whole point. Everything remembered is not everything injected:
a context block that grows without bound quietly eats the budget the
conversation needs.
"""

from __future__ import annotations

import re

from .models import Category, Importance, Memory, Status

DEFAULT_CAP = 12

#: A moment word next to a feeling word is a state, not a fact about someone.
_MOMENT = re.compile(r"\b(hoje|agora|ontem|today|right now|tonight)\b", re.IGNORECASE)
_FEELING = re.compile(
    r"\b(cansad|ansios|triste|feliz|animad|estressad|desanimad|irritad"
    r"|tired|anxious|sad|happy|excited|stressed|angry|bored)\w*",
    re.IGNORECASE,
)


def is_passing_state(text: str) -> bool:
    """A feeling pinned to a moment. True for hours, wrong for months."""
    return bool(_MOMENT.search(text) and _FEELING.search(text))


def select(memories: list[Memory], cap: int = DEFAULT_CAP) -> list[Memory]:
    """Critical and high fill the first slots; medium and low fill what is left.

    Ties break on recency, so a fact restated last week outranks the same-weight
    fact from March.
    """
    live = [m for m in memories if m.status is not Status.EXPIRED]
    live.sort(key=lambda m: (m.importance.rank, -m.updated_at.timestamp()))
    return live[:cap]


def context_block(memories: list[Memory], cap: int = DEFAULT_CAP) -> str:
    """Render for injection into a system instruction.

    Eager injection, not a tool the model has to remember to call. A model that
    must ask for context mostly does not ask.
    """
    chosen = select(memories, cap)
    if not chosen:
        return ""
    lines = ["What you know about this person:"]
    for m in chosen:
        mark = " (unconfirmed)" if m.status is Status.HYPOTHESIS else ""
        lines.append(f"- [{m.category.value}] {m.content}{mark}")
    return "\n".join(lines)


def should_persist(memory: Memory) -> bool:
    """The write filter. Most of what a turn produces is not worth keeping.

    Passing mood is the case this exists for: "tired today" is true for hours
    and wrong for months, and a memory layer that keeps it tells the model
    something false every day after. Critical and high still pass, because a
    person saying something matters more than this heuristic guessing it does
    not.
    """
    if memory.importance in (Importance.CRITICAL, Importance.HIGH):
        return True
    if memory.category is Category.EPHEMERAL:
        return True
    if len(memory.content) < 12:
        return False
    return not is_passing_state(memory.content)
