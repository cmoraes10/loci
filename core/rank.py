"""Which memories reach the model, and in what order.

The cap is the whole point. Everything remembered is not everything injected:
a context block that grows without bound quietly eats the budget the
conversation needs.
"""

from __future__ import annotations

from .models import Importance, Memory, Status

DEFAULT_CAP = 12


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
    something false every day after.
    """
    if memory.importance in (Importance.CRITICAL, Importance.HIGH):
        return True
    if memory.category.value == "ephemeral":
        return True
    if len(memory.content) < 12:
        return False
    return True
