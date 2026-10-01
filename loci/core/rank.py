"""Which memories reach the model, and in what order.

The cap is the whole point. Everything remembered is not everything injected:
a context block that grows without bound quietly eats the budget the
conversation needs.
"""

from __future__ import annotations

import re
from datetime import date, timedelta

from .models import Category, Importance, Memory, Status

DEFAULT_CAP = 12
_URGENCY_WINDOW = timedelta(days=7)

#: A moment word next to a feeling word is a state, not a fact about someone.
_MOMENT = re.compile(r"\b(hoje|agora|ontem|today|right now|tonight)\b", re.IGNORECASE)
_FEELING = re.compile(
    r"\b(cansad|ansios|triste|feliz|animad|estressad|desanimad|irritad"
    r"|tired|anxious|sad|happy|excited|stressed|angry|bored)\w*",
    re.IGNORECASE,
)

# Within an importance tier: active facts first, then hypotheses, then completed.
# Expired memories are never included but the value is here for completeness.
_STATUS_TIER = {Status.ACTIVE: 0, Status.HYPOTHESIS: 1, Status.COMPLETED: 2, Status.EXPIRED: 3}

# Numbers with 4+ consecutive digits cover card numbers, account numbers, PINs,
# and most personally sensitive sequences — never worth storing.
_SENSITIVE_NUMBER = re.compile(r"\b\d{4,}\b")
_SECRETISH = re.compile(
    r"\b(password|senha|token|otp|cvv|secret|api[_\-]?key|access[_\-]?key)\b",
    re.IGNORECASE,
)
# Role-like labels followed by a colon are the classic prompt injection opener.
_ROLE_PREFIX = re.compile(r"\b(system|user|assistant|human|ai)\s*:", re.IGNORECASE)


def is_passing_state(text: str) -> bool:
    """A feeling pinned to a moment. True for hours, wrong for months."""
    return bool(_MOMENT.search(text) and _FEELING.search(text))


def is_safe(memory: Memory) -> bool:
    """Returns False when the memory looks like a secret or a sensitive number.

    Prevents the store from accidentally becoming a credential vault. Called
    before writing — a safe() check upstream means the model never sees this
    content injected back into a system instruction.
    """
    for field_value in (memory.content, memory.title):
        if _SECRETISH.search(field_value):
            return False
        if _SENSITIVE_NUMBER.search(field_value):
            return False
    return True


def _neutralize_for_prompt(text: str) -> str:
    """Strip delimiters that a stored fact could use to inject instructions.

    Called on every line of context_block — the memory content was sanitized
    when written, but an extra pass here keeps the injection surface minimal
    regardless of how the memory was created.
    """
    text = text.replace("\r", " ").replace("\n", " ")
    text = text.replace("---", "—")
    text = text.replace("</", "< /")
    text = text.replace("<|", "< |")
    text = text.replace("[INST]", "[ INST]").replace("[/INST]", "[ /INST]")
    text = text.replace("#", "hash ")
    text = text.replace("```", "'''")  # triple-backtick opens a code block in some Markdown parsers
    text = _ROLE_PREFIX.sub(lambda m: m.group(1) + " ", text)
    return text.strip()


def _urgency_boost(m: Memory, today: date) -> int:
    """0 if this memory has a deadline arriving within the urgency window, 1 otherwise.

    A deadline due in three days should surface before a same-importance fact
    with no time pressure — 0 sorts before 1, so the urgent item wins the tie.
    Hypotheses are included: an unconfirmed allergy due tomorrow is still urgent.
    """
    if (
        m.review_after is not None
        and m.status in (Status.ACTIVE, Status.HYPOTHESIS)
        and m.review_after <= today + _URGENCY_WINDOW
    ):
        return 0
    return 1


def _deadline_label(m: Memory, today: date) -> str:
    if not m.review_after:
        return ""
    delta = (m.review_after - today).days
    if delta < 0:
        days = abs(delta)
        return f" (overdue by {days} day{'s' if days != 1 else ''})"
    if delta == 0:
        return " (due today)"
    if delta == 1:
        return " (due tomorrow)"
    if delta <= _URGENCY_WINDOW.days:
        return f" (due in {delta} days)"
    return ""


def select(
    memories: list[Memory],
    cap: int = DEFAULT_CAP,
    today: date | None = None,
) -> list[Memory]:
    """Critical and high fill the first slots; medium and low fill what is left.

    Within each importance tier: memories with an approaching deadline surface
    first, then active facts, then hypotheses, then completed. Ties break on
    recency so a fact restated last week outranks the same-weight fact from March.
    """
    today = today or date.today()
    live = [m for m in memories if m.status is not Status.EXPIRED]
    live.sort(key=lambda m: (
        m.importance.rank,
        _urgency_boost(m, today),
        _STATUS_TIER.get(m.status, 3),
        -m.updated_at.timestamp(),
    ))
    return live[:cap]


def context_block(
    memories: list[Memory],
    cap: int = DEFAULT_CAP,
    today: date | None = None,
) -> str:
    """Render for injection into a system instruction.

    Eager injection, not a tool the model has to remember to call. A model that
    must ask for context mostly does not ask.
    """
    today = today or date.today()
    chosen = select(memories, cap, today)
    if not chosen:
        return ""
    lines = ["What you know about this person:"]
    for m in chosen:
        if m.status is Status.HYPOTHESIS:
            # Deadline label is dropped for hypotheses — showing (unconfirmed) keeps
            # the reader aware that the fact is not yet verified, which matters more.
            mark = " (unconfirmed)"
        elif m.status is Status.COMPLETED:
            mark = " (completed)"
        else:
            mark = _deadline_label(m, today)
        lines.append(f"- [{m.category.value}] {_neutralize_for_prompt(m.content)}{mark}")
    return "\n".join(lines)


def review_block(memories: list[Memory], today: date | None = None) -> str:
    """Render memories that are due for a check-in.

    The coaching surface: facts the person attached a review date to, now
    arriving. An agent that surfaces these proactively is doing something a
    search box cannot.
    """
    today = today or date.today()
    due = [
        m for m in memories
        if m.review_after is not None
        and m.review_after <= today
        and m.status in (Status.ACTIVE, Status.HYPOTHESIS)
    ]
    if not due:
        return ""
    due.sort(key=lambda m: (m.review_after, m.importance.rank))
    lines = ["Things to check in on:"]
    for m in due:
        if m.status is Status.HYPOTHESIS:
            mark = " (unconfirmed)"
        else:
            mark = _deadline_label(m, today)
        lines.append(f"- [{m.category.value}] {_neutralize_for_prompt(m.content)}{mark}")
    return "\n".join(lines)


def should_persist(memory: Memory) -> bool:
    """The write filter. Most of what a turn produces is not worth keeping.

    Security gate runs first: secrets and prompt-injection patterns are never
    stored regardless of importance. After that: passing mood is the main
    target — "tired today" is true for hours and wrong for months. Critical and
    high importance still pass the mood filter because the person stating
    something matters more than the heuristic disagreeing.
    """
    if not is_safe(memory):
        return False
    if memory.importance in (Importance.CRITICAL, Importance.HIGH):
        return True
    if memory.category is Category.EPHEMERAL:
        return True
    if len(memory.content) < 12:
        return False
    return not is_passing_state(memory.content)
