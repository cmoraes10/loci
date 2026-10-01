"""The shape of a memory. Nothing here knows about a host, a model or a database."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, datetime, timezone
from enum import Enum


class Category(str, Enum):
    ROUTINE = "routine"
    STUDY = "study"
    PREFERENCES = "preferences"
    FINANCE = "finance"
    GOALS = "goals"
    RELATIONSHIPS = "relationships"
    CONSTRAINTS = "constraints"
    EPHEMERAL = "ephemeral"


class Importance(str, Enum):
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"

    @property
    def rank(self) -> int:
        """Lower sorts first. Ranking reads this, nothing else."""
        return {"critical": 0, "high": 1, "medium": 2, "low": 3}[self.value]


class Status(str, Enum):
    ACTIVE = "active"
    HYPOTHESIS = "hypothesis"
    COMPLETED = "completed"  # a goal or task was accomplished — semantically different from expired
    EXPIRED = "expired"      # ran out of time, was never confirmed, or became false


class Source(str, Enum):
    USER_MESSAGE = "user_message"
    ASSISTANT_INFERENCE = "assistant_inference"
    TOOL_RESULT = "tool_result"
    MANUAL = "manual"
    CONSOLIDATION = "consolidation"


TITLE_MAX = 80
CONTENT_MAX = 220


def _now() -> datetime:
    return datetime.now(timezone.utc)


@dataclass
class Memory:
    """One durable fact.

    `key` is the stable upsert handle: two extractions that mean the same thing
    carry the same key and the second updates the first instead of piling up.
    """

    key: str
    title: str
    content: str
    category: Category
    importance: Importance
    source: Source
    status: Status = Status.ACTIVE
    confidence: float = 1.0
    evidence_count: int = 1
    review_after: date | None = None
    created_at: datetime = field(default_factory=_now)
    updated_at: datetime = field(default_factory=_now)

    def __post_init__(self) -> None:
        self.title = self.title.strip()[:TITLE_MAX]
        self.content = self.content.strip()[:CONTENT_MAX]

    def merge(self, other: "Memory") -> "Memory":
        """Same key seen again. Keep the newer wording, raise the evidence.

        Status follows a simple promotion rule: COMPLETED wins over ACTIVE
        (the person said they finished it), but HYPOTHESIS cannot demote
        something already confirmed — an unconfirmed extraction does not
        override a known fact.
        """
        if other.status is Status.COMPLETED:
            merged_status = Status.COMPLETED
        elif other.status is Status.ACTIVE and self.status is not Status.COMPLETED:
            merged_status = Status.ACTIVE
        else:
            merged_status = self.status

        return Memory(
            key=self.key,
            title=other.title or self.title,
            content=other.content or self.content,
            category=other.category,
            importance=min(self.importance, other.importance, key=lambda i: i.rank),  # min rank = highest importance (CRITICAL=0)
            source=other.source,
            status=merged_status,
            confidence=max(self.confidence, other.confidence),
            evidence_count=self.evidence_count + 1,
            # `or` keeps self's deadline when other passes None — a deadline
            # cannot be cleared through a merge, only by explicit set_status.
            review_after=other.review_after or self.review_after,
            created_at=self.created_at,
            updated_at=_now(),
        )
