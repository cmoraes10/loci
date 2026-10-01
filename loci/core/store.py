"""Persistence. SQLite because a memory layer that needs a server is not installable."""

from __future__ import annotations

import sqlite3
from datetime import date, datetime, timezone
from pathlib import Path

from .models import Category, Importance, Memory, Source, Status

SCHEMA = """
CREATE TABLE IF NOT EXISTS memories (
    key            TEXT PRIMARY KEY,
    title          TEXT NOT NULL,
    content        TEXT NOT NULL,
    category       TEXT NOT NULL,
    importance     TEXT NOT NULL,
    source         TEXT NOT NULL,
    status         TEXT NOT NULL,
    confidence     REAL NOT NULL,
    evidence_count INTEGER NOT NULL,
    review_after   TEXT,
    created_at     TEXT NOT NULL,
    updated_at     TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_status_importance ON memories (status, importance);
CREATE INDEX IF NOT EXISTS idx_category ON memories (category);
CREATE INDEX IF NOT EXISTS idx_review_after ON memories (review_after);
"""


def _parse_dt(value: str) -> datetime:
    """Read an ISO datetime and guarantee it is UTC-aware.

    Rows written before the +00:00 suffix was enforced would be naive strings.
    Adding the timezone here rather than at write time avoids a migration.
    """
    dt = datetime.fromisoformat(value)
    if dt.tzinfo is None:
        return dt.replace(tzinfo=timezone.utc)
    return dt


def _row_to_memory(row: sqlite3.Row) -> Memory:
    return Memory(
        key=row["key"],
        title=row["title"],
        content=row["content"],
        category=Category(row["category"]),
        importance=Importance(row["importance"]),
        source=Source(row["source"]),
        status=Status(row["status"]),
        confidence=row["confidence"],
        evidence_count=row["evidence_count"],
        review_after=date.fromisoformat(row["review_after"]) if row["review_after"] else None,
        created_at=_parse_dt(row["created_at"]),
        updated_at=_parse_dt(row["updated_at"]),
    )


class Store:
    def __init__(self, path: str | Path = "~/.loci/memory.db") -> None:
        self.path = Path(path).expanduser()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.db = sqlite3.connect(self.path, check_same_thread=False)
        self.db.row_factory = sqlite3.Row
        self.db.executescript(SCHEMA)

    def close(self) -> None:
        self.db.close()

    def __enter__(self) -> "Store":
        return self

    def __exit__(self, *_) -> None:
        self.close()

    def __del__(self) -> None:
        try:
            self.db.close()
        except Exception:  # noqa: BLE001
            pass

    def upsert(self, memory: Memory) -> Memory:
        """Write, merging into an existing key rather than duplicating it."""
        existing = self.get(memory.key)
        final = existing.merge(memory) if existing else memory
        self.db.execute(
            """INSERT INTO memories VALUES (:key,:title,:content,:category,:importance,
                   :source,:status,:confidence,:evidence_count,:review_after,
                   :created_at,:updated_at)
               ON CONFLICT(key) DO UPDATE SET
                   title=excluded.title, content=excluded.content,
                   category=excluded.category, importance=excluded.importance,
                   source=excluded.source, status=excluded.status,
                   confidence=excluded.confidence, evidence_count=excluded.evidence_count,
                   review_after=excluded.review_after, updated_at=excluded.updated_at""",
            {
                "key": final.key,
                "title": final.title,
                "content": final.content,
                "category": final.category.value,
                "importance": final.importance.value,
                "source": final.source.value,
                "status": final.status.value,
                "confidence": final.confidence,
                "evidence_count": final.evidence_count,
                "review_after": final.review_after.isoformat() if final.review_after else None,
                "created_at": final.created_at.isoformat(),
                "updated_at": final.updated_at.isoformat(),
            },
        )
        self.db.commit()
        return final

    def get(self, key: str) -> Memory | None:
        row = self.db.execute("SELECT * FROM memories WHERE key = ?", (key,)).fetchone()
        return _row_to_memory(row) if row else None

    def active(self, category: Category | None = None) -> list[Memory]:
        """All non-expired memories, including hypotheses and completed ones."""
        sql = "SELECT * FROM memories WHERE status != 'expired'"
        args: list[str] = []
        if category:
            sql += " AND category = ?"
            args.append(category.value)
        return [_row_to_memory(r) for r in self.db.execute(sql, args)]

    def due_for_review(self, today: date | None = None) -> list[Memory]:
        """Memories that asked to be checked — deadlines that arrived, commitments to revisit.

        These are the facts that hold the person accountable. An agent that surfaces them
        proactively is doing something a search box cannot.
        """
        today = today or date.today()
        rows = self.db.execute(
            """SELECT * FROM memories
               WHERE status IN ('active', 'hypothesis')
                 AND review_after IS NOT NULL
                 AND review_after <= ?""",
            (today.isoformat(),),
        )
        return [_row_to_memory(r) for r in rows]

    def search(self, text: str, limit: int = 20) -> list[Memory]:
        """Substring match. Deliberately not embeddings.

        A few hundred short facts fit in one pass, and a vector store turns an
        installable library into a service someone has to run.
        """
        escaped = text.lower().replace("!", "!!").replace("%", "!%").replace("_", "!_")
        like = f"%{escaped}%"
        rows = self.db.execute(
            """SELECT * FROM memories
               WHERE status != 'expired'
                 AND (lower(title) LIKE ? ESCAPE '!' OR lower(content) LIKE ? ESCAPE '!')
               LIMIT ?""",
            (like, like, limit),
        )
        return [_row_to_memory(r) for r in rows]

    def forget(self, key: str) -> bool:
        cur = self.db.execute("DELETE FROM memories WHERE key = ?", (key,))
        self.db.commit()
        return cur.rowcount > 0

    def set_status(self, key: str, status: Status) -> None:
        self.db.execute("UPDATE memories SET status = ? WHERE key = ?", (status.value, key))
        self.db.commit()
