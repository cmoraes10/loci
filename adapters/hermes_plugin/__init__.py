"""Hermes adapter: the same core, plus the two hooks MCP cannot offer.

`pre_llm_call` injects the context block before the model sees the turn.
`post_llm_call` reads the finished exchange and extracts, in the background,
after the reply is already on its way. Nothing here sits on the path the
person waits on.
"""

from __future__ import annotations

import json
import os
import threading

from core import (
    Category,
    Importance,
    Memory,
    Source,
    Store,
    context_block,
    extract,
    should_persist,
)

_store = Store(os.environ.get("LOCI_DB", "~/.loci/memory.db"))


def _remember(args: dict, **kwargs) -> str:
    try:
        memory = Memory(
            key=f"{args['category']}_{abs(hash(args['content'])) % 10**10}",
            title=args["content"][:80],
            content=args["content"],
            category=Category(args.get("category", "preferences")),
            importance=Importance(args.get("importance", "medium")),
            source=Source.USER_MESSAGE,
        )
    except (KeyError, ValueError) as exc:
        # A typed error back to the model, not an exception. The loop recovers
        # and retries with better arguments instead of dying mid-turn.
        return json.dumps({"ok": False, "error": str(exc)})
    if not should_persist(memory):
        return json.dumps({"ok": True, "stored": False})
    return json.dumps({"ok": True, "key": _store.upsert(memory).key})


def _recall(args: dict, **kwargs) -> str:
    found = _store.search(args["query"]) if args.get("query") else _store.active()
    return json.dumps({"ok": True, "context": context_block(found)})


def _forget(args: dict, **kwargs) -> str:
    return json.dumps({"ok": True, "removed": _store.forget(args.get("key", ""))})


def _inject_context(payload: dict) -> dict:
    """Eager injection into the system instruction, capped and ranked."""
    block = context_block(_store.active())
    if block:
        payload["system"] = f"{payload.get('system', '')}\n\n{block}".strip()
    return payload


def _extract_async(payload: dict) -> dict:
    """Fire and forget. The reply has already gone out."""

    def work() -> None:
        try:
            for memory in extract(payload.get("user", ""), payload.get("assistant", "")):
                if should_persist(memory):
                    _store.upsert(memory)
        except Exception:  # noqa: BLE001 - never let memory break a turn
            pass

    threading.Thread(target=work, daemon=True).start()
    return payload


REMEMBER_SCHEMA = {
    "type": "object",
    "properties": {
        "content": {"type": "string", "maxLength": 220},
        "category": {"type": "string", "enum": [c.value for c in Category]},
        "importance": {"type": "string", "enum": [i.value for i in Importance]},
    },
    "required": ["content"],
}
RECALL_SCHEMA = {"type": "object", "properties": {"query": {"type": "string"}}}
FORGET_SCHEMA = {"type": "object", "properties": {"key": {"type": "string"}}, "required": ["key"]}


def register(ctx) -> None:
    ctx.register_tool(name="remember", toolset="mnemo", schema=REMEMBER_SCHEMA, handler=_remember)
    ctx.register_tool(name="recall", toolset="mnemo", schema=RECALL_SCHEMA, handler=_recall)
    ctx.register_tool(name="forget", toolset="mnemo", schema=FORGET_SCHEMA, handler=_forget)
    ctx.register_hook("pre_llm_call", _inject_context)
    ctx.register_hook("post_llm_call", _extract_async)
