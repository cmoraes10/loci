# loci

Typed long-term memory for AI agents. One core, two adapters.

The name comes from the method of loci, the ancient memory palace technique where every recollection gets a place and a structure. That is the thesis of this project.

An agent that forgets you every session is a search box with manners. This is the layer that fixes that: durable facts about a person, typed and ranked, extracted from the conversation rather than typed into a settings page, consolidated daily so the list does not rot.

Ported from the memory system running in production in
[Lima](https://getlima.app), rewritten host-agnostic.

## What it does

Eight categories (routine, study, preferences, finance, goals, relationships,
constraints, ephemeral), four importance tiers, a status lifecycle. A
preference and a deadline are not the same kind of thing and do not age
the same way.

Extraction runs on two paths. A model reads the exchange for meaning; regex
heuristics catch the phrasings you wrote them for. Merged with model precedence.
When the provider is down the regex path still runs.

A write filter keeps the store from becoming a mood diary. Passing mood is the
case it exists for. "Tired today" is true for hours and wrong for months.

The injection cap keeps context from growing without bound. Twelve facts reach
the model, ranked by importance and then by deadline proximity and recency.

Daily consolidation and decay keep the list honest. Exact duplicates merge;
guesses nobody confirmed expire on their own; ephemeral facts carry a 24-hour
TTL; completed goals expire after 90 days.

## The two adapters

|  | MCP server | Hermes plugin |
| --- | --- | --- |
| Runs in | Any MCP client | Hermes only |
| Tools | `remember`, `recall`, `forget` | the same three |
| Automatic extraction | no | yes |
| Context injection | on request | before every call |

MCP is request and response. Nothing calls a server when a turn ends, so there
is no moment for it to read the exchange on its own. Through MCP the agent has
to call `remember` on purpose. Automatic extraction needs a hook in the host,
and Hermes has one (`post_llm_call`). Same core underneath; the Hermes adapter
just has somewhere to stand.

## Install

```sh
pip install mowave-loci
pip install mowave-loci[mcp]
```

MCP client config (after `pip install mowave-loci[mcp]`):

```json
{
  "mcpServers": {
    "loci": {
      "command": "python",
      "args": ["-m", "loci.adapters.mcp_server.server"]
    }
  }
}
```

Hermes:

```sh
hermes plugins install <you>/loci
hermes plugins enable loci
```

The store is a SQLite file at `~/.loci/memory.db`, overridable with `LOCI_DB`.
No server to run, because a memory layer that needs one is not installable.

## Use it directly

```python
from loci import Store, extract, should_persist, context_block

store = Store()
for memory in extract("eu prefiro respostas curtas", assistant_text=reply):
    if should_persist(memory):
        store.upsert(memory)

system_prompt += "\n\n" + context_block(store.active())
```

## Where changes go

This repo owns the shape of a memory and its lifecycle. It does not own:

| Thing | Owner |
| --- | --- |
| Model calls, API keys, provider SDKs | your app (inject a `ModelExtractor`) |
| The agent loop, tool dispatch, sessions | the host (Hermes, OpenClaw, Claude Code) |
| Scheduling the daily jobs | the host's cron, s6 or systemd |
| Transport and auth | the adapter |

The core imports nothing but the standard library. If a change here would make
that untrue, it belongs in an adapter.

## Tests

```sh
pytest -q
```

92 tests against a real SQLite store on a temp file. Nothing of ours is
mocked: a memory layer whose tests pass against a fake store tells you nothing
about the one people run.

## License

MIT.
