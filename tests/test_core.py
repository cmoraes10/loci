"""Tests run against the real store, on a temp file. No mocks of our own code."""

from datetime import date, timedelta

import pytest

from loci import (
    Category,
    Importance,
    Memory,
    Source,
    Status,
    Store,
    consolidate,
    context_block,
    decay,
    heuristic_extract,
    is_passing_state,
    merge_paths,
    select,
    should_persist,
)


@pytest.fixture()
def store(tmp_path):
    return Store(tmp_path / "test.db")


def make(key: str, importance=Importance.MEDIUM, **kw) -> Memory:
    return Memory(
        key=key,
        title=kw.pop("title", key),
        content=kw.pop("content", f"conteudo de {key} com tamanho suficiente"),
        category=kw.pop("category", Category.PREFERENCES),
        importance=importance,
        source=kw.pop("source", Source.USER_MESSAGE),
        **kw,
    )


def test_content_is_clamped():
    assert len(make("k", content="x" * 500).content) == 220


def test_upsert_merges_instead_of_duplicating(store):
    store.upsert(make("same_key"))
    merged = store.upsert(make("same_key", content="a mesma coisa dita outra vez"))
    assert merged.evidence_count == 2
    assert len(store.active()) == 1


def test_merge_keeps_the_stronger_importance(store):
    store.upsert(make("k", importance=Importance.LOW))
    merged = store.upsert(make("k", importance=Importance.CRITICAL))
    assert merged.importance is Importance.CRITICAL


def test_select_caps_and_orders_by_importance():
    memories = [make(f"low{i}", Importance.LOW) for i in range(20)]
    memories += [make("crit", Importance.CRITICAL)]
    chosen = select(memories, cap=5)
    assert len(chosen) == 5
    assert chosen[0].key == "crit"


def test_expired_never_reaches_the_model():
    live = make("live")
    gone = make("gone", status=Status.EXPIRED)
    assert "gone" not in context_block([live, gone])


def test_write_filter_drops_short_low_value_facts():
    assert not should_persist(make("k", Importance.LOW, content="ok"))
    assert should_persist(make("k", Importance.CRITICAL, content="ok"))


def test_heuristics_find_a_stated_preference():
    found = heuristic_extract("eu prefiro respostas curtas e diretas")
    assert found and found[0].category is Category.PREFERENCES
    assert found[0].status is Status.HYPOTHESIS


def test_model_wins_over_heuristic_on_the_same_key():
    heuristic = make("shared", content="versao da regex")
    model = make("shared", content="versao do modelo")
    merged = merge_paths([model], [heuristic])
    assert len(merged) == 1
    assert merged[0].content == "versao do modelo"


def test_consolidate_removes_exact_duplicates(store):
    store.upsert(make("a", content="mesmo conteudo exatamente igual aqui"))
    store.upsert(make("b", content="Mesmo   Conteudo exatamente igual aqui"))
    assert consolidate(store) == 1
    assert len(store.active()) == 1


def test_decay_expires_an_unconfirmed_hypothesis(store):
    store.upsert(
        make("guess", status=Status.HYPOTHESIS, review_after=date.today() - timedelta(days=1))
    )
    assert decay(store) == 1
    assert store.get("guess").status is Status.EXPIRED


def test_decay_leaves_a_confirmed_fact_alone(store):
    store.upsert(make("solid", status=Status.ACTIVE))
    assert decay(store) == 0


def test_passing_mood_is_not_stored():
    """The case the write filter exists for."""
    assert not should_persist(make("k", Importance.MEDIUM, content="estou cansado hoje"))
    assert not should_persist(make("k", Importance.LOW, content="feeling tired today"))


def test_a_stated_fact_that_merely_mentions_today_is_kept():
    assert should_persist(make("k", Importance.MEDIUM, content="hoje decidi migrar para Postgres"))


def test_the_person_outranks_the_heuristic():
    """Critical and high pass even when they look like passing mood."""
    assert should_persist(make("k", Importance.HIGH, content="estou cansado hoje"))
