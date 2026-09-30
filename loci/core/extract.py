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

# Each tuple is (pattern, category, importance). Group 1 is the captured phrase
# used to generate a stable key; group 0 is stored as the memory content.
#
# Ordering matters: more specific patterns first, so a sentence like
# "decidi que meu prazo é sexta" hits GOALS via "decidi", not via "prazo".
HEURISTICS: list[tuple[str, Category, Importance]] = [

    # Decisions — the person committed to something.
    (
        r"\b(?:decidi|escolhi|vou de|optei por|resolvi|I decided|I chose|I'm going with|I opted for)\s+(.{4,80})",
        Category.GOALS, Importance.HIGH,
    ),

    # Hard constraints — what the person cannot or must do.
    (
        r"\b(?:n[ãa]o posso|n[ãa]o consigo|estou impedido de|tenho que obrigatoriamente"
        r"|I can't|I cannot|I'm not allowed to|I must|I'm required to)\s+(.{4,80})",
        Category.CONSTRAINTS, Importance.HIGH,
    ),

    # Soft constraints — what the person needs or is obligated to do.
    (
        r"\b(?:tenho que|preciso|sou obrigado a|I need to|I have to|I'm obligated to)\s+(.{4,80})",
        Category.CONSTRAINTS, Importance.MEDIUM,
    ),

    # Deadlines and due dates.
    (
        r"\b(?:meu prazo|minha entrega|deadline|due by|due on|precisa estar pronto|entrego)\s+(.{4,80})",
        Category.GOALS, Importance.HIGH,
    ),

    # Goals and intentions that are not yet decisions.
    (
        r"\b(?:meu objetivo|minha meta|quero|planejo|pretendo|my goal is|my target is"
        r"|I want to|I plan to|I'm planning to|I intend to)\s+(.{4,80})",
        Category.GOALS, Importance.MEDIUM,
    ),

    # Preferences — likes and dislikes.
    (
        r"\b(?:eu )?(?:prefiro|gosto de|adoro|amo|odeio|detesto|n[ãa]o gosto de"
        r"|I prefer|I like|I love|I hate|I dislike|I can't stand)\s+(.{4,80})",
        Category.PREFERENCES, Importance.MEDIUM,
    ),

    # Routines — habitual actions.
    (
        r"\b(?:sempre|nunca|todo dia|toda semana|geralmente|costumo|normalmente"
        r"|I always|I never|every day|every week|I usually|I typically|I regularly)\s+(.{4,80})",
        Category.ROUTINE, Importance.MEDIUM,
    ),

    # Finance — income, budget, spending.
    (
        r"\b(?:ganho|gasto|economizo|meu or[çc]amento|meu sal[áa]rio|minha renda"
        r"|I earn|I spend|I save|my budget|my salary|my income)\s+(.{4,80})",
        Category.FINANCE, Importance.HIGH,
    ),

    # Study — active learning or courses.
    (
        r"\b(?:estou estudando|estou aprendendo|fa[çc]o curso de|estou fazendo|matriculei"
        r"|I'm studying|I'm learning|I'm taking a course|I enrolled in)\s+(.{4,80})",
        Category.STUDY, Importance.MEDIUM,
    ),

    # Relationships — named people in the person's life.
    (
        r"\b(?:meu pai|minha m[ãa]e|meu filho|minha filha|meu irm[ãa]o|minha irm[ãa]"
        r"|meu chefe|minha chefe|meu namorado|minha namorada|meu marido|minha esposa"
        r"|my father|my mother|my son|my daughter|my brother|my sister"
        r"|my boss|my partner|my husband|my wife|my colleague)\s+(.{4,60})",
        Category.RELATIONSHIPS, Importance.MEDIUM,
    ),

    # Health — allergies. CRITICAL because every meal and medication recommendation
    # must account for this; a false negative here has real consequences.
    (
        r"\b(?:sou al[eé]rgic[oa]|tenho alergia|alergia a"
        r"|I'?m allergic|I have an? allergy|allergic to)\s+(.{3,80})",
        Category.CONSTRAINTS, Importance.CRITICAL,
    ),

    # Health — chronic conditions. CRITICAL for the same reason: permanent context
    # the model must never ignore when advising on food, exercise, or medication.
    (
        r"\b(tenho diabetes|sou diab[eé]tic[oa]|tenho press[ãa]o alta|tenho hipertens[ãa]o"
        r"|tenho ansiedade|tenho depress[ãa]o|sou cel[íi]ac[oa]|tenho intoler[aâ]ncia"
        r"|I have diabetes|I'?m diabetic|I have high blood pressure|I have hypertension"
        r"|I have anxiety|I have depression|I'?m celiac|I have (?:a )?(?:food |chronic )?intolerance)\b",
        Category.CONSTRAINTS, Importance.CRITICAL,
    ),

    # Health — dietary restrictions. Same CRITICAL weight as allergies in practice.
    (
        r"\b(n[ãa]o como carne|sou vegano|sou vegetarian[oa]|sem lactose|sem gl[úu]ten"
        r"|I don'?t eat meat|I'?m vegan|I'?m vegetarian|lactose[- ]?free|gluten[- ]?free)\b",
        Category.CONSTRAINTS, Importance.CRITICAL,
    ),
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
