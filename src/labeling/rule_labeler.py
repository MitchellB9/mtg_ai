from __future__ import annotations

from dataclasses import dataclass
from typing import Any
import re

KEYWORD_TERMS = [
    "flying",
    "trample",
    "vigilance",
    "haste",
    "lifelink",
    "deathtouch",
    "menace",
    "reach",
    "first strike",
    "double strike",
    "hexproof",
    "indestructible",
    "defender",
    "flash",
    "ward",
    "prowess",
]


NON_FRENCH_VANILLA_SIGNALS = [
    "when ",
    "whenever ",
    "at the beginning",
    ":",
    "draw",
    "create",
    "destroy",
    "exile target",
    "return target",
    "search your library",
]


@dataclass(frozen=True)
class LabelMatch:
    label: str
    confidence: float
    source: str
    reason: str


def _text(row: dict[str, Any]) -> str:
    return str(row.get("oracle_text_norm") or row.get("oracle_text") or "").lower()


def _types(row: dict[str, Any], *keys: str) -> list[str]:
    """
    Input:
        Card row and one or more possible type-column names.

    Logic:
        Supports both old and new type-column naming styles.

    Output:
        Lowercase type values.
    """
    for key in keys:
        value = row.get(key, [])
        if isinstance(value, list):
            return [str(v).lower() for v in value]

    return []


def _has(text: str, pattern: str) -> bool:
    return re.search(pattern, text, flags=re.IGNORECASE | re.MULTILINE) is not None


def _add(
    labels: list[LabelMatch],
    label: str,
    confidence: float,
    reason: str,
    source: str = "rule",
) -> None:
    labels.append(LabelMatch(label, confidence, source, reason))


def label_card(row: dict[str, Any]) -> list[LabelMatch]:
    text = _text(row)

    basic_types = _types(row, "basic_types", "types")
    sub_types = _types(row, "sub_types", "subtypes")

    labels: list[LabelMatch] = []

    is_creature = "creature" in basic_types
    is_artifact = "artifact" in basic_types

    # -------------------------
    # Creature simplicity
    # -------------------------
    if is_creature and not text.strip():
        _add(labels, "vanilla_creature", 0.95, "Creature with no oracle text.")

    if is_creature and text.strip():
        if any(k in text for k in KEYWORD_TERMS) and not any(
            s in text for s in NON_FRENCH_VANILLA_SIGNALS
        ):
            _add(
                labels,
                "french_vanilla_creature",
                0.70,
                "Creature text appears mostly keyword-based.",
            )

    # -------------------------
    # Mana / ramp
    # -------------------------
    if _has(text, r"search your library for .* land") or _has(
        text, r"put .* land .* onto the battlefield"
    ):
        _add(
            labels,
            "land_ramp",
            0.90,
            "Searches for or puts lands onto the battlefield.",
        )

    if _has(text, r"\badd\s+\{[^}]+\}") or _has(
        text, r"\badd (one|two|three|x|an amount of|.*) mana\b"
    ):
        if is_creature:
            _add(labels, "mana_dork", 0.90, "Creature that produces mana.")
        elif is_artifact:
            _add(labels, "mana_rock", 0.90, "Artifact that produces mana.")
        else:
            _add(
                labels,
                "ritual",
                0.75,
                "Noncreature/nonartifact spell or permanent produces mana.",
            )

    if _has(text, r"spells? you cast cost .* less") or _has(
        text, r"costs? \{\d+\} less to cast"
    ):
        _add(labels, "cost_reducer", 0.85, "Reduces spell costs.")

    # -------------------------
    # Draw / card selection
    # -------------------------
    if _has(text, r"\bdraw (a|one|two|three|four|x|\d+) cards?\b"):
        _add(labels, "card_draw", 0.85, "Draws cards.")

    if _has(text, r"exile .* (top|from the top) .* library") and _has(
        text, r"you may (cast|play)"
    ):
        _add(
            labels,
            "impulse_draw",
            0.85,
            "Exiles cards from library and allows casting/playing them.",
        )

    if _has(text, r"draw .* cards?.*discard") or _has(
        text, r"draw a card,? then discard"
    ):
        _add(labels, "loot", 0.85, "Draw then discard effect.")

    if _has(text, r"discard .* cards?.*draw") or _has(
        text, r"discard a card,? then draw"
    ):
        _add(labels, "rummage", 0.85, "Discard then draw effect.")

    # -------------------------
    # Removal / interaction
    # -------------------------
    if _has(text, r"\bdestroy target\b"):
        _add(labels, "targeted_removal_destroy", 0.85, "Destroys a target.")

    if _has(text, r"\bexile target\b"):
        _add(labels, "targeted_removal_exile", 0.85, "Exiles a target.")

    if _has(text, r"\breturn target .* to (its owner's )?hand\b"):
        _add(
            labels,
            "targeted_removal_bounce",
            0.85,
            "Returns a target permanent/spell to hand.",
        )

    if _has(text, r"\bcounter target (spell|activated ability|triggered ability)\b"):
        _add(labels, "counterspell", 0.90, "Counters a spell or ability.")

    if _has(text, r"\bdestroy all\b|\bexile all\b|\beach creature\b|\ball creatures\b"):
        _add(labels, "board_wipe", 0.80, "Mass removal or affects all/every creature.")

    # -------------------------
    # Buffs / debuffs
    # -------------------------
    if _has(text, r"target creature gets \+\d/\+\d") or _has(
        text, r"gets \+\d/\+\d until end of turn"
    ):
        _add(
            labels,
            "pump_spell",
            0.85,
            "Temporarily increases creature power/toughness.",
        )

    if _has(text, r"target creature gets -\d/-\d") or _has(
        text, r"gets -\d/-\d until end of turn"
    ):
        _add(
            labels,
            "debuff_spell",
            0.85,
            "Temporarily decreases creature power/toughness.",
        )

    if _has(text, r"creatures you control get \+\d/\+\d") or _has(
        text, r"other creatures you control get"
    ):
        _add(labels, "anthem", 0.85, "Buffs multiple creatures you control.")

    # -------------------------
    # Tokens
    # -------------------------
    if _has(text, r"\bcreate .* token") or _has(
        text, r"\bcreate .* (treasure|clue|food|blood|map)"
    ):
        _add(labels, "token_maker", 0.90, "Creates tokens.")

    # -------------------------
    # Sacrifice
    # -------------------------
    if ":" in text and _has(
        text, r"sacrifice (a|another|any number of|one or more|this|[a-z ]+)"
    ):
        _add(labels, "sac_outlet", 0.85, "Sacrifice appears as an activated cost.")

    if _has(text, r"whenever you sacrifice") or _has(
        text, r"whenever .* (dies|is put into a graveyard)"
    ):
        _add(labels, "sac_payoff", 0.80, "Rewards sacrifice or death events.")

    # -------------------------
    # Discard
    # -------------------------
    if _has(text, r"discard a card:|discard .* cards?:"):
        _add(labels, "discard_outlet", 0.85, "Discard appears as an activated cost.")

    if _has(text, r"whenever (you|an opponent|a player) discards") or _has(
        text, r"opponent.*discard.*loses life"
    ):
        _add(labels, "discard_payoff", 0.85, "Rewards discard events.")

    # -------------------------
    # Mill / graveyard / reanimation
    # -------------------------
    if _has(text, r"\bmill\b") or _has(
        text, r"puts? the top .* cards? .* into .* graveyard"
    ):
        _add(labels, "mill", 0.85, "Moves cards from library to graveyard.")

    if _has(
        text,
        r"from your graveyard|in your graveyard|cards? in graveyards|escape|flashback|delve|threshold",
    ):
        _add(
            labels,
            "graveyard_payoff",
            0.75,
            "Uses or benefits from graveyard contents.",
        )

    if _has(
        text, r"return target .* card from your graveyard to the battlefield"
    ) or _has(text, r"return .* from your graveyard to the battlefield"):
        _add(
            labels, "reanimation", 0.90, "Returns cards from graveyard to battlefield."
        )

    # -------------------------
    # Blink
    # -------------------------
    if _has(text, r"exile target .* you control.*return") or _has(
        text, r"exile .* then return .* battlefield"
    ):
        _add(labels, "blink_enabler", 0.90, "Exiles and returns permanents.")

    # -------------------------
    # Lifegain payoff
    # -------------------------
    if _has(text, r"whenever you gain life|if you gained life|each time you gain life"):
        _add(labels, "lifegain_payoff", 0.90, "Rewards life gain.")

    # -------------------------
    # Group hug
    # -------------------------
    if _has(text, r"each player draws|each opponent draws|players draw"):
        _add(labels, "group_hug_draw", 0.85, "Lets multiple players draw cards.")

    if _has(text, r"each player may search .* land") or _has(
        text, r"each player.*put .* land .* battlefield"
    ):
        _add(labels, "group_hug_ramp", 0.85, "Lets multiple players ramp.")

    # -------------------------
    # Tutor
    # -------------------------
    if _has(text, r"search your library for .* card") and not _has(
        text, r"land card|basic land"
    ):
        _add(labels, "tutor", 0.85, "Searches library for a nonland card.")

    # -------------------------
    # Stax / protection
    # -------------------------
    if _has(
        text,
        r"can't cast|can't attack|can't block|doesn't untap|skip .* step|spells cost .* more|players can't|opponents can't",
    ):
        _add(
            labels,
            "stax_piece",
            0.75,
            "Restricts actions, resources, untapping, casting, or combat.",
        )

    if _has(
        text,
        r"gain hexproof|gains hexproof|indestructible until end|gains indestructible|protection from|prevent all damage|phase out",
    ):
        _add(labels, "protection", 0.80, "Protects permanents or players.")

    # Deduplicate, keeping highest-confidence match per label
    best = dedupe_label_matches(labels)

    if not best:
        best["needs_review"] = LabelMatch(
            "needs_review",
            0.30,
            "rule",
            "No confident rule label matched.",
        )

    return sorted(best.values(), key=lambda m: m.confidence, reverse=True)


def dedupe_label_matches(labels: list[LabelMatch]) -> dict[str, LabelMatch]:
    """
    Input:
        Raw label matches.

    Logic:
        Keeps the highest-confidence match for each label.

    Output:
        Dictionary of best match per label.
    """
    best: dict[str, LabelMatch] = {}

    for match in labels:
        if match.label not in best or match.confidence > best[match.label].confidence:
            best[match.label] = match

    return best
