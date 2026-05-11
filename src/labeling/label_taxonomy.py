from __future__ import annotations

from dataclasses import dataclass
from typing import Literal


LabelCategory = Literal[
    "card_role",
    "mechanic",
    "archetype_piece",
    "deck_function",
    "needs_review",
]


@dataclass(frozen=True)
class LabelDefinition:
    label: str
    category: LabelCategory
    description: str


LABEL_TAXONOMY: dict[str, LabelDefinition] = {
    "vanilla_creature": LabelDefinition(
        label="vanilla_creature",
        category="card_role",
        description="Creature with no rules text beyond possible flavor/reminder absence.",
    ),
    "french_vanilla_creature": LabelDefinition(
        label="french_vanilla_creature",
        category="card_role",
        description="Creature whose text is mostly keyword abilities.",
    ),
    "land_ramp": LabelDefinition(
        label="land_ramp",
        category="card_role",
        description="Card that searches the library for lands.",
    ),
    "ritual": LabelDefinition(
        label="ritual",
        category="card_role",
        description="Card that adds mana when cast.",
    ),
    "mana_dork": LabelDefinition(
        label="mana_dork",
        category="card_role",
        description="Creature that produces mana.",
    ),
    "mana_rock": LabelDefinition(
        label="mana_rock",
        category="card_role",
        description="Artifact that produces mana.",
    ),
    "cost_reducer": LabelDefinition(
        label="cost_reducer",
        category="card_role",
        description="Reduces the mana cost of other spells.",
    ),
    "card_draw": LabelDefinition(
        label="card_draw",
        category="card_role",
        description="Card that draws cards.",
    ),
    "impulse_draw": LabelDefinition(
        label="impulse_draw",
        category="card_role",
        description="Card that exiles cards from the library to be cast.",
    ),
    "loot": LabelDefinition(
        label="loot",
        category="card_role",
        description="Draw then discard style card selection.",
    ),
    "rummage": LabelDefinition(
        label="rummage",
        category="card_role",
        description="Discard then draw style card selection.",
    ),
    "targeted_removal_destroy": LabelDefinition(
        label="targeted_removal_destroy",
        category="card_role",
        description="Destroys a target permanent.",
    ),
    "targeted_removal_exile": LabelDefinition(
        label="targeted_removal_exile",
        category="card_role",
        description="Exiles a target permanent.",
    ),
    "targeted_removal_bounce": LabelDefinition(
        label="targeted_removal_bounce",
        category="card_role",
        description="Returns a target permanent to hand.",
    ),
    "board_wipe": LabelDefinition(
        label="board_wipe",
        category="card_role",
        description="Mass removal affecting many permanents or creatures.",
    ),
    "counterspell": LabelDefinition(
        label="counterspell",
        category="card_role",
        description="Counters spells or abilities.",
    ),
    "pump_spell": LabelDefinition(
        label="pump_spell",
        category="card_role",
        description="Temporarily increases target creature's power/toughness.",
    ),
    "debuff_spell": LabelDefinition(
        label="debuff_spell",
        category="card_role",
        description="Temporarily decreases target creature's power/toughness.",
    ),
    "anthem": LabelDefinition(
        label="anthem",
        category="card_role",
        description="Static or recurring effect that buffs multiple creatures.",
    ),
    "token_maker": LabelDefinition(
        label="token_maker",
        category="card_role",
        description="Creates creature, artifact, treasure, clue, food, or other tokens.",
    ),
    "sac_outlet": LabelDefinition(
        label="sac_outlet",
        category="card_role",
        description="Allows sacrificing permanents as a cost or repeatable action.",
    ),
    "sac_payoff": LabelDefinition(
        label="sac_payoff",
        category="card_role",
        description="Rewards permanents dying or being sacrificed.",
    ),
    "discard_outlet": LabelDefinition(
        label="discard_outlet",
        category="card_role",
        description="Allows the controller to discard cards for value or as a cost.",
    ),
    "discard_payoff": LabelDefinition(
        label="discard_payoff",
        category="card_role",
        description="Rewards opponents or players discarding cards.",
    ),
    "mill": LabelDefinition(
        label="mill",
        category="card_role",
        description="Card that moves other cards from the library to the graveyard.",
    ),
    "graveyard_payoff": LabelDefinition(
        label="graveyard_payoff",
        category="card_role",
        description="Benefits from graveyard contents or cards leaving graveyards.",
    ),
    "reanimation": LabelDefinition(
        label="reanimation",
        category="card_role",
        description="Returns creature or permanent cards from graveyard to battlefield.",
    ),
    "blink_enabler": LabelDefinition(
        label="blink_enabler",
        category="card_role",
        description="Exiles and returns permanents to reuse ETB/LTB effects.",
    ),
    "lifegain_payoff": LabelDefinition(
        label="lifegain_payoff",
        category="card_role",
        description="Rewards gaining life.",
    ),
    "group_hug_draw": LabelDefinition(
        label="group_hug_draw",
        category="card_role",
        description="Lets multiple players draw cards.",
    ),
    "group_hug_ramp": LabelDefinition(
        label="group_hug_ramp",
        category="card_role",
        description="Lets multiple players ramp.",
    ),
    "tutor": LabelDefinition(
        label="tutor",
        category="card_role",
        description="Searches library for a specific card or card type.",
    ),
    "stax_piece": LabelDefinition(
        label="stax_piece",
        category="card_role",
        description="Restricts actions, resources, untapping, casting, or combat.",
    ),
    "protection": LabelDefinition(
        label="protection",
        category="card_role",
        description="Protects permanents or players through hexproof, indestructible, prevention, or similar effects.",
    ),
    "needs_review": LabelDefinition(
        label="needs_review",
        category="needs_review",
        description="Card could not be confidently labeled or has ambiguous text.",
    ),
}


def validate_taxonomy() -> None:
    """
    Input:
        LABEL_TAXONOMY.

    Logic:
        Confirms dictionary keys match their LabelDefinition labels.

    Output:
        Raises ValueError if taxonomy entries are inconsistent.
    """
    mismatches = [
        key for key, definition in LABEL_TAXONOMY.items() if key != definition.label
    ]

    if mismatches:
        raise ValueError(f"Taxonomy key/label mismatches: {mismatches}")


def get_label_definition(label: str) -> LabelDefinition | None:
    """
    Input:
        Label name.

    Logic:
        Looks up a label definition if the label exists.

    Output:
        LabelDefinition or None.
    """
    return LABEL_TAXONOMY.get(label)


def get_label_names() -> list[str]:
    """
    Input:
        LABEL_TAXONOMY.

    Logic:
        Returns all known label names in stable sorted order.

    Output:
        Sorted label-name list.
    """
    return sorted(LABEL_TAXONOMY.keys())


validate_taxonomy()


# TODO Phase 4:
# Experiment with richer label taxonomy designs:
#   - split broad card_role labels into mechanics, functions, and archetype roles
#   - support hierarchical labels such as removal > exile_removal > creature_removal
#   - add label metadata for expected false positives and rule confidence
#   - map labels to deck archetypes and synergy graph concepts
#   - compare hand-built taxonomy against labels discovered from clustering
