from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

CANONICAL_TYPES = {
    "Artifact",
    "Battle",
    "Creature",
    "Enchantment",
    "Instant",
    "Kindred",
    "Land",
    "Planeswalker",
    "Sorcery",
}


@dataclass(frozen=True)
class ParsedTypeLine:
    """
    Parsed representation of a Scryfall type_line.
    """

    faces: list[dict[str, Any]]
    basic_types: list[str]
    super_types: list[str]
    sub_types: list[str]


def split_faces(type_line: Optional[str]) -> list[str]:
    """
    Input:
        Raw Scryfall type_line.

    Logic:
        Splits multi-face type lines on Scryfall's face separator.

    Output:
        List of individual face type lines.
    """
    if not type_line:
        return []

    return [face.strip() for face in type_line.split("//") if face.strip()]


def split_main_and_subtypes(face_type_line: str) -> tuple[str, str]:
    """
    Input:
        One face's type line.

    Logic:
        Splits the main type section from the subtype section.

    Output:
        Tuple of (main_types_text, subtype_text).
    """
    if "—" in face_type_line:
        main, subtypes = face_type_line.split("—", maxsplit=1)
        return main.strip(), subtypes.strip()

    if " - " in face_type_line:
        main, subtypes = face_type_line.split(" - ", maxsplit=1)
        return main.strip(), subtypes.strip()

    return face_type_line.strip(), ""


def parse_face_type_line(face_type_line: str) -> dict[str, Any]:
    """
    Input:
        One face's type line.

    Logic:
        Separates known card types from supertypes and extracts subtypes.

    Output:
        Dictionary containing parsed type components for the face.
    """
    main_text, subtype_text = split_main_and_subtypes(face_type_line)

    main_tokens = [token for token in main_text.split() if token]

    basic_types = [token for token in main_tokens if token in CANONICAL_TYPES]
    super_types = [token for token in main_tokens if token not in CANONICAL_TYPES]
    sub_types = [token for token in subtype_text.split() if token]

    return {
        "raw_face_type_line": face_type_line,
        "basic_types": basic_types,
        "super_types": super_types,
        "sub_types": sub_types,
    }


def parse_type_line(type_line: Optional[str]) -> ParsedTypeLine:
    """
    Input:
        Raw Scryfall type_line.

    Logic:
        Parses each face, then aggregates card types, supertypes, and subtypes
        across all faces.

    Output:
        ParsedTypeLine object with face-level and card-level type features.
    """
    raw_faces = split_faces(type_line)
    parsed_faces = [parse_face_type_line(face) for face in raw_faces]

    basic_types: list[str] = []
    super_types: list[str] = []
    sub_types: list[str] = []

    for face in parsed_faces:
        basic_types.extend(face["basic_types"])
        super_types.extend(face["super_types"])
        sub_types.extend(face["sub_types"])

    return ParsedTypeLine(
        faces=parsed_faces,
        basic_types=basic_types,
        super_types=super_types,
        sub_types=sub_types,
    )


# TODO Phase 4:
# Experiment with richer type parsing approaches:
#   - preserve face-specific type features for double-faced/modal cards
#   - normalize subtype aliases and historical creature type changes
#   - build type hierarchy features for artifact creatures, enchantment creatures, etc.
#   - compare flat type lists vs structured type graphs
#   - extract type-change effects from oracle text, not just printed type_line
