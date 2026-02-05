from __future__ import annotations

from dataclasses import dataclass
from typing import List, Dict, Any, Tuple, Optional

# Canonical "card types" (the ones that are NOT supertypes)
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
    faces: List[
        Dict[str, Any]
    ]  # list of {basic_types, super_types, sub_types, raw_face_type_line}
    basic_types: List[str]
    super_types: List[str]
    sub_types: List[str]


def _split_faces(type_line: Optional[str]) -> List[str]:
    if not type_line:
        return []
    # multi-faced types use " // "
    return [s.strip() for s in type_line.split("//")]


def _split_main_and_subtypes(face_type_line: str) -> Tuple[str, str]:
    """
    Scryfall uses an em dash "—" (U+2014) typically, but sometimes hyphen appears.
    We'll support: " - " and "—" variants.
    """
    if "—" in face_type_line:
        parts = [p.strip() for p in face_type_line.split("—", maxsplit=1)]
    elif " - " in face_type_line:
        parts = [p.strip() for p in face_type_line.split(" - ", maxsplit=1)]
    elif "-" in face_type_line and " - " not in face_type_line:
        # last resort: split once (can be risky if a subtype contains hyphen)
        parts = [p.strip() for p in face_type_line.split("-", maxsplit=1)]
    else:
        parts = [face_type_line.strip(), ""]

    main = parts[0]
    sub = parts[1] if len(parts) > 1 else ""
    return main, sub


def _parse_face(face_type_line: str) -> Dict[str, Any]:
    main, sub = _split_main_and_subtypes(face_type_line)

    main_tokens = [t for t in main.split(" ") if t]  # preserve order
    basic_types = [t for t in main_tokens if t in CANONICAL_TYPES]
    super_types = [t for t in main_tokens if t not in CANONICAL_TYPES]

    # subtype tokens are space-separated, e.g. "Human Soldier"
    sub_types = [t for t in sub.split(" ") if t] if sub else []

    return {
        "raw_face_type_line": face_type_line,
        "basic_types": basic_types,
        "super_types": super_types,
        "sub_types": sub_types,
    }


def parse_type_line(type_line: Optional[str]) -> ParsedTypeLine:
    faces_raw = _split_faces(type_line)
    faces_parsed = [_parse_face(face) for face in faces_raw] if faces_raw else []

    basic_all: List[str] = []
    super_all: List[str] = []
    sub_all: List[str] = []

    for f in faces_parsed:
        basic_all.extend(f["basic_types"])
        super_all.extend(f["super_types"])
        sub_all.extend(f["sub_types"])

    return ParsedTypeLine(
        faces=faces_parsed,
        basic_types=basic_all,
        super_types=super_all,
        sub_types=sub_all,
    )
