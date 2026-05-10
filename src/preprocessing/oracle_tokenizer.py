from __future__ import annotations

import re
from dataclasses import dataclass
from typing import List, Optional

# Matches Magic mana symbols such as {G}, {2}, {U/B}, {X}, {T}.
MANA_RE = re.compile(r"\{[^}]+\}")
# Matches power/toughness counter patterns such as +1/+1 or -1/-1.
COUNTER_RE = re.compile(r"(?<!\w)[+-]\d/[+-]\d(?!\w)")
# Matches standalone integer numbers.
NUMBER_RE = re.compile(r"\b\d+\b")

# Very starter set of multiword templating phrases that you may want as single tokens.
# Keep this small at first; add more once you measure impact.
MULTIWORD_PHRASES = [
    "enters the battlefield",
    "until end of turn",
    "at the beginning of",
    "as long as",
    "you may",
    "target creature",
    "target player",
    "target opponent",
    "draw a card",
]

# Removes common punctuation while preserving apostrophes.
PUNCT_RE = re.compile(r"[.,;:!?]+")


@dataclass(frozen=True)
class TokenizeOptions:
    """
    Configuration controlling oracle-text tokenization behavior.
    """
    keep_newlines: bool = True
    lowercase: bool = True


def collapse_multiword_phrases(text: str) -> str:
    """
    Input:
        Normalized oracle text.

    Logic:
        Converts selected multiword phrases into underscore-connected tokens.

    Output:
        Text with selected phrases preserved as single units.
    """
    phrases = sorted(MULTIWORD_PHRASES, key=len, reverse=True)
    output = text

    for phrase in phrases:
        output = re.sub(
            rf"\b{re.escape(phrase)}\b",
            phrase.replace(" ", "_"),
            output,
            flags=re.IGNORECASE,
        )

    return output

def tokenize_line(line: str, opts: TokenizeOptions) -> list[str]:
    """
    Input:
        One normalized oracle-text line.

    Logic:
        Applies casing, phrase collapsing, mana extraction, punctuation cleanup,
        numeric tagging, and counter tagging.

    Output:
        Tokens extracted from the line.
    """
    work = line.strip()

    if opts.lowercase:
        work = work.lower()

    work = collapse_multiword_phrases(work)

    mana_symbols = MANA_RE.findall(work)
    work_without_mana = MANA_RE.sub(" MANA_SYMBOL ", work)

    work_without_mana = PUNCT_RE.sub(" ", work_without_mana)

    rough_tokens = [token for token in work_without_mana.split(" ") if token]

    tokens: list[str] = []
    mana_index = 0

    for token in rough_tokens:
        if token == "mana_symbol":
            symbol = mana_symbols[mana_index] if mana_index < len(mana_symbols) else "{?}"
            tokens.append(f"MANA:{symbol.upper()}")
            mana_index += 1
            continue

        if COUNTER_RE.fullmatch(token):
            tokens.append(f"COUNTER:{token}")
            continue

        if NUMBER_RE.fullmatch(token):
            tokens.append(f"NUM:{token}")
            continue

        tokens.append(token)

    return tokens

def tokenize_oracle_text(
    normalized_text: Optional[str],
    opts: Optional[TokenizeOptions] = None,
) -> List[str]:
    """
    Input:
        Normalized oracle text.

    Logic:
        Tokenizes oracle text while optionally preserving ability-line boundaries.

    Output:
        List of oracle-text tokens.
    """
    if opts is None:
        opts = TokenizeOptions()

    text = (normalized_text or "").strip()

    if not text:
        return []

    tokens: list[str] = []

    for line in text.split("\n"):
        line = line.strip()

        if not line:
            if opts.keep_newlines and tokens and tokens[-1] != "NEWLINE":
                tokens.append("NEWLINE")
            continue

        tokens.extend(tokenize_line(line, opts))

        if opts.keep_newlines:
            tokens.append("NEWLINE")

    while tokens and tokens[-1] == "NEWLINE":
        tokens.pop()

    return tokens

# TODO Phase 4:
# Experiment with alternative tokenization strategies:
#   - compare phrase-token lists vs learned n-grams from vectorizers
#   - create mechanic-aware tokens for costs, targets, zones, and timing
#   - preserve mana symbols as structured color/value features
#   - tokenize by ability line, sentence, clause, or cost/effect boundary
#   - compare flat tokenization against entity-aware oracle parsing
#   - build separate token streams for rules text, costs, and generated objects
