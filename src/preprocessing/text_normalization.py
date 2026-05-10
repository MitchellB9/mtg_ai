from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Optional

# Parenthetical reminder text
REMINDER_RE = re.compile(r"\([^)]*\)") 

# Horizontal whitespace normalization
WS_RE = re.compile(r"[ \t]+") 

# Collapse excessive blank lines while preserving ability separation
MULTILINE_BLANKS_RE = re.compile(r"\n{3,}")


@dataclass(frozen=True)
class TextNormalizationOptions:
    strip_reminder_text: bool = False
    replace_card_name: bool = True

# TODO:
# Explore entity-aware oracle-text parsing instead of simple name replacement.
#
# Challenges:
#   - self references may use partial names ("Frodo")
#   - cards may reference other named cards ("Festering Newt")
#   - mechanics may reference linked partners/backgrounds/etc.
#   - named tokens and external entities carry semantic meaning
#
# Future approaches:
#   - distinguish self references from external card references
#   - build structured entity extraction for referenced cards
#   - preserve relationship semantics for synergy analysis
#   - compare entity-preserving vs entity-normalized NLP pipelines
def safe_card_name_pattern(card_name: str) -> re.Pattern:
    """
    Input:
        Raw card name.

    Logic:
        Builds a conservative regex matching the full card name while reducing
        accidental partial-word replacements.

    Output:
        Compiled regex pattern for card-name replacement.
    """
    escaped = re.escape(card_name.strip())
    # match on word boundaries where possible (names can include commas/apostrophes)
    return re.compile(rf"(?<!\w){escaped}(?!\w)")

def normalize_line_whitespace(text: str) -> str:
    """
    Input:
        Oracle text string.

    Logic:
        Normalizes repeated spaces/tabs while preserving newline structure.

    Output:
        Text with cleaned horizontal whitespace.
    """
    text = WS_RE.sub(" ", text)

    lines = [line.strip() for line in text.split("\n")]

    return "\n".join(lines)

def normalize_blank_lines(text: str) -> str:
    """
    Input:
        Oracle text string.

    Logic:
        Collapses excessive blank lines while preserving paragraph/ability
        separation.

    Output:
        Text with normalized blank-line spacing.
    """
    return MULTILINE_BLANKS_RE.sub("\n\n", text)

def replace_card_name_references(
    text: str,
    card_name: Optional[str],
) -> str:
    """
    Input:
        Oracle text and the source card name.

    Logic:
        Replaces explicit references to the card's own name with a generic token.

    Output:
        Oracle text with self-name references normalized.
    """
    if not card_name:
        return text

    pattern = safe_card_name_pattern(card_name)

    return pattern.sub("this card", text)

def normalize_oracle_text(
    oracle_text: Optional[str],
    card_name: Optional[str],
    opts: Optional[TextNormalizationOptions] = None,
) -> str:
    """
    Input:
        Raw oracle text and card name.

    Logic:
        Applies configurable oracle-text normalization steps for downstream NLP.

    Output:
        Normalized oracle text string.
    """

    if opts is None:
        opts = TextNormalizationOptions()

    text = oracle_text or ""
    # Standardize newline formats
    text = text.replace("\r\n", "\n").replace("\r", "\n").strip()

    if not text:
        return ""

    # Optional reminder text removal
    if opts.strip_reminder_text:
        text = REMINDER_RE.sub("", text)

    # Replace explicit self-name references
    if opts.replace_card_name:
        text = replace_card_name_references(text, card_name)

    # Normalize whitespace while preserving ability-line structure
    text = WS_RE.sub(" ", text)
    # Remove excessive blank-line spacing
    text = normalize_blank_lines(text)

    return text.strip()

# TODO Phase 4:
# Experiment with alternative normalization strategies:
#   - preserve vs remove reminder text as separate NLP branches
#   - canonicalize mana symbols into structured semantic tokens
#   - compare "this card" replacement vs preserving exact card names
#   - sentence/ability segmentation before tokenization
#   - structured extraction of costs, effects, targets, and conditions
#   - compare punctuation-preserving vs punctuation-stripped normalization
#   - evaluate mechanic-aware preprocessing pipelines
