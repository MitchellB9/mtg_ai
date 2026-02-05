from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Optional

REMINDER_RE = re.compile(r"\([^)]*\)")  # simple: removes parenthetical reminder text
WS_RE = re.compile(r"[ \t]+")
MULTILINE_BLANKS_RE = re.compile(r"\n{3,}")


@dataclass(frozen=True)
class TextNormalizationOptions:
    strip_reminder_text: bool = False
    replace_card_name: bool = True


def _safe_name_pattern(card_name: str) -> re.Pattern:
    """
    Build a conservative regex that matches the card name as a whole phrase.
    We avoid aggressive matching to reduce false replacements.
    """
    escaped = re.escape(card_name.strip())
    # match on word boundaries where possible (names can include commas/apostrophes)
    return re.compile(rf"(?<!\w){escaped}(?!\w)")


def normalize_oracle_text(
    oracle_text: Optional[str],
    card_name: Optional[str],
    opts: Optional[TextNormalizationOptions] = None,
) -> str:
    if opts is None:
        opts = TextNormalizationOptions()

    text = oracle_text or ""
    text = text.replace("\r\n", "\n").replace("\r", "\n").strip()

    if not text:
        return ""

    if opts.strip_reminder_text:
        text = REMINDER_RE.sub("", text)

    # Replace the exact printed name with "this card"
    if opts.replace_card_name and card_name:
        pat = _safe_name_pattern(card_name)
        text = pat.sub("this card", text)

    # Normalize whitespace
    text = WS_RE.sub(" ", text)
    # Keep newlines (they matter: abilities are line-separated)
    text = "\n".join(line.strip() for line in text.split("\n"))
    text = MULTILINE_BLANKS_RE.sub("\n\n", text)
    return text.strip()
