from __future__ import annotations

import re
from dataclasses import dataclass
from typing import List, Optional

MANA_RE = re.compile(r"\{[^}]+\}")
COUNTER_RE = re.compile(r"\b[+-]\d/[+-]\d\b")
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

PUNCT_RE = re.compile(r"[.,;:!?]+")


@dataclass(frozen=True)
class TokenizeOptions:
    keep_newlines: bool = True
    lowercase: bool = True


def _collapse_multiword_phrases(text: str) -> str:
    # Replace phrases with underscored versions to keep them together.
    # Do longest-first to avoid partial overlaps.
    phrases = sorted(MULTIWORD_PHRASES, key=len, reverse=True)
    out = text
    for p in phrases:
        out = re.sub(
            rf"\b{re.escape(p)}\b", p.replace(" ", "_"), out, flags=re.IGNORECASE
        )
    return out


def tokenize_oracle_text(
    normalized_text: Optional[str], opts: Optional[TokenizeOptions] = None
) -> List[str]:
    if opts is None:
        opts = TokenizeOptions()

    text = (normalized_text or "").strip()
    if not text:
        return []

    # Preserve NEWLINE token boundaries if requested
    lines = text.split("\n")
    tokens: List[str] = []

    for li, line in enumerate(lines):
        line = line.strip()
        if not line:
            # Blank line -> treat like NEWLINE separator
            if opts.keep_newlines and tokens and tokens[-1] != "NEWLINE":
                tokens.append("NEWLINE")
            continue

        work = line
        if opts.lowercase:
            work = work.lower()

        work = _collapse_multiword_phrases(work)

        # Pull mana symbols into placeholders first
        mana_symbols = MANA_RE.findall(work)
        work_wo_mana = MANA_RE.sub(" MANA_SYMBOL ", work)

        # Basic punctuation cleanup (don’t nuke apostrophes; they matter sometimes)
        work_wo_mana = PUNCT_RE.sub(" ", work_wo_mana)

        # Token split
        rough = [t for t in work_wo_mana.split(" ") if t]

        # Re-inject mana symbols in order
        mana_i = 0
        for t in rough:
            if t == "mana_symbol":
                sym = mana_symbols[mana_i] if mana_i < len(mana_symbols) else "{?}"
                tokens.append(f"MANA:{sym.upper()}")
                mana_i += 1
                continue

            # Counters
            if COUNTER_RE.fullmatch(t):
                tokens.append(f"COUNTER:{t}")
                continue

            # Pure numbers
            if NUMBER_RE.fullmatch(t):
                tokens.append(f"NUM:{t}")
                continue

            tokens.append(t)

        if opts.keep_newlines:
            tokens.append("NEWLINE")

    # Trim trailing NEWLINE if present
    while tokens and tokens[-1] == "NEWLINE":
        tokens.pop()

    return tokens
