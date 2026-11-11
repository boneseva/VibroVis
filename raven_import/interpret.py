"""
Interpret annotation values as species, without altering the original text.

Rules (ruleset ``config.INTERPRETATION_RULESET``):
- Surrounding whitespace is ignored; one or more trailing '?' mark the value uncertain.
- Exact vocabulary match -> rule "exact"; case-insensitive match -> "case_normalized";
  configured synonym (case-insensitive) -> "synonym".
- "?" alone -> no species, uncertain, rule "uncertain_only"; empty -> rule "empty";
  anything else (speech, other animals, notes) -> no species, rule "unrecognized".
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, Optional


@dataclass(frozen=True)
class Interpretation:
    species: Optional[str]
    uncertain: bool
    rule: str


def interpret_label(value: Optional[str], vocabulary: Iterable[str], synonyms: Dict[str, str]) -> Interpretation:
    text = (value or "").strip()
    if not text:
        return Interpretation(None, False, "empty")
    core = text.rstrip("?").strip()
    uncertain = core != text
    if not core:
        return Interpretation(None, True, "uncertain_only")
    vocab = list(vocabulary)
    if core in vocab:
        return Interpretation(core, uncertain, "exact")
    folded = {v.casefold(): v for v in vocab}
    if core.casefold() in folded:
        return Interpretation(folded[core.casefold()], uncertain, "case_normalized")
    syn = {k.casefold(): v for k, v in (synonyms or {}).items()}
    if core.casefold() in syn:
        return Interpretation(syn[core.casefold()], uncertain, "synonym")
    return Interpretation(None, uncertain, "unrecognized")
