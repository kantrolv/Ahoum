"""Pydantic schemas — the typed contracts that cross module boundaries.

Why pydantic everywhere: the LLM returns free-form text, but the rest of the
pipeline needs *guaranteed-shaped* data. We hand the model a JSON schema, ask it
to fill it in (Ollama structured output), then validate. If the model drifts, we
fail loudly here instead of corrupting the registry. This same pattern powers the
Phase 3 scorer, so the registry build doubles as a rehearsal.
"""

from __future__ import annotations

import re
from typing import Literal

from pydantic import BaseModel, Field, field_validator, model_validator

from . import config

# A level marker at the START of an anchor: "1", "1.", "1:", "1)", "1 =",
# "Level 1 -", etc. Group 1 = the digit, group 2 = the remaining description.
_LEVEL_PREFIX = re.compile(r"^\s*(?:level\s*)?([1-5])\s*[\).:=\-]?\s*(.*)$", re.I)
# Two or more "N=" / "N:" markers in one cell => a legend dump, not a single anchor.
_LEGEND_MARK = re.compile(r"[1-5]\s*[=:]")
# A bare parenthesized level like "(3)" — a leaked level marker, not real text.
_BARE_LEVEL = re.compile(r"\([1-5]\)")


def _dedup_key(text: str) -> str:
    """Letters-only lowercase key, for spotting (near-)duplicate anchor text."""
    return re.sub(r"[^a-z0-9]+", "", text.lower())


def normalize_anchors(anchors: list[str]) -> list[str] | None:
    """Return five clean anchors ordered ascending (lowest->highest), or None.

    Two valid shapes are accepted, because the model produces both:

      * NUMBERED — every anchor begins with its level number (e.g. an old row
        with "5: ..." sitting in the scale_1 field). We trust the numbers,
        reorder by them, and strip the prefix. This is what repairs the
        wrong-order rows.
      * UNNUMBERED — no anchor has a leading number (the normal shape when the
        model fills the named scale_1..scale_5 fields directly). The field order
        IS the level order, so we accept it as-is.

    None means "malformed — regenerate": a legend dump ("1=.. 2=.. 3=.."), an
    empty anchor, numbers that don't form a clean {1,2,3,4,5} set
    (duplicated/skipped levels), or a confusing mix of numbered and unnumbered.

    One helper, three callers: the schema validator (fix at generation), the
    deterministic normalization pass (fix rows already written), and the audit.
    """
    if len(anchors) != 5:
        return None

    cells = [str(a).strip() for a in anchors]
    for cell in cells:
        if not cell:
            return None  # empty anchor
        if len(_LEGEND_MARK.findall(cell)) >= 2:
            return None  # legend dump packed into one field
        if _BARE_LEVEL.search(cell):
            return None  # leaked "(N)" level marker => generic-label echo

    parsed: list[tuple[int | None, str]] = []
    for cell in cells:
        m = _LEVEL_PREFIX.match(cell)
        if m and m.group(2).strip():
            parsed.append((int(m.group(1)), m.group(2).strip()))
        else:
            parsed.append((None, cell))

    levels = [lvl for lvl, _ in parsed]
    if all(lvl is not None for lvl in levels):
        if sorted(levels) != [1, 2, 3, 4, 5]:
            return None  # duplicated or skipped levels -> malformed
        by_level = {lvl: txt for lvl, txt in parsed}
        result = [by_level[i] for i in range(1, 6)]
    elif all(lvl is None for lvl in levels):
        result = cells  # no numbers: trust the field order
    else:
        return None  # ambiguous mix of numbered and unnumbered anchors

    # Reject (near-)duplicate anchors: five levels must be five distinct texts.
    if len({_dedup_key(t) for t in result}) < 5:
        return None
    return result

# Literal types built from config so there is ONE source of truth for the
# allowed values. The model's output is constrained to exactly these.
CategoryT = Literal[
    "linguistic",
    "pragmatic",
    "emotion",
    "personality",
    "behavioral",
    "cognitive",
    "safety",
    "spiritual",
    "other",
]
PolarityT = Literal["positive", "negative", "neutral"]


class FacetEnrichment(BaseModel):
    """What the LLM must produce for a single facet during Phase 1 enrichment."""

    category: CategoryT = Field(
        description="Single best-fit category from the fixed taxonomy."
    )
    definition: str = Field(
        description="One clear sentence defining the facet as a measurable trait.",
        min_length=10,
        max_length=400,
    )
    polarity: PolarityT = Field(
        description=(
            "positive = a high score is generally adaptive/desirable; "
            "negative = a high score is generally maladaptive/undesirable; "
            "neutral = descriptive/quantitative with no inherent valence."
        )
    )
    scale_1: str = Field(description="Anchor for score 1 (absent / not present).")
    scale_2: str = Field(description="Anchor for score 2 (slight / faint trace).")
    scale_3: str = Field(description="Anchor for score 3 (moderate / clearly present).")
    scale_4: str = Field(description="Anchor for score 4 (strong / prominent).")
    scale_5: str = Field(description="Anchor for score 5 (very strong / dominant).")

    @model_validator(mode="after")
    def _ascending_anchors(self):
        """Auto-reorder anchors to ascending intensity; reject if unparseable.

        Most rows self-heal here (the model often emits the right text with the
        right leading number but in the wrong field) — we reorder and strip the
        prefix. Truly malformed output (legend dump, missing level) raises, which
        makes `chat_structured` retry with a corrective nudge.
        """
        fixed = normalize_anchors(
            [self.scale_1, self.scale_2, self.scale_3, self.scale_4, self.scale_5]
        )
        if fixed is None:
            raise ValueError(
                "anchors must be five distinct levels 1..5, each starting with its "
                "level number, ascending absent->dominant, with no 'N=' legend text"
            )
        self.scale_1, self.scale_2, self.scale_3, self.scale_4, self.scale_5 = fixed
        return self

    def scale_anchors(self) -> dict[int, str]:
        return {
            1: self.scale_1,
            2: self.scale_2,
            3: self.scale_3,
            4: self.scale_4,
            5: self.scale_5,
        }


class FacetScore(BaseModel):
    """One facet's score for one conversation turn (Phase 3 scorer output)."""

    facet_id: str = Field(description="The facet id being scored, e.g. F0134.")
    score: int = Field(
        ge=config.SCALE_MIN,
        le=config.SCALE_MAX,
        description="Integer 1-5: how strongly the turn exhibits this facet, "
        "judged against the facet's own rubric (1=absent, 5=dominant).",
    )
    confidence: float = Field(
        ge=0.0,
        le=1.0,
        description="0.0-1.0: how certain this score is, given how much evidence "
        "the turn actually provides for this facet (0=pure guess, 1=certain).",
    )

    @field_validator("score", mode="before")
    @classmethod
    def _clamp_score(cls, v):
        """Clamp to 1..5. Small models sometimes emit 0 (absent) or 6."""
        try:
            v = int(round(float(v)))
        except (TypeError, ValueError):
            return config.SCALE_MIN
        return max(config.SCALE_MIN, min(config.SCALE_MAX, v))

    @field_validator("confidence", mode="before")
    @classmethod
    def _coerce_confidence(cls, v):
        """Coerce confidence into 0..1. The model sometimes uses the 1-5 score
        scale by mistake (e.g. confidence=5); rescale those by /5 instead of
        rejecting the whole batch and crashing the run."""
        try:
            v = float(v)
        except (TypeError, ValueError):
            return 0.5
        if v > 1.0:
            v = v / 5.0  # model mistakenly used the 1-5 scale
        return max(0.0, min(1.0, v))


class BatchScores(BaseModel):
    """The scorer returns one FacetScore per facet in the batch."""

    scores: list[FacetScore]


# Sanity check at import time: the Literal must match config exactly, so the two
# can never silently drift apart.
assert set(CategoryT.__args__) == set(config.FACET_CATEGORIES), (
    "CategoryT and config.FACET_CATEGORIES disagree — update both together."
)
assert set(PolarityT.__args__) == set(config.FACET_POLARITIES)
