"""Phase 1b — enrich each clean facet with LLM-drafted metadata.

Per facet, one structured LLM call produces: category, definition, polarity, and
the five rubric anchors (scale_1..scale_5). This is a one-time *offline* registry
build, not part of per-conversation scoring — so we favor reliability (one facet
per call, temperature 0, schema-validated) over speed.

Scaling note: enriching 5000 facets is the same loop with more rows. The work is
embarrassingly parallel and cached/resumable, so it never blocks the live scorer.

Run standalone:
    python -m facet_eval.enrich --sample 10     # draft 10, print, write nothing
    python -m facet_eval.enrich --all           # draft all, write facets_enriched.csv
"""

from __future__ import annotations

import argparse
import re

import pandas as pd

from . import config
from .clean import load_clean_facets
from .llm import LLMError, chat_structured
from .schema import FacetEnrichment

# Collapse any whitespace run (incl. newlines/tabs) to a single space so every
# definition and rubric anchor is a clean single line in the CSV.
_WS = re.compile(r"\s+")


def _clean_text(s: str) -> str:
    return _WS.sub(" ", str(s)).strip()

# Built once; injected into the prompt so the model sees the exact taxonomy and
# scale it must use. Editing the scale in config.py automatically updates this.
_CATEGORY_LINES = "\n".join(
    f"  - {c}" for c in config.FACET_CATEGORIES
)
_SCALE_LINES = "\n".join(
    f"  {k} = {v}" for k, v in config.SCALE_ANCHORS.items()
)

_SYSTEM = f"""You are a psychometrician building a facet-scoring registry.
For a given FACET NAME you produce metadata used to score conversation turns.

Pick exactly ONE category from this fixed taxonomy:
{_CATEGORY_LINES}

Write a single-sentence DEFINITION of the facet as a measurable trait/behavior
that could be observed in how a person speaks in a conversation.

Set POLARITY:
  - positive : a HIGH score is generally adaptive / desirable
  - negative : a HIGH score is generally maladaptive / undesirable
  - neutral  : descriptive or quantitative, no inherent good/bad valence

Write five RUBRIC ANCHORS describing what each score level looks like FOR THIS
FACET specifically. The generic intensity meaning of each level is:
{_SCALE_LINES}

STRICT ANCHOR RULES (these are mandatory):
  - ASCENDING: scale_1 = the trait ABSENT/lowest, scale_5 = the trait
    DOMINANT/highest. Intensity must increase 1 -> 2 -> 3 -> 4 -> 5.
  - Begin every anchor with its own level number, e.g. "1: ...", "5: ...".
  - Describe increasing amounts of THIS FACET ITSELF. Never describe its
    opposite. (For "Merriness", scale_5 is intense joy — NOT sadness.)
  - Each anchor is ONE short sentence. Put exactly ONE level in each field.
    Do NOT pack multiple levels or a "1=... 2=... 3=..." legend into one field.
Make each anchor concrete and facet-specific (not just "slightly present").

CATEGORY TIE-BREAK RULES (apply in order):
  1. Dispositional personality traits — the Big Five / HEXACO and their facets
     (Openness, Conscientiousness, Extraversion, Agreeableness, Neuroticism,
     Honesty-Humility, Assertiveness, etc.) -> personality, EVEN IF they involve
     thinking, ideas, or social behavior.
  2. Clinical or mental-health conditions and symptoms (depression, anxiety,
     burnout, mania, hysteria, psychoticism, suicidality, substance/drug use,
     trauma/violence exposure) -> safety.
  3. cognitive = abilities, skills, reasoning, memory, or domain knowledge
     (IQ, numerical/spatial reasoning, working memory, anatomy knowledge).
  4. behavioral = habits, action tendencies, and lifestyle behaviors
     (risk-taking, procrastination, caffeine/diet/exercise habits).

EXAMPLES (facet name -> category):
  Openness -> personality
  Neuroticism -> personality
  Depression Symptoms -> safety
  Burnout Symptoms -> safety
  Numerical Reasoning -> cognitive
  Working Memory Index -> cognitive
  Risktaking -> behavioral
  Storytelling proficiency -> linguistic
  I Ching hexagram 12 resonance level -> spiritual
  Quran khatam cycles per year -> spiritual
  Reiki sessions / year -> spiritual

Any facet about religion, faith, spiritual practice, ritual, meditation,
astrology, or esoteric/metaphysical systems -> spiritual (NOT other).

Return ONLY the JSON object."""


# A hand-picked spread that exercises every category + hard edge cases (a
# spiritual practice, a lab value, a quantitative habit). Used by --diverse so the
# sample review is representative instead of 10 near-identical personality traits.
_DIVERSE_SAMPLE = [
    "Risktaking",                      # behavioral / personality
    "Storytelling proficiency",        # linguistic
    "Eye-contact duration",            # pragmatic (non-verbal)
    "Joyfulness",                      # emotion
    "Openness",                        # personality (Big Five)
    "Numerical Reasoning",             # cognitive
    "Depression Symptoms",             # safety / clinical
    "I Ching hexagram 36 resonance level",  # spiritual
    "Caffeine intake (mg/day)",        # behavioral / quantitative edge case
    "Basophil count",                  # safety/biometric edge case (likely neutral)
]


def _user_prompt(facet_name: str) -> str:
    return f'FACET NAME: "{facet_name}"'


def enrich_facet(facet_name: str) -> FacetEnrichment:
    """Draft enrichment metadata for one facet via the local LLM."""
    return chat_structured(
        messages=[
            {"role": "system", "content": _SYSTEM},
            {"role": "user", "content": _user_prompt(facet_name)},
        ],
        schema=FacetEnrichment,
    )


_COLUMNS = [
    "facet_id", "facet_name", "category", "polarity", "definition",
    "scale_1", "scale_2", "scale_3", "scale_4", "scale_5",
]


def _row_record(row, e: FacetEnrichment) -> dict:
    return {
        "facet_id": row.facet_id,
        "facet_name": _clean_text(row.facet_name),
        "category": e.category,
        "polarity": e.polarity,
        "definition": _clean_text(e.definition),
        "scale_1": _clean_text(e.scale_1),
        "scale_2": _clean_text(e.scale_2),
        "scale_3": _clean_text(e.scale_3),
        "scale_4": _clean_text(e.scale_4),
        "scale_5": _clean_text(e.scale_5),
    }


def normalize_enriched_csv(path=None) -> int:
    """Collapse whitespace in all text columns of an existing enriched CSV.

    Idempotent one-time fix for rows written before `_row_record` cleaned text
    (the running process loaded the old code). Guarantees single-line anchors.
    """
    path = path or config.ENRICHED_FACETS_CSV
    df = pd.read_csv(path, dtype=str).fillna("")
    for c in ["facet_name", "definition",
              "scale_1", "scale_2", "scale_3", "scale_4", "scale_5"]:
        df[c] = df[c].map(_clean_text)
    df.to_csv(path, index=False)
    return len(df)


def enrich_dataframe(df: pd.DataFrame, *, verbose: bool = True) -> pd.DataFrame:
    """Enrich every row of a cleaned facet frame in memory (used for samples)."""
    records: list[dict] = []
    n = len(df)
    for i, row in enumerate(df.itertuples(index=False), start=1):
        e = enrich_facet(row.facet_name)
        records.append(_row_record(row, e))
        if verbose:
            print(f"[{i:>4}/{n}] {row.facet_id} {row.facet_name[:42]:42} -> "
                  f"{e.category}/{e.polarity}")
    return pd.DataFrame.from_records(records)


def enrich_to_csv(df: pd.DataFrame, out_path, *, resume: bool = True) -> pd.DataFrame:
    """Enrich every facet, writing each row to `out_path` as it completes.

    Crash-safe and resumable: on restart, facets already present in `out_path`
    are skipped. This is what makes the ~1 hr / 399-facet (or future 5000-facet)
    bulk run safe to background — a crash at row 350 loses nothing.
    """
    import csv

    done_ids: set[str] = set()
    if resume and out_path.exists():
        prev = pd.read_csv(out_path, dtype=str)
        done_ids = set(prev["facet_id"].tolist())
        print(f"resuming: {len(done_ids)} facets already enriched in {out_path.name}")

    todo = df[~df["facet_id"].isin(done_ids)]
    n = len(todo)
    write_header = not out_path.exists()
    with open(out_path, "a", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=_COLUMNS)
        if write_header:
            writer.writeheader()
        for i, row in enumerate(todo.itertuples(index=False), start=1):
            try:
                e = enrich_facet(row.facet_name)
            except LLMError as err:
                # A facet the strict validator couldn't satisfy after retries:
                # skip it (left missing in the CSV) and keep going. The QC pass
                # regenerates missing facet_ids, so the run never aborts.
                print(f"[{i:>4}/{n}] {row.facet_id} {row.facet_name[:40]:40} -> "
                      f"SKIPPED ({type(err).__name__})", flush=True)
                continue
            writer.writerow(_row_record(row, e))
            fh.flush()  # durable after every row
            print(f"[{i:>4}/{n}] {row.facet_id} {row.facet_name[:40]:40} -> "
                  f"{e.category}/{e.polarity}", flush=True)

    return pd.read_csv(out_path, dtype=str)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="LLM-enrich the facet registry.")
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--sample", type=int, metavar="N", help="enrich first N, print only")
    g.add_argument("--diverse", action="store_true",
                   help="enrich a curated cross-category sample, print only")
    g.add_argument("--all", action="store_true", help="enrich all, write enriched CSV")
    args = ap.parse_args()

    clean = load_clean_facets()

    if args.sample or args.diverse:
        if args.diverse:
            subset = clean[clean["facet_name"].isin(_DIVERSE_SAMPLE)].copy()
            # preserve the curated order
            order = {name: i for i, name in enumerate(_DIVERSE_SAMPLE)}
            subset = subset.sort_values(
                "facet_name", key=lambda s: s.map(order)
            )
        else:
            subset = clean.head(args.sample)
        enriched = enrich_dataframe(subset)
        pd.set_option("display.max_colwidth", 60)
        print("\n=== SAMPLE ENRICHMENT ===")
        for _, r in enriched.iterrows():
            print(f"\n{r['facet_id']}  {r['facet_name']}")
            print(f"  category : {r['category']}   polarity: {r['polarity']}")
            print(f"  definition: {r['definition']}")
            for k in range(1, 6):
                print(f"    {k}: {r[f'scale_{k}']}")
    else:
        enriched = enrich_to_csv(clean, config.ENRICHED_FACETS_CSV, resume=True)
        print(f"\nwrote {config.ENRICHED_FACETS_CSV} ({len(enriched)} rows)")
