"""Central configuration for the conversation facet evaluator.

Every tunable lives here (or in a `.env` that overrides these defaults) so that
no module hardcodes a model name, path, or threshold. This is also where the
*scale* is defined once and reused everywhere, which matters for the 5000-facet
scaling goal: facets are data, but the scoring *contract* (1-5 ordinal) is shared
infrastructure.
"""

from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()  # read a local .env if present; real env vars win over defaults


def _get(key: str, default: str) -> str:
    return os.environ.get(key, default)


# --- Paths ---------------------------------------------------------------
# Resolve everything relative to the repo root so the code runs from anywhere.
REPO_ROOT: Path = Path(__file__).resolve().parents[2]
DATA_DIR: Path = REPO_ROOT / "data"
OUTPUT_DIR: Path = REPO_ROOT / "output"

RAW_FACETS_CSV: Path = DATA_DIR / "facets_raw.csv"
ENRICHED_FACETS_CSV: Path = DATA_DIR / "facets_enriched.csv"

CHROMA_DIR: Path = REPO_ROOT / "chroma_db"
CHROMA_COLLECTION: str = "facets"


# --- Models --------------------------------------------------------------
# Scoring LLM: open-weights, <=16B, served locally by Ollama. The pipeline is
# model-agnostic — every LLM call goes through llm.chat_structured(), so swapping
# the model is THIS ONE LINE (or the OLLAMA_MODEL env var), no code changes.
# 3B chosen because 7B swaps heavily on 8 GB RAM; can re-run on 7B (e.g. Colab)
# for the final outputs by overriding OLLAMA_MODEL — nothing else changes.
OLLAMA_MODEL: str = _get("OLLAMA_MODEL", "qwen2.5:3b-instruct-q4_K_M")
OLLAMA_HOST: str = _get("OLLAMA_HOST", "http://localhost:11434")

# Embedding model for the registry / router (small, fast, CPU-friendly).
EMBED_MODEL: str = _get("EMBED_MODEL", "BAAI/bge-small-en-v1.5")


# --- Scoring scale (single source of truth) ------------------------------
# Five ordered integers. `polarity` per facet decides whether a high score is
# "good" or "bad"; the scale itself is pure intensity so it generalizes to any
# number of facets without bespoke anchors.
SCALE_MIN: int = 1
SCALE_MAX: int = 5
SCALE_ANCHORS: dict[int, str] = {
    1: "absent / not present",
    2: "slight / faint trace",
    3: "moderate / clearly present",
    4: "strong / prominent",
    5: "very strong / dominant",
}


# --- Router / scorer knobs ----------------------------------------------
# Top-K facets retrieved per turn by the router (Phase 4).
ROUTER_TOP_K: int = int(_get("ROUTER_TOP_K", "30"))
# Safety floor: always also include the top-N safety-category facets by
# similarity, even if they fall below the global top-K. A missed safety facet is
# never scored at all, so safety recall gets a guaranteed minimum independent of
# how much other content crowds the global ranking. (e.g. anxious turn: Depression
# Symptoms is ~#33 globally but a top safety facet — the floor rescues it.)
ROUTER_SAFETY_FLOOR: int = int(_get("ROUTER_SAFETY_FLOOR", "12"))
SAFETY_CATEGORY: str = "safety"
# Facets per LLM call (Phase 3). Small batches keep the prompt focused and the
# JSON parseable; this is the antithesis of a one-shot all-facets prompt.
SCORE_BATCH_SIZE: int = int(_get("SCORE_BATCH_SIZE", "12"))
# If True, score every facet in the registry; if False, only router top-K.
SCORE_ALL_FACETS: bool = _get("SCORE_ALL_FACETS", "false").lower() == "true"


# --- Facet categories (used for enrichment in Phase 1) -------------------
# Coarse, stable taxonomy. The four PDF domains (linguistic/pragmatic/emotion/
# safety) + personality/behavioral/cognitive + spiritual (the data has a large
# faith/practice cluster). "other" is a fallback so the classifier is never forced
# into a wrong bucket. Coarse on purpose: stable as the registry grows to 5000.
FACET_CATEGORIES: tuple[str, ...] = (
    "linguistic",   # language form: grammar, vocabulary, brevity, spelling
    "pragmatic",    # communication & interpersonal: listening, leadership, negotiation
    "emotion",      # affect & mood: happiness, hostility, blissfulness
    "personality",  # dispositional traits: Big Five/HEXACO, openness, assertiveness
    "behavioral",   # habits/tendencies/lifestyle actions: risk-taking, caffeine intake
    "cognitive",    # reasoning, memory, knowledge/skills: numerical reasoning, IQ
    "safety",       # risk/clinical/biometric: depression, drug use, lab values
    "spiritual",    # faith & practice: I Ching, dhikr, Quran, Reiki, astrology
    "other",        # fallback for genuine misfits
)

# Polarity: does a HIGH score read as generally adaptive, maladaptive, or neither?
FACET_POLARITIES: tuple[str, ...] = ("positive", "negative", "neutral")
