# PROGRESS

Living checklist for the Conversation Facet Evaluator. Updated at the end of each phase.

## Locked decisions
- **Language:** Python 3.14 (venv). *Originally 3.11, but Homebrew's 3.11 **and**
  3.12 bottles are broken by a `pyexpat`/`libexpat` linkage bug on macOS 26 — they
  can't even import XML, so pip won't bootstrap. Brew's 3.14 works and has full
  arm64 cp314 wheels for torch/chromadb/sentence-transformers (verified via
  `pip --dry-run`). Defensible interview note.*
- **Scoring LLM:** Qwen2.5-**3B**-Instruct `q4_K_M` via Ollama (7B swapped hard on
  8 GB; switched via a one-line config change — model-agnostic design. Can re-run
  final scoring on 7B via `OLLAMA_MODEL` override, e.g. on Colab.)
- **Embeddings:** `BAAI/bge-small-en-v1.5` (sentence-transformers)
- **Vector store:** ChromaDB (persistent, local)
- **Structured output:** pydantic schema + Ollama JSON mode
- **Score scale:** 5 ordered ints, 1=absent … 5=very strong; per-facet `polarity` flips meaning
- **Facets:** 398 raw in `data/facets_raw.csv`; architecture targets ≥5000 with no code changes

## Phases
- [x] **Phase 0 — Setup & scaffold**
  - [x] Confirm hardware (Apple M3, 8 GB RAM) + score scale (1–5 intensity)
  - [x] Repo structure, `requirements.txt`, `.gitignore`, `config.py`, stub README
  - [x] Python 3.14 venv created (3.11/3.12 bottles broken — see note above)
  - [x] Dependencies installed + all imports verified
  - [ ] Ollama installed + Qwen2.5-7B pulled *(deferred to Phase 3, where it's first needed)*
- [~] **Phase 1 — Clean & enrich facet registry** → `data/facets_enriched.csv`
  - [x] Taxonomy locked: 8 categories + `other` (linguistic, pragmatic, emotion,
        personality, behavioral, cognitive, safety, spiritual)
  - [x] `clean.py` — strip leading enums/trailing colons, dedupe → 399 clean facets
  - [x] `schema.py` (FacetEnrichment) + `llm.py` (Ollama structured-output wrapper)
  - [x] `enrich.py` — per-facet hybrid LLM drafting (category/definition/polarity/anchors)
  - [x] Qwen2.5-7B pulled; Ollama running as a brew service (persistent)
  - [x] Diverse 10-facet sample approved; prompt refined with tie-break rules +
        few-shot anchors (personality / safety-clinical / spiritual now correct)
  - [x] Bulk-generated all 399 on 3B → `data/facets_enriched.csv`
  - [~] **Anchor QC** (reviewer caught 3 defect classes):
    - [x] `schema.normalize_anchors` — reorder by leading level number when
          present, trust field order when absent, reject legend dumps / dup-skip
          levels / ambiguous mixes (returns None → regenerate)
    - [x] `schema` validator auto-applies it at generation (ordering guaranteed
          going forward); strict prompt: ascending, describe-facet-not-opposite,
          no `N=` legends; bulk run resilient (skip+continue on failure)
    - [x] `registry_qc.py` — deterministic normalize + detect malformed +
          embedding-based inversion detection + regenerate malformed/missing/
          inverted + verify all ascend
    - [x] Ran `registry_qc --fix`: regenerated 29 (10 malformed ∪ 20 inversion
          suspects incl. F0009 Merriness) + 7 more generic-echo/duplicate rows
          (incl. F0129) caught after hardening the detector with `(N)`-marker and
          duplicate-anchor rejection
    - [x] **VERIFIED:** 399 rows, ids F0001..F0399, 0 fail ascend/format,
          no embedded newlines in anchors
  - **Phase 1 COMPLETE** ✅ (registry reviewed + repaired)
- [~] **Phase 2 — Vector store** (`vectorstore.py`)
  - [x] bge-small-en-v1.5 embeddings (normalized, cosine), load-once embedder
  - [x] ChromaDB persistent store; upsert by stable facet_id; full rubric in metadata
  - [x] `query_facets(turn, k)` — bge query-instruction, returns scored facets
  - [x] Smoke-tested on partial registry: relevant facets retrieved (risk→Risktaking,
        joy→Merriness/Enthusiasm)
  - [x] **Final full build on clean 399** (`--rebuild`): verified count=399, ids
        exactly F0001..F0399, no stale/missing
  - [x] Diverse query check: spiritual retrieval excellent; hostile/anxious turns
        have noisy top-8 but **strong recall@30** (Hostility/Passive-Aggressive/
        Coarseness for anger; Depression/Fearfulness/Negative-Affect/Burnout for
        anxiety all within the router's top-30 window)
  - **Phase 2 COMPLETE** ✅
- [x] **Phase 3 — Scorer** (`scorer.py`)
  - [x] `score(turn, facet_batch) → [{facet_id, facet_name, score, confidence}]`
  - [x] Rubrics pulled from store metadata (no CSV re-read); Qwen JSON mode;
        `FacetScore`/`BatchScores` pydantic schemas (score 1-5, confidence 0-1)
  - [x] Coverage loop: re-requests omitted facet_ids (small models drop some);
        missing-after-retries surfaced as score=None, never faked
  - [x] Live demo on a hostile turn: absent facets→1, irritability→5,
        coarseness→4; confidence populated + varies (0.8-0.95)
  - Known (3B): occasional mis-score (e.g. retrieved-but-opposite "Avoiding"→4);
        confidence uncalibrated → Phase 7. 7B via OLLAMA_MODEL override for finals.
  - **Phase 3 COMPLETE** ✅
- [x] **Phase 4 — Router + pipeline** (`router.py`, `pipeline.py`)
  - [x] Router: top-K ∪ **safety floor** (top-N safety facets by similarity, even
        below top-K) — verified it rescues Depression Symptoms (#33 global) etc.;
        each facet tagged via=topk/safety_floor/both
  - [x] `query_facets(where=...)` metadata filter for the in-category safety query
  - [x] Pipeline: turn → route → batch → score → assemble turn×facet matrix → JSON
        (`output/scores/<id>.json`); per-turn n_routed/n_scored/n_missing +
        rescued_by_safety_floor; matrix display with ·=not-routed X=None
  - [x] Ran on demo_mixed_001 (5 turns): matrix tracks the emotional arc
        (anxious→hostile→grateful), 0 missing, safety floor scored end-to-end
  - **Phase 4 COMPLETE** ✅
- [~] **Phase 5 — 50+ conversations + scores** (zip; REQUIRED deliverable)
  - [x] Authored 54 conversations (`data/build_conversations.py`) + demo = **55**,
        6 each across polite/hostile/anxious/manipulative/joyful/evasive/safety-
        risky/neutral + 7 mixed-arc; safety-risky each target a distinct safety facet
  - [x] Resumable batch runner (`pipeline --all`, skips already-scored)
  - [x] Smoke-tested 3 (safety/hostile/neutral): floor works (Depression 4-5),
        case-appropriate, 0 missing
  - [x] **Robustness fix** (run crashed at 10/55): 3B sometimes emits confidence
        on the 1-5 scale (e.g. 5) → `FacetScore` now coerces out-of-range
        confidence (/5) and score (clamp 1-5) instead of rejecting; `score()`
        catches `LLMError` per batch so one bad batch can't kill the run
  - [x] **Full run complete: 55/55 scored**, 189 turns, 7,559 facet-scores,
        **100% coverage (0 missing)**, safety floor 1,889 rescues (7.3/safety-turn)
  - [x] Deliverable zip built (`report --zip`): 55 convs + 55 matrices + manifest
  - **Phase 5 COMPLETE** ✅
- [~] **Phase 6 — README + docs + GitHub** (architecture, scaling, run steps; REQUIRED)
  - [x] README drafted during the Phase 5 run (architecture, ≥5000 scaling,
        no-one-shot batching, safety floor, model-agnostic, None handling,
        known-issues/future-work); coverage-stats placeholder pending run
  - [x] `report.py` built + tested: coverage stats, auto-fill README block
        (sentinel-bounded), deliverable zip. Run `report --all` when scores land.
  - [x] Coverage stats auto-filled + 2 sample matrices (joyful, safety/depression)
  - [x] Local commit (no push, per user) — origin=kantrolv/Ahoum, gh authed
  - **Phase 6 README COMPLETE** ✅ (user pushes manually)
- [x] **Phase 7 (brownie) — Streamlit UI** (`app/streamlit_app.py`): browse scored
      conversations as a colour-coded turn×facet matrix + live-score a typed turn.
      Boot-tested (HTTP 200), matrix/styler validated. Run: `streamlit run app/streamlit_app.py`
- [ ] **Phase 7+ — Brownie:** confidence refinement → Streamlit UI → Docker/compose → deploy

## How to run (so far)
```bash
# Phase 0 env (already created):
/opt/homebrew/opt/python@3.14/bin/python3.14 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Open questions / notes
- Ollama not yet installed; needed for Phase 3. Will install + pull the model then.
- Phase 1 category/definition generation method to be approved before bulk run.
