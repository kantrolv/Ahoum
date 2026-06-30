# Conversation Facet Evaluator

A production-shaped benchmark that scores **every turn** of a multi-turn
conversation against a large registry of psychological / linguistic / behavioral
**facets** (399 today) on a **5-level ordered scale** (1 = absent … 5 = dominant),
with a per-score **confidence**.

The system is built so the registry can grow to **≥ 5000 facets with no code
changes** — facets are *data*, not code. It uses only **open-weights models
(≤ 16 B)** served locally by Ollama, and never sends the whole facet list to the
model in one shot.

> Status & phase-by-phase log: [`PROGRESS.md`](PROGRESS.md).

---

## What it does

Given a conversation like:

```json
{ "conversation_id": "demo_mixed_001",
  "turns": [ {"speaker": "Client", "text": "My heart's racing and I can't stop shaking..."}, ... ] }
```

it produces, for **each turn**, a score across the facets relevant to that turn:

```json
{ "facet_id": "F0061", "facet_name": "Depression Symptoms",
  "category": "safety", "via": "safety_floor",
  "similarity": 0.474, "score": 4, "confidence": 0.8 }
```

and assembles a **turn × facet matrix** per conversation
(`output/scores/<id>.json`).

---

## Architecture

Two halves: an **offline registry build** (run once; rebuilt only when facets
change) and an **online scoring pipeline** (run per conversation).

### Offline — build the registry (facets become data)

```
data/facets_raw.csv  (399 raw facet names, messy)
      │  clean.py        strip artifacts (trailing colons, "800." enums), dedupe,
      │                  assign stable ids F0001…F0399
      │  enrich.py       LLM drafts per facet: category, definition, polarity,
      │                  and five rubric anchors (scale_1..scale_5)
      │  registry_qc.py  deterministic ordering fix + regenerate malformed /
      │                  inverted rows; verify every row ascends 1→5
      ▼
data/facets_enriched.csv
      │  vectorstore.py  embed "name: definition" with bge-small (cosine),
      ▼                  store in ChromaDB with the full rubric in metadata
ChromaDB collection  (399 vectors; add rows to scale — no re-embed of existing)
```

### Online — score a conversation (per turn)

```
conversation ──► for each turn:

  ┌─ ROUTER (router.py) ───────────────────────────────────────────────┐
  │  embed(turn) ─► ChromaDB ANN search                                 │
  │     • global top-K facets                       ← recall            │
  │     • ∪ top-N safety-category facets             ← SAFETY FLOOR      │
  │  → ~30–40 relevant facets, rubrics already attached (store metadata)│
  └────────────────────────────────────────────────────────────────────┘
                              │  split into batches of SCORE_BATCH_SIZE (12)
  ┌─ SCORER (scorer.py) ──────▼─────────────────────────────────────────┐
  │  turn + small batch(rubrics) ─► Qwen via Ollama (JSON mode)          │
  │  ─► pydantic-validated {facet_id, score 1–5, confidence 0–1}         │
  │  coverage loop re-requests any facet the model omitted              │
  └─────────────────────────────────────────────────────────────────────┘
                              │
                              ▼
        turn × facet score matrix  ──►  output/scores/<id>.json
```

Everything funnels through one LLM wrapper (`llm.chat_structured`) and one config
module (`config.py`).

---

## Why this scales to ≥ 5000 facets (the core design argument)

**The registry is data; the pipeline is fixed-size work.**

- **Adding facets = adding rows** to `facets_enriched.csv` and upserting vectors.
  No pipeline code changes, no prompt edits. Existing vectors are never recomputed
  (upsert by stable `facet_id`).
- **The router bounds the work per turn.** A turn is matched to the store with a
  *single* approximate-nearest-neighbour lookup; we then score only the ~30–40
  facets it returns. So a turn costs the *same* whether the registry holds 399 or
  5000 facets — the registry grows, the per-turn work does not.
- **The scorer only ever sees a bounded batch** (≤ `SCORE_BATCH_SIZE`). Prompt
  size and quality stay constant regardless of registry size.

### No "one-shot all-facets prompt" — and why

Stuffing 399 (let alone 5000) facets into a single prompt fails three ways: it
blows the context window, it collapses scoring quality (the model can't attend to
hundreds of rubrics at once), and it doesn't scale. Instead we **retrieve then
batch**: the router narrows to what's relevant, and the scorer grades a dozen
facets at a time against their explicit rubrics. This is the central constraint
of the design, not an optimization.

---

## The safety floor (guaranteed safety recall)

Scoring only happens on *routed* facets, so **a safety facet that falls below the
global top-K cutoff is never scored at all** — a silent miss exactly where a miss
is most costly. The router therefore unions the global top-K with the **top-N
safety-category facets by similarity** (`ROUTER_SAFETY_FLOOR`, default 12).

Concretely: for a panic-attack turn, *Depression Symptoms* ranks ~#33 globally
(crowded out by somatic facets) — outside top-30 — but it's a top safety facet,
so the floor pulls it into the scored batch. Each routed facet is tagged
`via = topk | safety_floor | both` so the rescue is auditable.

---

## Scoring scale

Five ordered integers, the **same scale for every facet** (so it generalizes to
any number of facets without bespoke anchors):

| score | meaning |
|---|---|
| 1 | absent / not present |
| 2 | slight / faint trace |
| 3 | moderate / clearly present |
| 4 | strong / prominent |
| 5 | very strong / dominant |

Each facet also has a **polarity** (`positive` / `negative` / `neutral`) recording
whether a *high* score reads as adaptive, maladaptive, or merely descriptive — the
scale stays pure intensity, polarity carries the valence. Facets are tagged with
one of **8 categories** (`linguistic, pragmatic, emotion, personality, behavioral,
cognitive, safety, spiritual`) plus an `other` fallback, used as routing/analysis
metadata.

---

## Never fake a score (honest `None` handling)

Small models occasionally return JSON that silently omits a few facets. We do
**not** paper over this with a default value:

1. The scorer detects omitted `facet_id`s and **re-requests only those** (the
   coverage loop, up to `coverage_rounds` extra calls).
2. Anything still unscored is returned with `score = None` and surfaced in the
   output (`n_missing`, and an `X` in the matrix display) — **never silently
   dropped, never faked**.

If the model server is down or a rubric can't be produced, the code **fails
loudly** rather than emitting a plausible-looking fake.

---

## Model-agnostic by design

Every LLM call funnels through `llm.chat_structured`, and the model name lives
only in `config.py` / the `OLLAMA_MODEL` env var. **Swapping the scoring model is
a one-line change** — no pipeline code is touched.

We default to **Qwen2.5-3B-Instruct (Q4)** because the 7B swaps heavily on an
8 GB machine. To run final scoring on the stronger 7B (e.g. on Colab or a bigger
box), set just:

```bash
OLLAMA_MODEL=qwen2.5:7b-instruct-q4_K_M
```

The same indirection means any open-weights ≤ 16 B model Ollama serves (Qwen,
Llama, Mistral, Gemma…) drops in without edits.

---

## Stack

| Concern | Choice |
|---|---|
| Language | Python 3.14 (see note below) |
| Scoring LLM | Qwen2.5-**3B**-Instruct (Q4) via **Ollama** — open-weights, ≤ 16 B |
| Embeddings | `BAAI/bge-small-en-v1.5` (sentence-transformers), cosine |
| Vector store | ChromaDB (persistent, local) |
| Structured output | pydantic + Ollama JSON mode |
| Registry | pandas (CSV → enriched CSV) |
| API / UI / container | FastAPI · Streamlit · Docker *(brownie phases)* |

> **Why Python 3.14, not 3.11?** Homebrew's `python@3.11` and `@3.12` bottles have
> a broken `pyexpat`/`libexpat` linkage on macOS 26 (can't import XML → `pip`
> won't bootstrap). Brew's 3.14 works and has full arm64 wheels for every
> dependency (verified by dry-run).

---

## Quickstart

```bash
# 1. Environment
python3.14 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# 2. Local model server (open-weights, ≤16B)
brew install ollama && brew services start ollama
ollama pull qwen2.5:3b-instruct-q4_K_M

# 3. Build the registry (once)  — ~45 min on 3B; resumable
python -m facet_eval.clean --write
python -m facet_eval.enrich --all
python -m facet_eval.registry_qc --fix

# 4. Build the vector store
python -m facet_eval.vectorstore --build --rebuild

# 5. Score a conversation, or the whole set (resumable)
python -m facet_eval.pipeline data/conversations/demo_mixed_001.json
python -m facet_eval.pipeline --all
```

Per-module entry points double as smoke tests, e.g.:

```bash
python -m facet_eval.router          # show safety-floor rescues for a sample turn
python -m facet_eval.scorer          # live-score a hostile turn (high/low facets)
python -m facet_eval.vectorstore --query "I feel hopeless and exhausted" --k 8
```

---

## Configuration (`src/facet_eval/config.py`)

Single source of truth; overridable via `.env` (see `.env.example`).

| Setting | Default | Meaning |
|---|---|---|
| `OLLAMA_MODEL` | `qwen2.5:3b-instruct-q4_K_M` | scoring model (one-line swap) |
| `EMBED_MODEL` | `BAAI/bge-small-en-v1.5` | router/registry embeddings |
| `ROUTER_TOP_K` | `30` | global facets retrieved per turn |
| `ROUTER_SAFETY_FLOOR` | `12` | top-N safety facets always included |
| `SCORE_BATCH_SIZE` | `12` | facets per LLM scoring call |
| `SCALE_MIN` / `SCALE_MAX` | `1` / `5` | the ordinal scale |

---

## Repository layout

```
data/
  facets_raw.csv            399 raw facet names
  facets_enriched.csv       enriched registry (id, category, polarity, definition, scale_1..5)
  build_conversations.py    authors the 54 evaluation conversations
  conversations/            one JSON per conversation (the inputs)
src/facet_eval/
  config.py        all tunables + the scale (single source of truth)
  schema.py        pydantic contracts + anchor normalization/validation
  clean.py         raw → cleaned, de-duplicated facet list
  enrich.py        LLM enrichment of each facet
  registry_qc.py   audit/repair anchors (ordering, malformed, inversions)
  llm.py           Ollama structured-output wrapper (retries + validation)
  vectorstore.py   bge-small embeddings + ChromaDB (build / query)
  router.py        top-K + safety floor
  scorer.py        batched, coverage-checked turn scorer
  pipeline.py      conversation → matrix → JSON (+ resumable batch runner)
output/scores/     per-conversation score matrices (JSON)
```

---

## Results & deliverables

- **Registry:** 399 facets, all anchors verified ascending 1→5, no duplicates.
- **Conversations:** 55, spanning every named case (polite, hostile, anxious,
  manipulative, joyful, evasive, safety-risky, neutral) + mixed-arc.
- **Score matrices:** `output/scores/*.json`, zipped as the deliverable.

<!-- COVERAGE_STATS_START -->
> **Coverage statistics** (model `qwen2.5:3b-instruct-q4_K_M`, top-K 30, safety floor 12):
> - conversations scored: **55/55** · turns scored: **189** · facet-scores produced: **7,559**
> - turns with full coverage (0 missing): **100.0%** · total `None` after retries: **0**
> - safety-floor rescues across the set: **1889** (mean **7.3** per safety-risky turn)
> - score distribution (1→5): 1:5360 · 2:575 · 3:584 · 4:703 · 5:337 · mean score **1.69** · mean confidence **0.78**
<!-- COVERAGE_STATS_END -->

### Sample matrices (scores track the content)

A **joyful** conversation — joy facets dominate every turn (`·` = not routed that
turn):

```
                       turn:  0  1  2
F0088 High-spiritedness        5  5  5
F0257 Joyfulness               5  4  5
F0115 Happiness                4  4  5
F0332 Vivacity                 5  ·  5
F0009 Merriness                4  5  4
```

A **safety-risky** (depression) conversation — distress facets rise, and the
**safety floor** rescues `F0061 Depression Symptoms` (below the global top-K) so
it gets scored:

```
                       turn:  0  1  2     (turn 1 = supportive listener)
F0084 Depression: sadness      5  ·  5
F0061 Depression Symptoms      4  ·  5   ← rescued by safety floor
F0103 Burnout Symptoms         5  ·  2
F0034 Desperation              1  ·  5
F0217 Withdrawnness            1  ·  5
F0396 Peacefulness             ·  5  ·   (the listener's calming turn)
```

Full matrices for all 55 conversations are in `output/scores/` and the
deliverable zip.

---

## Known issues & future work

These are honest limitations of the *current* configuration, not architectural
flaws — each has a clear mitigation path.

- **3B scoring noise.** The 3B occasionally over-scores a marginal retrieved facet
  (e.g. `Spelling Accuracy 5` on a hostile turn) or mis-applies a rubric. The
  *right* facets reliably light up, but precision on borderline facets is rough.
  *Mitigation:* the one-line `OLLAMA_MODEL` swap to the 7B (e.g. on Colab) reduces
  this materially; the architecture is unchanged.
- **bge-small retrieval precision.** `bge-small` gives strong *recall* (relevant
  facets land in the top-30 window) but noisy *precision* at the very top for
  emotionally charged turns (a spiritual-ritual facet once topped an anxiety
  query). This is by design tolerable — the router owns recall, the scorer owns
  precision (it correctly scores irrelevant retrieved facets as 1). *Mitigation:*
  `bge-base`/`bge-large` or richer facet text embeddings for sharper top-K.
- **Confidence calibration.** The model self-reports confidence (0–1); values are
  populated and vary, but are not calibrated. *Future work:* derive confidence
  from token logprobs, self-consistency across samples, or anchor-distance, and
  validate against held-out human labels.
- **Registry artifacts.** A handful of raw entries are section headers
  (e.g. "HEXACO … Facets") or unscoreable lab values (e.g. "Basophil count"); they
  remain in the registry tagged `other`/`neutral` and the router simply rarely
  retrieves them — harmless, and avoids lossy deletion.
- **Throughput.** Local 3B scoring of 55 conversations takes a few hours.
  *Future work:* parallelize across Ollama workers, cache per-(turn,facet) scores,
  or batch turns.

### Brownie-point phases (planned)

Per-score confidence refinement · Streamlit UI · FastAPI service · Dockerised
baseline (app + Ollama via docker-compose) · optional deployed URL.

---

## Design decisions (interview notes)

- **Registry-driven, not prompt-driven** — facets live in data so the system
  scales by adding rows, satisfying the ≥5000-without-rewrite requirement.
- **Retrieve-then-batch** — bounded per-turn work and bounded prompts; the
  explicit rejection of a one-shot prompt.
- **Safety as a first-class recall guarantee** — the safety floor, because a
  missed safety facet is never scored.
- **pydantic at every boundary** — the model returns free text; the pipeline
  needs typed data, so we constrain (JSON schema) and validate (fail loud).
- **One config, one LLM wrapper** — model-agnostic, no hardcoded names/paths.
- **Honest failure** — `None` over fake scores; loud errors over silent defaults.
```
