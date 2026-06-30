"""Phase 3 — score a conversation turn against a small batch of facets.

`score(turn_text, facet_batch) -> [{facet_id, score, confidence}]`

The whole design hinges on this being *batched*: we never put all 399 (or 5000)
facets in one prompt. The router hands us a small, relevant batch; we send the
turn plus those facets' rubrics to the local LLM in JSON mode and validate the
result with pydantic. Adding facets never changes this function — it always sees
a bounded batch, so prompt size and quality stay constant as the registry grows.

The facet batch is exactly what the vector store returns (name, definition, and
the five rubric anchors live in the store's metadata), so the scorer reads its
rubrics straight from retrieval — no separate CSV lookup at scoring time.

Run a live demo:
    python -m facet_eval.scorer
"""

from __future__ import annotations

from . import config
from .llm import LLMError, chat_structured
from .schema import BatchScores
from .vectorstore import get_facets, query_facets

_SYSTEM = """You are a precise conversation analyst. You are given ONE
conversation turn and a batch of behavioral/psychological FACETS, each with a
1-5 rubric. For EVERY facet, decide how strongly THIS turn exhibits it.

Rules:
  - Score 1-5 strictly by the facet's own rubric: 1 = the facet is absent /
    not evidenced in the turn; 5 = it is dominant. Use the anchor texts.
  - Judge ONLY from evidence in the turn. If the turn says nothing relevant to
    a facet, its score is 1 (absent) — do not invent evidence.
  - confidence is a DECIMAL between 0.0 and 1.0 (e.g. 0.4, 0.85) — NOT the 1-5
    score scale. It is how sure you are of THIS score given how much the turn
    reveals about the facet. Little/ambiguous evidence -> low (near 0).
  - Return a score for EVERY facet_id provided, using the exact ids given.
Return ONLY the JSON object."""


def _format_batch(facet_batch: list[dict]) -> str:
    """Render the batch as compact, rubric-bearing blocks for the prompt."""
    blocks = []
    for f in facet_batch:
        anchors = "\n".join(
            f"      {k}: {f[f'scale_{k}']}" for k in range(1, 6)
        )
        blocks.append(
            f"- {f['facet_id']} — {f['facet_name']}\n"
            f"    definition: {f['definition']}\n"
            f"    rubric:\n{anchors}"
        )
    return "\n".join(blocks)


def _score_call(turn_text: str, facets: list[dict]) -> dict:
    """One LLM call: score `facets`, return {facet_id: FacetScore} it covered."""
    user = (
        f"CONVERSATION TURN:\n\"\"\"\n{turn_text}\n\"\"\"\n\n"
        f"FACETS TO SCORE ({len(facets)}):\n{_format_batch(facets)}\n\n"
        f"Return JSON: a 'scores' array with one object "
        f"{{facet_id, score, confidence}} for EACH of the {len(facets)} facets."
    )
    result = chat_structured(
        messages=[
            {"role": "system", "content": _SYSTEM},
            {"role": "user", "content": user},
        ],
        schema=BatchScores,
    )
    wanted = {f["facet_id"] for f in facets}
    return {s.facet_id: s for s in result.scores if s.facet_id in wanted}


def score(turn_text: str, facet_batch: list[dict], *, coverage_rounds: int = 2) -> list[dict]:
    """Score one turn against one batch of facets, ensuring full coverage.

    Small models sometimes return JSON that silently omits a few facets. We
    detect the omissions and re-request ONLY the missing facets (up to
    `coverage_rounds` extra calls) so every facet gets a score.

    Args:
        turn_text: the conversation turn to evaluate.
        facet_batch: facets to score; each dict must carry facet_id, facet_name,
            definition, and scale_1..scale_5 (exactly the vector-store metadata).

    Returns:
        One dict per input facet: {facet_id, facet_name, score, confidence}, in
        `facet_batch` order. Any facet still unscored after the retries is
        returned with score=None (surfaced, never silently dropped or faked).
    """
    if not facet_batch:
        return []

    scored: dict = {}
    pending = facet_batch
    for _ in range(coverage_rounds + 1):
        if not pending:
            break
        try:
            scored.update(_score_call(turn_text, pending))
        except LLMError as err:
            # A batch the model couldn't return valid JSON for after retries.
            # Leave its facets unscored (-> None, surfaced) rather than crash the
            # whole conversation/run. Never fake, never abort.
            print(f"    batch scoring failed, leaving {len(pending)} unscored: "
                  f"{type(err).__name__}", flush=True)
            break
        pending = [f for f in facet_batch if f["facet_id"] not in scored]

    out: list[dict] = []
    for f in facet_batch:
        s = scored.get(f["facet_id"])
        out.append(
            {
                "facet_id": f["facet_id"],
                "facet_name": f.get("facet_name", ""),
                "score": s.score if s else None,
                "confidence": s.confidence if s else None,
            }
        )
    return out


def _demo() -> None:
    """Score a hostile turn against a mix of clearly-high and clearly-low facets."""
    turn = ("Shut up, you idiot. I'm sick of your incompetence and you'll regret "
            "crossing me — you're pathetic and useless.")

    # Relevant facets (should score HIGH) come from retrieval; we add a few
    # obviously-absent facets (should score LOW) so we can see the scale work.
    relevant = query_facets(turn, top_k=8)
    absent = get_facets(["F0351", "F0257", "F0208", "F0138"])  # numeracy, joy, storytelling, I Ching
    batch = relevant + [f for f in absent
                        if f["facet_id"] not in {r["facet_id"] for r in relevant}]

    print(f'TURN: "{turn}"\n')
    print(f"Scoring a batch of {len(batch)} facets "
          f"({len(relevant)} retrieved + {len(batch) - len(relevant)} planted-absent)\n")

    results = score(turn, batch)
    meta = {f["facet_id"]: f for f in batch}
    results.sort(key=lambda r: (-(r["score"] or 0), -(r["confidence"] or 0)))

    print(f"{'score':>5} {'conf':>5}  facet")
    print("-" * 64)
    for r in results:
        f = meta[r["facet_id"]]
        sc = r["score"]
        anchor = f.get(f"scale_{sc}", "") if sc else ""
        print(f"{str(sc):>5} {r['confidence']!s:>5}  {r['facet_id']} {f['facet_name'][:26]:26}"
              f" [{f['category']}]")
        if sc:
            print(f"                 └ rubric[{sc}]: {anchor[:78]}")


if __name__ == "__main__":
    _demo()
