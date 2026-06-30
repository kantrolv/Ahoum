"""Phase 4b — the end-to-end pipeline.

conversation -> per turn { router (top-K + safety floor) -> batched scorer }
            -> assemble a turn x facet score matrix -> save JSON.

Nothing here knows how many facets exist. Each turn is routed to a bounded set
and scored in bounded batches, so the same code runs unchanged for 399 or 5000
facets — the registry/store grow, the pipeline does not.

Run:
    python -m facet_eval.pipeline data/conversations/demo_mixed_001.json
    python -m facet_eval.pipeline            # defaults to the demo conversation
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

from . import config
from .router import route
from .scorer import score

# Metadata the router/store provide that we keep alongside each score.
_CARRY = ("facet_name", "category", "polarity", "via")


def _chunks(items: list, size: int):
    for i in range(0, len(items), size):
        yield items[i:i + size]


def score_turn(turn_text: str) -> list[dict]:
    """Route one turn, score it in batches, return per-facet results.

    Each result: {facet_id, facet_name, category, polarity, via, similarity,
    score, confidence}. `score`/`confidence` are None if the model never
    returned that facet (surfaced, not dropped).
    """
    routed = route(turn_text)
    meta = {f["facet_id"]: f for f in routed}

    scored: list[dict] = []
    for batch in _chunks(routed, config.SCORE_BATCH_SIZE):
        scored.extend(score(turn_text, batch))

    out: list[dict] = []
    for s in scored:
        f = meta[s["facet_id"]]
        out.append(
            {
                "facet_id": s["facet_id"],
                **{k: f.get(k) for k in _CARRY},
                "similarity": f.get("score"),
                "score": s["score"],
                "confidence": s["confidence"],
            }
        )
    return out


def score_conversation(conv: dict) -> dict:
    """Score every turn of a conversation and assemble the result document."""
    turns_out = []
    for i, turn in enumerate(conv["turns"]):
        results = score_turn(turn["text"])
        n_missing = sum(1 for r in results if r["score"] is None)
        turns_out.append(
            {
                "index": i,
                "speaker": turn.get("speaker", ""),
                "text": turn["text"],
                "n_routed": len(results),
                "n_scored": len(results) - n_missing,
                "n_missing": n_missing,
                "rescued_by_safety_floor": [
                    r["facet_id"] for r in results if r["via"] == "safety_floor"
                ],
                "scores": results,
            }
        )
        print(f"  turn {i} ({turn.get('speaker','')}): routed {len(results)}, "
              f"missing {n_missing}", flush=True)

    return {
        "conversation_id": conv.get("conversation_id", "unknown"),
        "scored_at": datetime.now(timezone.utc).isoformat(),
        "config": {
            "model": config.OLLAMA_MODEL,
            "embed_model": config.EMBED_MODEL,
            "router_top_k": config.ROUTER_TOP_K,
            "safety_floor": config.ROUTER_SAFETY_FLOOR,
            "batch_size": config.SCORE_BATCH_SIZE,
            "scale": f"{config.SCALE_MIN}-{config.SCALE_MAX}",
        },
        "n_turns": len(conv["turns"]),
        "turns": turns_out,
    }


def save_result(result: dict, out_dir: Path | None = None) -> Path:
    out_dir = out_dir or (config.OUTPUT_DIR / "scores")
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{result['conversation_id']}.json"
    path.write_text(json.dumps(result, indent=2))
    return path


# --- display -------------------------------------------------------------

def print_matrix(result: dict, *, top_n: int = 18) -> None:
    """Print a facet x turn score matrix for the most active facets.

    '·' = facet not routed for that turn; 'X' = routed but the model returned
    no score (None). Rows are the facets with the highest score anywhere in the
    conversation, so the table stays readable even with a sparse 399-wide matrix.
    """
    n_turns = result["n_turns"]
    # facet_id -> {turn_index: score}, plus names and a max for ranking
    grid: dict[str, dict] = {}
    name: dict[str, str] = {}
    for t in result["turns"]:
        for r in t["scores"]:
            fid = r["facet_id"]
            name[fid] = r["facet_name"]
            grid.setdefault(fid, {})[t["index"]] = r["score"]

    def _max(fid: str) -> int:
        return max((s or 0) for s in grid[fid].values())

    ranked = sorted(grid, key=_max, reverse=True)[:top_n]

    header = "turn:   " + " ".join(f"{i:>2}" for i in range(n_turns))
    print("\n" + header)
    print("-" * len(header))
    for fid in ranked:
        cells = []
        for i in range(n_turns):
            if i not in grid[fid]:
                cells.append(" ·")
            elif grid[fid][i] is None:
                cells.append(" X")
            else:
                cells.append(f"{grid[fid][i]:>2}")
        print(f"{' '.join(cells)}   {fid} {name[fid][:30]}")
    print("\nlegend: number=score(1-5)  ·=not routed this turn  X=routed but unscored(None)")


def _load(path: Path) -> dict:
    return json.loads(path.read_text())


def run_batch(conv_dir: Path | None = None, *, resume: bool = True) -> None:
    """Score every conversation in `conv_dir`, resumably.

    Crash-safe: a conversation whose output JSON already exists is skipped, so a
    killed/restarted run picks up where it left off (same pattern as enrichment).
    """
    conv_dir = conv_dir or (config.DATA_DIR / "conversations")
    out_dir = config.OUTPUT_DIR / "scores"
    out_dir.mkdir(parents=True, exist_ok=True)

    paths = sorted(p for p in conv_dir.glob("*.json") if not p.name.startswith("_"))
    todo = [p for p in paths
            if not (resume and (out_dir / f"{_load(p)['conversation_id']}.json").exists())]
    print(f"{len(paths)} conversations; {len(paths) - len(todo)} already scored; "
          f"{len(todo)} to do (model={config.OLLAMA_MODEL})\n")

    for i, p in enumerate(todo, start=1):
        conv = _load(p)
        print(f"[{i}/{len(todo)}] {conv['conversation_id']} "
              f"({conv.get('case_type','?')}, {len(conv['turns'])} turns)", flush=True)
        result = score_conversation(conv)
        save_result(result)
    print(f"\nbatch complete: {len(todo)} scored this run, output in {out_dir}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Score a conversation end-to-end.")
    ap.add_argument("conversation", nargs="?",
                    default=str(config.DATA_DIR / "conversations" / "demo_mixed_001.json"),
                    help="path to a conversation JSON")
    ap.add_argument("--all", action="store_true",
                    help="resumably score every conversation in data/conversations/")
    args = ap.parse_args()

    if args.all:
        run_batch()
    else:
        conv = _load(Path(args.conversation))
        print(f"scoring conversation '{conv.get('conversation_id')}' "
              f"({len(conv['turns'])} turns) with {config.OLLAMA_MODEL}...")
        result = score_conversation(conv)
        out = save_result(result)
        print(f"\nsaved -> {out}")
        print_matrix(result)
