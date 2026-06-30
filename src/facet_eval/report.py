"""Phase 5/6 — coverage statistics + deliverable packaging.

Reads the saved score matrices in output/scores/, computes coverage statistics,
fills the README's coverage block, and zips the deliverable (conversations +
their score matrices).

Run:
    python -m facet_eval.report --stats         # print stats
    python -m facet_eval.report --fill-readme   # write stats into README.md
    python -m facet_eval.report --zip           # build the deliverable zip
    python -m facet_eval.report --all           # stats + fill-readme + zip
"""

from __future__ import annotations

import argparse
import json
import zipfile
from collections import Counter, defaultdict
from pathlib import Path

from . import config

_SCORES_DIR = config.OUTPUT_DIR / "scores"
_CONV_DIR = config.DATA_DIR / "conversations"
_README = config.REPO_ROOT / "README.md"


def _case_type_by_id() -> dict[str, str]:
    """Map conversation_id -> case_type from the conversation inputs."""
    out: dict[str, str] = {}
    for p in _CONV_DIR.glob("*.json"):
        if p.name.startswith("_"):
            continue
        doc = json.loads(p.read_text())
        out[doc["conversation_id"]] = doc.get("case_type", "unspecified")
    return out


def compute_stats(scores_dir: Path = _SCORES_DIR) -> dict:
    """Aggregate coverage statistics over all saved score matrices."""
    case_of = _case_type_by_id()
    results = [json.loads(p.read_text())
               for p in sorted(scores_dir.glob("*.json")) if not p.name.startswith("_")]

    n_turns = n_scores = n_missing = full_cov_turns = 0
    score_hist: Counter = Counter()
    conf_total = 0.0
    rescues = 0
    safety_turns = 0
    safety_rescues = 0
    by_case: dict[str, int] = defaultdict(int)

    for r in results:
        case = case_of.get(r["conversation_id"], "unspecified")
        by_case[case] += 1
        for t in r["turns"]:
            n_turns += 1
            n_missing += t["n_missing"]
            n_scores += len(t["scores"])
            if t["n_missing"] == 0:
                full_cov_turns += 1
            rescues += len(t.get("rescued_by_safety_floor", []))
            if case == "safety-risky":
                safety_turns += 1
                safety_rescues += len(t.get("rescued_by_safety_floor", []))
            for s in t["scores"]:
                if s["score"] is not None:
                    score_hist[s["score"]] += 1
                if s["confidence"] is not None:
                    conf_total += s["confidence"]

    n_real = sum(score_hist.values())
    return {
        "conversations": len(results),
        "by_case": dict(sorted(by_case.items())),
        "turns": n_turns,
        "facet_scores": n_scores,
        "missing_after_retries": n_missing,
        "full_coverage_pct": round(100 * full_cov_turns / n_turns, 1) if n_turns else 0.0,
        "score_histogram": {k: score_hist.get(k, 0) for k in range(config.SCALE_MIN, config.SCALE_MAX + 1)},
        "mean_score": round(sum(k * v for k, v in score_hist.items()) / n_real, 2) if n_real else 0.0,
        "mean_confidence": round(conf_total / n_real, 3) if n_real else 0.0,
        "safety_floor_rescues_total": rescues,
        "safety_risky_turns": safety_turns,
        "mean_rescues_per_safety_turn": round(safety_rescues / safety_turns, 1) if safety_turns else 0.0,
    }


def render_markdown(s: dict, *, total_conversations: int = 55) -> str:
    """Render the coverage block that replaces the README placeholder."""
    hist = " · ".join(f"{k}:{s['score_histogram'][k]}"
                      for k in range(config.SCALE_MIN, config.SCALE_MAX + 1))
    return (
        "> **Coverage statistics** "
        f"(model `{config.OLLAMA_MODEL}`, top-K {config.ROUTER_TOP_K}, "
        f"safety floor {config.ROUTER_SAFETY_FLOOR}):\n"
        f"> - conversations scored: **{s['conversations']}/{total_conversations}** · "
        f"turns scored: **{s['turns']}** · facet-scores produced: **{s['facet_scores']:,}**\n"
        f"> - turns with full coverage (0 missing): **{s['full_coverage_pct']}%** · "
        f"total `None` after retries: **{s['missing_after_retries']}**\n"
        f"> - safety-floor rescues across the set: **{s['safety_floor_rescues_total']}** "
        f"(mean **{s['mean_rescues_per_safety_turn']}** per safety-risky turn)\n"
        f"> - score distribution (1→5): {hist} · mean score **{s['mean_score']}** · "
        f"mean confidence **{s['mean_confidence']}**"
    )


def fill_readme(md: str, readme: Path = _README) -> None:
    """Replace the content between the COVERAGE_STATS sentinels with `md`."""
    text = readme.read_text()
    start = "<!-- COVERAGE_STATS_START -->"
    end = "<!-- COVERAGE_STATS_END -->"
    i, j = text.find(start), text.find(end)
    if i == -1 or j == -1:
        raise RuntimeError("COVERAGE_STATS sentinels not found in README")
    new = text[:i] + start + "\n" + md + "\n" + text[j:]
    readme.write_text(new)


def make_zip(out_path: Path | None = None) -> Path:
    """Zip the conversations + their score matrices as the deliverable."""
    out_path = out_path or (config.OUTPUT_DIR / "facet_eval_deliverable.zip")
    convs = [p for p in _CONV_DIR.glob("*.json") if not p.name.startswith("_")]
    scores = [p for p in _SCORES_DIR.glob("*.json") if not p.name.startswith("_")]
    with zipfile.ZipFile(out_path, "w", zipfile.ZIP_DEFLATED) as z:
        for p in convs:
            z.write(p, f"conversations/{p.name}")
        for p in scores:
            z.write(p, f"scores/{p.name}")
        z.writestr("MANIFEST.txt",
                   f"{len(convs)} conversations, {len(scores)} score matrices\n"
                   f"model={config.OLLAMA_MODEL} top_k={config.ROUTER_TOP_K} "
                   f"safety_floor={config.ROUTER_SAFETY_FLOOR} scale={config.SCALE_MIN}-{config.SCALE_MAX}\n")
    return out_path


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Coverage stats + deliverable packaging.")
    ap.add_argument("--stats", action="store_true")
    ap.add_argument("--fill-readme", action="store_true")
    ap.add_argument("--zip", action="store_true")
    ap.add_argument("--all", action="store_true")
    args = ap.parse_args()

    stats = compute_stats()
    if args.stats or args.all or not any([args.fill_readme, args.zip]):
        print(json.dumps(stats, indent=2))
        print("\n--- README block ---\n" + render_markdown(stats))
    if args.fill_readme or args.all:
        fill_readme(render_markdown(stats))
        print(f"\nfilled coverage block in {_README}")
    if args.zip or args.all:
        z = make_zip()
        print(f"wrote deliverable zip -> {z}")
