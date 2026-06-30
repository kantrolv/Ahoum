"""Phase 1c — quality control + repair for the enriched registry.

Three classes of defect are handled:

  1. ORDERING — anchors written in the wrong field order (scale_1 holding the
     level-5 text, etc.). Fixed *deterministically* by `normalize_registry`,
     which parses each anchor's leading level number and reorders to ascending.
     No LLM, no cost, no ambiguity.

  2. MALFORMED — anchors that cannot be parsed into five distinct ascending
     levels: legend dumps ("1=.. 2=.. 3=.."), duplicated/skipped levels, or no
     level number at all. `normalize_anchors` returns None for these; they are
     collected and REGENERATED with the strict prompt+validator.

  3. INVERSION / off-construct — anchors that describe the *opposite* of the
     facet (e.g. Merriness anchors describing sadness). Not catchable by
     parsing, so we use an embedding heuristic: anchors whose meaning is far
     from the facet *name* are flagged and regenerated.

Plus MISSING facet_ids (skipped during the bulk run) are regenerated/added.

Run standalone:
    python -m facet_eval.registry_qc --audit          # report only, no writes
    python -m facet_eval.registry_qc --fix            # normalize + regen + verify
    python -m facet_eval.registry_qc --fix --invert-k 20
"""

from __future__ import annotations

import argparse
from types import SimpleNamespace

import numpy as np
import pandas as pd

from . import config
from .clean import load_clean_facets
from .enrich import _COLUMNS, _row_record, enrich_facet
from .schema import normalize_anchors
from .vectorstore import embed_texts

ANCHOR_COLS = ["scale_1", "scale_2", "scale_3", "scale_4", "scale_5"]


def _anchors(row) -> list[str]:
    return [str(row[c]) for c in ANCHOR_COLS]


def normalize_registry(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """Reorder+strip anchors for every parseable row, in place.

    Returns (df, malformed_ids) where malformed_ids could NOT be parsed into
    five distinct ascending levels and therefore need regeneration.
    """
    malformed: list[str] = []
    for i, row in df.iterrows():
        fixed = normalize_anchors(_anchors(row))
        if fixed is None:
            malformed.append(row["facet_id"])
        else:
            for col, val in zip(ANCHOR_COLS, fixed):
                df.at[i, col] = val
    return df, malformed


def detect_inversions(
    df: pd.DataFrame, *, k: int | None = None
) -> tuple[list[str], dict[str, float]]:
    """Flag rows whose anchors are semantically farthest from the facet name.

    The facet *name* is ground truth ("Merriness"); if the combined anchors
    embed far from it, the anchors are likely describing the wrong construct.
    Returns (suspect_ids, sim_by_id). We flag the lowest-K (default ~5%) so the
    detector is threshold-free and regeneration cost is bounded.
    """
    names = df["facet_name"].tolist()
    joined = df[ANCHOR_COLS].agg(" ".join, axis=1).tolist()
    nv = np.asarray(embed_texts(names))           # normalized -> dot == cosine
    av = np.asarray(embed_texts(joined))
    sims = (nv * av).sum(axis=1)
    sim_by_id = dict(zip(df["facet_id"], (round(float(s), 4) for s in sims)))

    k = k if k is not None else max(10, round(0.05 * len(df)))
    order = np.argsort(sims)  # ascending: least similar first
    suspect_ids = [df.iloc[idx]["facet_id"] for idx in order[:k]]
    return suspect_ids, sim_by_id


def find_missing(clean_df: pd.DataFrame, df: pd.DataFrame) -> list[str]:
    have = set(df["facet_id"])
    return [fid for fid in clean_df["facet_id"] if fid not in have]


def regenerate(
    df: pd.DataFrame, clean_df: pd.DataFrame, ids: list[str]
) -> pd.DataFrame:
    """Regenerate (or add) the given facet_ids with the strict enrich path.

    Returns a new frame in canonical facet_id order. Rows produced here pass
    through the schema validator, so they are already ascending/clean.
    """
    name_by_id = dict(zip(clean_df["facet_id"], clean_df["facet_name"]))
    rows: dict[str, dict] = {
        r["facet_id"]: {c: r[c] for c in _COLUMNS}
        for _, r in df.iterrows()
    }
    for j, fid in enumerate(ids, start=1):
        name = name_by_id[fid]
        try:
            e = enrich_facet(name)
        except Exception as err:  # noqa: BLE001 - report and continue
            print(f"  [{j}/{len(ids)}] {fid} {name[:38]:38} -> FAILED ({err})")
            continue
        rows[fid] = _row_record(
            SimpleNamespace(facet_id=fid, facet_name=name), e
        )
        print(f"  [{j}/{len(ids)}] {fid} {name[:38]:38} -> "
              f"{e.category}/{e.polarity}", flush=True)

    ordered = [rows[fid] for fid in clean_df["facet_id"] if fid in rows]
    return pd.DataFrame(ordered, columns=_COLUMNS)


def audit(df: pd.DataFrame) -> list[str]:
    """Return facet_ids whose anchors still do NOT parse to 5 ascending levels."""
    return [r["facet_id"] for _, r in df.iterrows()
            if normalize_anchors(_anchors(r)) is None]


def run_fix(*, invert_k: int | None, write: bool) -> None:
    path = config.ENRICHED_FACETS_CSV
    df = pd.read_csv(path, dtype=str).fillna("")
    clean = load_clean_facets()
    print(f"loaded {len(df)} enriched rows; registry has {len(clean)} facets\n")

    # 1) deterministic ordering fix
    df, malformed = normalize_registry(df)
    print(f"ordering: normalized all parseable rows; {len(malformed)} malformed")

    # 2) missing facets (skipped during bulk run)
    missing = find_missing(clean, df)
    print(f"missing : {len(missing)} facet_ids absent from the CSV")

    # 3) inversion / off-construct suspects
    suspects, sims = detect_inversions(df, k=invert_k)
    print(f"inversion: flagged {len(suspects)} lowest name<->anchor similarity")

    regen_ids = sorted(set(malformed) | set(missing) | set(suspects))
    print(f"\n=> regenerating {len(regen_ids)} rows "
          f"(malformed ∪ missing ∪ inversion-suspects)\n")
    if not write:
        print("audit-only (no --fix): stopping before regeneration.")
        print("malformed:", malformed[:20])
        print("suspects :", [(s, sims[s]) for s in suspects[:20]])
        return

    if regen_ids:
        df = regenerate(df, clean, regen_ids)

    # 4) re-normalize everything + verify
    df, still_bad = normalize_registry(df)
    df = df.sort_values("facet_id").reset_index(drop=True)
    df.to_csv(path, index=False)

    leftover = audit(df)
    print(f"\nwrote {path} ({len(df)} rows)")
    print(f"VERIFY: rows that still don't ascend: {len(leftover)} "
          f"{leftover[:10]}")
    if leftover or len(df) != len(clean):
        print("WARNING: registry not fully clean — inspect the ids above.")
    else:
        print("OK: all rows ascend 1..5 and the registry is complete.")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Audit/repair the enriched registry.")
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--audit", action="store_true", help="report defects, write nothing")
    g.add_argument("--fix", action="store_true", help="normalize + regenerate + verify")
    ap.add_argument("--invert-k", type=int, default=None,
                    help="how many lowest-similarity rows to treat as inversions")
    args = ap.parse_args()
    run_fix(invert_k=args.invert_k, write=args.fix)
