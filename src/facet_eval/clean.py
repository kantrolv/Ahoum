"""Phase 1a — clean the raw facet list into a tidy, de-duplicated registry.

Deterministic only (no LLM). Every transformation is logged so the cleaning is
auditable. Output columns at this stage: facet_id, facet_name, raw_name.

Run standalone:
    python -m facet_eval.clean            # prints a report, writes nothing
    python -m facet_eval.clean --write    # writes data/facets_clean.csv
"""

from __future__ import annotations

import argparse
import re

import pandas as pd

from . import config

# Leading enumeration like "800. " or "596. " seen throughout the raw file.
_LEADING_ENUM = re.compile(r"^\s*\d+\.\s*")
# Collapse runs of whitespace.
_WS = re.compile(r"\s+")


def clean_name(raw: str) -> str:
    """Normalize one facet name without destroying meaning.

    - strip a leading "NNN. " enumeration
    - strip trailing colons / whitespace (artifact of the source doc)
    - collapse internal whitespace
    Internal structure like "Religious practice: Quran khatam cycles" is KEPT —
    it is informative context, not noise.
    """
    name = str(raw).strip()
    name = _LEADING_ENUM.sub("", name)
    name = name.rstrip(": ").strip()
    name = _WS.sub(" ", name)
    return name


def _norm_key(name: str) -> str:
    """Case/space-insensitive key used only to detect exact duplicates."""
    return _WS.sub(" ", name.lower()).strip()


def load_clean_facets() -> pd.DataFrame:
    """Load the raw CSV and return a cleaned, de-duplicated DataFrame.

    Returns columns: facet_id, facet_name, raw_name.
    facet_id is a stable zero-padded id (F0001…) — adding facets later appends
    new ids and never renumbers existing ones, which matters for the vector
    store and any saved score matrices.
    """
    raw = pd.read_csv(config.RAW_FACETS_CSV)
    # The single column may be named "Facets"; take the first column regardless.
    col = raw.columns[0]
    df = pd.DataFrame({"raw_name": raw[col].astype(str)})

    # Drop blanks / NaNs.
    df["raw_name"] = df["raw_name"].str.strip()
    df = df[df["raw_name"].ne("") & df["raw_name"].str.lower().ne("nan")]

    df["facet_name"] = df["raw_name"].map(clean_name)
    df = df[df["facet_name"].ne("")]

    # De-duplicate on the normalized key, keeping first occurrence.
    df["_key"] = df["facet_name"].map(_norm_key)
    df = df.drop_duplicates(subset="_key", keep="first").drop(columns="_key")

    df = df.reset_index(drop=True)
    df.insert(0, "facet_id", [f"F{i + 1:04d}" for i in range(len(df))])
    return df


def _report(df_raw: pd.DataFrame, df_clean: pd.DataFrame) -> None:
    n_raw = len(df_raw)
    n_clean = len(df_clean)
    print(f"raw rows (excl. header): {n_raw}")
    print(f"clean rows             : {n_clean}")
    print(f"removed (blank/dupe)   : {n_raw - n_clean}")
    changed = df_clean[df_clean["raw_name"] != df_clean["facet_name"]]
    print(f"names altered by cleaning: {len(changed)}")
    print("\nsample of altered names:")
    for _, r in changed.head(8).iterrows():
        print(f"  {r['raw_name']!r:60} -> {r['facet_name']!r}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Clean the raw facet registry.")
    ap.add_argument("--write", action="store_true", help="write data/facets_clean.csv")
    args = ap.parse_args()

    raw = pd.read_csv(config.RAW_FACETS_CSV)
    clean = load_clean_facets()
    _report(raw, clean)

    if args.write:
        out = config.DATA_DIR / "facets_clean.csv"
        clean.to_csv(out, index=False)
        print(f"\nwrote {out}")
