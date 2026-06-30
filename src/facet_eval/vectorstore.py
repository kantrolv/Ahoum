"""Phase 2 — embed the enriched facets and store them in ChromaDB.

This is the backbone of the scaling story. Instead of ever putting all facets in
a prompt, we turn each facet into a vector (its semantic fingerprint) and store it
once. At scoring time the router embeds a conversation turn and asks the store
"which facets are nearest?" — so a 5000-facet registry costs the same per turn as
a 300-facet one: a single approximate-nearest-neighbour lookup, then we only score
the handful that came back.

Adding facets = upsert more rows. No re-embedding of existing facets, no code
change. That is the registry pattern made physical.

Embedding model: BAAI/bge-small-en-v1.5 (384-dim, CPU-fast). For bge v1.5,
retrieval works best when the QUERY (the turn) is prefixed with a short
instruction; the stored facets are embedded plain. We normalize and use cosine
similarity.

Run standalone:
    python -m facet_eval.vectorstore --build            # build/refresh the store
    python -m facet_eval.vectorstore --build --rebuild  # wipe & rebuild
    python -m facet_eval.vectorstore --query "I'm so anxious I can't sleep" --k 8
"""

from __future__ import annotations

import argparse
from functools import lru_cache

import chromadb
import pandas as pd
from sentence_transformers import SentenceTransformer

from . import config

# bge-v1.5 retrieval instruction, prepended to QUERIES only (not stored facets).
_QUERY_INSTRUCTION = "Represent this sentence for searching relevant passages: "

# Metadata columns carried into the store so the scorer can pull a facet's full
# rubric straight from a query result — no second CSV read at scoring time.
_META_COLS = [
    "facet_name", "category", "polarity", "definition",
    "scale_1", "scale_2", "scale_3", "scale_4", "scale_5",
]


@lru_cache(maxsize=1)
def _embedder() -> SentenceTransformer:
    """Load the embedding model once per process (it is ~130 MB)."""
    return SentenceTransformer(config.EMBED_MODEL)


def embed_texts(texts: list[str], *, is_query: bool = False) -> list[list[float]]:
    """Embed texts, L2-normalized for cosine similarity.

    Queries get the bge retrieval instruction; stored facets do not.
    """
    if is_query:
        texts = [_QUERY_INSTRUCTION + t for t in texts]
    vecs = _embedder().encode(
        texts, normalize_embeddings=True, show_progress_bar=False
    )
    return vecs.tolist()


@lru_cache(maxsize=1)
def _client() -> chromadb.ClientAPI:
    config.CHROMA_DIR.mkdir(parents=True, exist_ok=True)
    return chromadb.PersistentClient(path=str(config.CHROMA_DIR))


def get_collection() -> chromadb.Collection:
    """Load-once accessor for the facet collection (cosine space)."""
    return _client().get_or_create_collection(
        name=config.CHROMA_COLLECTION,
        metadata={"hnsw:space": "cosine"},
    )


def _facet_document(name: str, definition: str) -> str:
    """The text we embed for a facet: its name plus definition (richer signal)."""
    return f"{name}: {definition}"


def build_store(csv_path=None, *, rebuild: bool = False) -> int:
    """Embed every enriched facet and upsert it into ChromaDB.

    Returns the number of facets in the collection afterwards. Idempotent: re-runs
    upsert by stable facet_id, so re-running after adding facets only adds the new
    ones. `rebuild=True` wipes the collection first.
    """
    csv_path = csv_path or config.ENRICHED_FACETS_CSV
    df = pd.read_csv(csv_path, dtype=str).fillna("")
    if df.empty:
        raise ValueError(f"no facets to index in {csv_path}")

    if rebuild:
        try:
            _client().delete_collection(config.CHROMA_COLLECTION)
        except Exception:
            pass  # collection may not exist yet — fine

    col = get_collection()

    docs = [
        _facet_document(r["facet_name"], r["definition"])
        for _, r in df.iterrows()
    ]
    embeddings = embed_texts(docs, is_query=False)
    metadatas = [{c: r[c] for c in _META_COLS} for _, r in df.iterrows()]

    col.upsert(
        ids=df["facet_id"].tolist(),
        embeddings=embeddings,
        documents=docs,
        metadatas=metadatas,
    )
    return col.count()


def get_facets(ids: list[str]) -> list[dict]:
    """Fetch facets (with full rubric metadata) by id, preserving input order."""
    col = get_collection()
    res = col.get(ids=ids, include=["metadatas"])
    by_id = {fid: {"facet_id": fid, **meta}
             for fid, meta in zip(res["ids"], res["metadatas"])}
    return [by_id[i] for i in ids if i in by_id]


def query_facets(
    text: str, top_k: int | None = None, *, where: dict | None = None
) -> list[dict]:
    """Return the top-K facets most relevant to `text`, nearest first.

    Each result: {facet_id, score (1 - cosine distance), ...metadata}.
    `where` is an optional ChromaDB metadata filter, e.g. {"category": "safety"}
    to rank only within one category (used by the router's safety floor).
    """
    top_k = top_k or config.ROUTER_TOP_K
    col = get_collection()
    q_emb = embed_texts([text], is_query=True)
    res = col.query(
        query_embeddings=q_emb,
        n_results=min(top_k, col.count()),
        where=where,
        include=["metadatas", "distances"],
    )
    out: list[dict] = []
    for fid, meta, dist in zip(
        res["ids"][0], res["metadatas"][0], res["distances"][0]
    ):
        out.append({"facet_id": fid, "score": round(1.0 - dist, 4), **meta})
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Build/query the facet vector store.")
    ap.add_argument("--build", action="store_true", help="embed + upsert all facets")
    ap.add_argument("--rebuild", action="store_true", help="wipe before building")
    ap.add_argument("--query", type=str, help="a sample turn to retrieve facets for")
    ap.add_argument("--k", type=int, default=8, help="top-K for --query")
    args = ap.parse_args()

    if args.build:
        n = build_store(rebuild=args.rebuild)
        print(f"store now holds {n} facets at {config.CHROMA_DIR}")

    if args.query:
        print(f'\nquery: "{args.query}"\n')
        for r in query_facets(args.query, top_k=args.k):
            print(f"  {r['score']:.3f}  {r['facet_id']}  "
                  f"{r['facet_name'][:38]:38} [{r['category']}/{r['polarity']}]")
