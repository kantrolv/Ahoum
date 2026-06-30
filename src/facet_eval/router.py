"""Phase 4a — the router: pick which facets to score for a given turn.

For each turn we retrieve the globally most-similar facets (top-K) AND, on top of
that, guarantee the top-N *safety* facets by similarity — the "safety floor".

Why the floor: scoring only happens on routed facets, so a safety facet that
falls below the global top-K cutoff is never scored at all. For a turn about a
panic attack, depression/burnout/etc. should always get a look even if lots of
other content crowds the global ranking. The floor makes safety recall a
first-class guarantee rather than a side effect of cosine similarity.

This is also the scaling mechanism: whether the registry holds 399 or 5000
facets, the router returns a bounded set (~top-K + floor), so the per-turn
scoring cost is constant.
"""

from __future__ import annotations

from . import config
from .vectorstore import query_facets


def route(
    turn_text: str,
    *,
    top_k: int | None = None,
    safety_floor: int | None = None,
) -> list[dict]:
    """Return the facets to score for one turn (top-K ∪ safety floor).

    Each facet dict carries the vector-store metadata plus:
      - score: similarity to the turn (1 - cosine distance)
      - via:   'topk', 'safety_floor', or 'both' — provenance, so the pipeline
               can show which facets the floor rescued.
    Sorted by similarity, highest first.
    """
    top_k = top_k if top_k is not None else config.ROUTER_TOP_K
    safety_floor = (
        safety_floor if safety_floor is not None else config.ROUTER_SAFETY_FLOOR
    )

    main = query_facets(turn_text, top_k=top_k)
    floor = (
        query_facets(
            turn_text,
            top_k=safety_floor,
            where={"category": config.SAFETY_CATEGORY},
        )
        if safety_floor > 0
        else []
    )

    in_main = {f["facet_id"] for f in main}
    in_floor = {f["facet_id"] for f in floor}

    merged: dict[str, dict] = {}
    for f in main + floor:
        fid = f["facet_id"]
        if fid not in merged:
            via = "both" if (fid in in_main and fid in in_floor) else (
                "topk" if fid in in_main else "safety_floor"
            )
            merged[fid] = {**f, "via": via}

    return sorted(merged.values(), key=lambda x: x["score"], reverse=True)


def rescued_by_floor(routed: list[dict]) -> list[dict]:
    """The safety facets the floor added that the global top-K would have missed."""
    return [f for f in routed if f["via"] == "safety_floor"]


if __name__ == "__main__":
    turn = ("My heart's racing, I can't stop shaking and I haven't slept in days "
            "— I feel like something terrible is about to happen.")
    routed = route(turn)
    print(f'turn: "{turn[:60]}..."')
    print(f"routed {len(routed)} facets "
          f"(top_k={config.ROUTER_TOP_K} + safety_floor={config.ROUTER_SAFETY_FLOOR})\n")
    print("safety facets RESCUED by the floor (below global top-K):")
    for f in rescued_by_floor(routed):
        print(f"  {f['score']:.3f}  {f['facet_id']}  {f['facet_name']}  [{f['category']}]")
