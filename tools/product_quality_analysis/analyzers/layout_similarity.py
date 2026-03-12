from __future__ import annotations

from itertools import combinations

from ..models import PageSnapshot


def _similarity(left: PageSnapshot, right: PageSnapshot) -> float:
    left_tokens = set(left.layout_signature + left.component_signatures[:20])
    right_tokens = set(right.layout_signature + right.component_signatures[:20])
    if not left_tokens and not right_tokens:
        return 1.0
    return float(len(left_tokens & right_tokens) / max(1, len(left_tokens | right_tokens)))


def analyze_layout_similarity(page_snapshots: list[PageSnapshot]) -> dict:
    pairs = [{"left": left.name, "right": right.name, "layout_similarity_score": round(_similarity(left, right), 4)} for left, right in combinations(page_snapshots, 2)]
    average = round(sum(item["layout_similarity_score"] for item in pairs) / max(1, len(pairs)), 4) if pairs else 1.0
    return {"pairs": pairs, "layout_similarity_score": average, "issues": []}
