from __future__ import annotations

import re
from pathlib import Path

from ..models import PageSnapshot


TOKEN_PATTERN = re.compile(r"--([a-z0-9-]+)\s*:")


def analyze_design_tokens(page_snapshots: list[PageSnapshot]) -> dict:
    pages: list[dict] = []
    for snapshot in page_snapshots:
        html = Path(snapshot.html_path).read_text(encoding="utf-8") if snapshot.html_path else ""
        tokens = sorted(set(TOKEN_PATTERN.findall(html)))
        pages.append({"page": snapshot.name, "token_count": len(tokens), "tokens": tokens[:20]})
    counts = [row["token_count"] for row in pages] if pages else [0]
    return {"pages": pages, "design_token_variance": max(counts) - min(counts), "issues": []}
