from __future__ import annotations

import re


def split_numbered_llm_lines(text: str, *, limit: int) -> list[str]:
    """Prefer lines numbered ``1., 2.)`` …; otherwise fall back to non-empty lines."""

    lines = [ln.strip() for ln in text.splitlines()]
    bullets: list[str] = []

    numbered = re.compile(r"^\s*\d+[.)]\s*(.+)$")
    for ln in lines:
        m = numbered.match(ln)
        if not m:
            continue
        val = m.group(1).strip()
        if val:
            bullets.append(val)
        if len(bullets) >= limit:
            break

    if bullets:
        return bullets[:limit]

    stripped = [ln for ln in lines if ln]
    return stripped[:limit]
