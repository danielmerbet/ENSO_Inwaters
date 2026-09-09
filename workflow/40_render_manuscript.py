#!/usr/bin/env python3
"""Step 40 - render the manuscript with the computed numbers filled in.

``paper/manuscript_template.md`` contains placeholders of the form
``{{events.n_super}}`` that address into ``results/paper_numbers.json``
by dotted path. This step substitutes them and writes
``paper/manuscript.md``.

Any placeholder whose value has not been computed is left in place and
listed at the end of the run, so an unfinished number is visible in the
draft rather than silently rendered as a plausible-looking blank.

Run:  python workflow/40_render_manuscript.py
"""

from __future__ import annotations

import json
import re
from pathlib import Path

from _common import step_setup

PLACEHOLDER = re.compile(r"\{\{([a-zA-Z0-9_.\[\]-]+)(?::([^}]+))?\}\}")


def lookup(data: dict, path: str):
    node = data
    for part in path.split("."):
        if isinstance(node, list):
            try:
                node = node[int(part)]
                continue
            except (ValueError, IndexError):
                return None
        if not isinstance(node, dict) or part not in node:
            return None
        node = node[part]
    return node


def fmt(value, spec: str | None) -> str:
    if value is None:
        return ""
    if isinstance(value, list):
        return ", ".join(str(v) for v in value)
    if spec:
        try:
            return format(value, spec)
        except (ValueError, TypeError):
            return str(value)
    if isinstance(value, float):
        return f"{value:.2f}"
    return str(value)


def main() -> int:
    cfg, log, args = step_setup("40_render_manuscript", __doc__)
    numbers_path = cfg.path("processed", "paper_numbers.json")
    template = Path(cfg.root) / "paper" / "manuscript_template.md"
    out_path = Path(cfg.root) / "paper" / "manuscript.md"

    if not template.exists():
        log.error("no template at %s", template)
        return 1
    if not numbers_path.exists():
        log.error("no %s - run step 31 first", numbers_path)
        return 1

    numbers = json.loads(numbers_path.read_text())
    text = template.read_text()
    unresolved: list[str] = []

    def replace(m: re.Match) -> str:
        path, spec = m.group(1), m.group(2)
        value = lookup(numbers, path)
        if value is None:
            unresolved.append(path)
            return m.group(0)
        return fmt(value, spec)

    rendered = PLACEHOLDER.sub(replace, text)
    n_total = len(PLACEHOLDER.findall(text))

    if not args.dry_run:
        out_path.write_text(rendered)
        log.info("wrote %s", out_path)
    log.info("%d placeholder(s), %d resolved, %d unresolved",
             n_total, n_total - len(unresolved), len(unresolved))
    if unresolved:
        log.warning("unresolved placeholders (left visible in the draft): %s",
                    ", ".join(sorted(set(unresolved))))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
