"""
Guards the "not tuned to Plan 5" (or any other single plan) promise as an
ongoing invariant rather than a one-time claim in a status document.

`PLANn`/`planN` identifiers are expected to appear throughout `backend/` --
in comments explaining *why* a generic fix was needed (e.g. "PLAN5's plot
boundary is dash-dot, so line clustering must not require a minimum inked
fraction") and in docstring examples. What must never appear is one used as
a literal branch condition: `if "PLAN5" in ...`, `== "PLAN5.pdf"`,
`filename.startswith("PLAN5")`, etc. That would mean the pipeline special-
cases one specific uploaded file rather than fixing the general extraction
weakness the file happened to expose.
"""

from __future__ import annotations

import re
from pathlib import Path

BACKEND_DIR = Path(__file__).resolve().parent.parent / "backend"

# A plan identifier used as part of an actual conditional/comparison
# expression: `== "PLAN5"`, `== 'PLAN5.pdf'`, `in ("PLAN5", ...)`,
# `.startswith("PLAN5")`, `plan_id == "PLAN5"`. Deliberately requires the
# identifier to sit inside quotes immediately adjacent to a comparison-ish
# operator/call, so a comment or docstring sentence mentioning PLAN5 in
# prose (no surrounding quotes-as-a-literal-being-compared) does not match.
_HARDCODED_PLAN_LITERAL_RE = re.compile(
    r"""
    (?:==|!=|\bin\b|startswith|endswith|\.get\()
    \s*
    \(?\s*
    ['"](PLAN\d+)(?:\.pdf|\.dxf)?['"]
    """,
    re.I | re.VERBOSE,
)


def _iter_python_files():
    for path in BACKEND_DIR.rglob("*.py"):
        if "__pycache__" in path.parts:
            continue
        yield path


def test_no_backend_code_branches_on_a_literal_plan_id():
    offenders: list[str] = []
    for path in _iter_python_files():
        text = path.read_text(encoding="utf-8", errors="ignore")
        for line_no, line in enumerate(text.splitlines(), start=1):
            stripped = line.strip()
            if stripped.startswith("#"):
                continue
            for match in _HARDCODED_PLAN_LITERAL_RE.finditer(line):
                offenders.append(f"{path.relative_to(BACKEND_DIR.parent)}:{line_no}: {stripped!r} (matched {match.group(1)!r})")
    assert not offenders, (
        "Extraction logic must never branch on a literal plan filename/id -- "
        "found apparent hardcoding:\n" + "\n".join(offenders)
    )
