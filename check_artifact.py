#!/usr/bin/env python3
"""Validate the self-contained RelationalRAG-Tables artifact.

This artifact folder is meant to stand on its own. The check verifies that the
required code, fixtures, stored results, scripts, docs, and supplemental PDF are
present, and that no manuscript LaTeX/PDF build files have leaked in. It is
conservative and safe to run before packaging or inside Docker.
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent

# Manuscript / LaTeX build artifacts that must never ship with the code artifact.
FORBIDDEN_SUFFIXES = {".tex", ".bbl", ".blg", ".aux", ".fls", ".fdb_latexmk", ".pdf"}
ALLOWED_PDFS = {"supplement.pdf"}

# Internal reports and assistant configuration do not belong in the release.
INTERNAL_PATH = re.compile(
    r"(?:audit|review|rebuttal|claude|codex|chatgpt|copilot|cursor|aider)", re.I
)
ASSISTANT_FILES = {"agents.md", "agent.md", "skill.md", "write.md", ".mcp.json"}
ASSISTANT_DIRS = {".agents", ".skills", ".ai", ".windsurf"}

# Generated-output and cache directories: skip them when scanning for leaks.
SKIP_DIRS = {
    ".git", ".venv", "venv", "env", "__pycache__", ".pytest_cache",
    ".mypy_cache", ".ruff_cache", "build", "dist",
    "relational_rag_tables.egg-info", "relational_rag_real.egg-info",
}

# Everything the folder needs to stand on its own.
REQUIRED = [
    "README.md",
    "REPRODUCE.md",
    "Dockerfile",
    "pyproject.toml",
    "requirements.txt",
    "requirements-real.txt",
    "run_smoke.sh",
    "run_full.sh",
    "check_artifact.py",
    "supplement.pdf",
    "relational_rag_tables",
    "relational_rag_tables/cli.py",
    "relational_rag_real",
    "tests",
    "fixtures/smoke",
    "artifacts/manifest.json",
]


def main() -> int:
    errors: list[str] = []

    for rel in REQUIRED:
        if not (ROOT / rel).exists():
            errors.append(f"missing required artifact path: {rel}")

    # No LaTeX manuscript / accidental build files beyond supplement.pdf.
    for path in ROOT.rglob("*"):
        if path.is_dir():
            continue
        rel = path.relative_to(ROOT)
        if any(part in SKIP_DIRS for part in rel.parts):
            continue
        if (any(INTERNAL_PATH.search(part) for part in rel.parts)
                or path.name.lower() in ASSISTANT_FILES
                or any(part.lower() in ASSISTANT_DIRS for part in rel.parts)):
            errors.append(f"internal report/assistant file in artifact: {rel}")
        if path.suffix.lower() == ".pdf" and path.name in ALLOWED_PDFS:
            continue
        if path.suffix.lower() in FORBIDDEN_SUFFIXES:
            errors.append(f"forbidden manuscript/build file in artifact: {rel}")

    # The stored-results manifest must parse and its tracked files must exist.
    manifest = ROOT / "artifacts" / "manifest.json"
    if manifest.exists():
        try:
            entries = json.loads(manifest.read_text())
            missing = [k for k in entries if not (ROOT / k).exists()]
            if missing:
                errors.append(
                    f"manifest references {len(missing)} missing result file(s), "
                    f"e.g. {missing[0]}"
                )
        except json.JSONDecodeError as exc:
            errors.append(f"artifacts/manifest.json does not parse: {exc}")

    if errors:
        print("artifact check: fail", file=sys.stderr)
        for err in errors:
            print(f"  - {err}", file=sys.stderr)
        return 1

    print("artifact check: pass")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
