#!/usr/bin/env python3
"""Verify immutable hashes and structural completeness of this audit package."""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

from compute_agreement import AuditInputError, validate_package


HERE = Path(__file__).resolve().parent
MANIFEST = HERE / "SOURCE_MANIFEST.sha256"


def verify_hashes() -> dict[str, object]:
    if not MANIFEST.is_file():
        raise AuditInputError(f"missing required file: {MANIFEST.name}")
    checked: list[str] = []
    failures: list[str] = []
    for line_number, line in enumerate(MANIFEST.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        try:
            expected, relative = line.split("  ", 1)
        except ValueError as error:
            raise AuditInputError(f"{MANIFEST.name}:{line_number}: malformed hash line") from error
        path = HERE / relative
        if not path.is_file():
            failures.append(f"missing:{relative}")
            continue
        actual = hashlib.sha256(path.read_bytes()).hexdigest()
        checked.append(relative)
        if actual != expected:
            failures.append(f"hash:{relative}")
    if failures:
        raise AuditInputError("source integrity check failed: " + ", ".join(failures))
    return {"status": "valid", "files_checked": len(checked), "files": checked}


def main() -> int:
    try:
        hashes = verify_hashes()
        _, structure = validate_package()
        print(json.dumps({"source_integrity": hashes, "structure": structure}, indent=2, ensure_ascii=False))
        return 0
    except (AuditInputError, OSError, ValueError) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
