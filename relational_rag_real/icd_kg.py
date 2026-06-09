"""Semi-real table+KG benchmark from public ICD-10-CM codes + Wikidata diseases.

Tests the paper's motivating example (ICD-10 J45.901 vs J45.9) on REAL data. A table
of clinical records carries specific ICD-10-CM billable codes (e.g. R509 "Fever,
unspecified"); a Wikidata disease KG carries category-level ICD-10 codes (e.g. R50
"fever of unknown origin"). Joining a record to its KG node requires matching codes
across real hierarchy/format drift. Exact code matching fails on drifted records; a
learned verifier with hierarchy + embedding features should recover them.
"""

from __future__ import annotations

import csv
import json
import random
import re
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

DATA = Path(__file__).resolve().parents[1] / "data" / "icd_kg"


def norm(code) -> str:
    return re.sub(r"[^A-Za-z0-9]", "", str(code)).upper()


def load_icd10() -> list[tuple[str, str]]:
    """(full_code_norm, description) for every ICD-10-CM billable code."""
    rows = []
    with (DATA / "icd10.csv").open(encoding="utf-8", errors="ignore") as handle:
        for r in csv.reader(handle):
            if len(r) >= 4 and r[2]:
                rows.append((norm(r[2]), r[3]))
    return rows


def load_kg() -> list[dict]:
    """Wikidata disease nodes: {qid, label, code, norm}, deduped by normalized code."""
    d = json.load((DATA / "wikidata_icd.json").open(encoding="utf-8"))
    seen, out = set(), []
    for b in d["results"]["bindings"]:
        label = b.get("itemLabel", {}).get("value", "")
        code = b.get("icd", {}).get("value", "")
        if not label or not code or label.startswith("Q"):
            continue
        n = norm(code)
        if n in seen:
            continue
        seen.add(n)
        out.append({"qid": b["item"]["value"].split("/")[-1], "label": label, "code": code, "norm": n})
    return out


# OCR/data-entry confusions seen in real clinical coding.
_OCR = {"0": "O", "O": "0", "1": "I", "I": "1", "5": "S", "S": "5", "8": "B", "B": "8", "2": "Z", "Z": "2"}


def corrupt_code(code: str, rng: random.Random, n_edits: int = 1) -> str:
    """Apply OCR/typo-like edits so the code is no longer a clean prefix of its category."""
    chars = list(code)
    for _ in range(n_edits):
        if not chars:
            break
        op = rng.choice(["ocr", "ocr", "transpose", "delete"])
        i = rng.randrange(len(chars))
        if op == "ocr" and chars[i] in _OCR:
            chars[i] = _OCR[chars[i]]
        elif op == "transpose" and i + 1 < len(chars):
            chars[i], chars[i + 1] = chars[i + 1], chars[i]
        elif op == "delete" and len(chars) > 2:
            chars.pop(i)
    return "".join(chars)


@dataclass
class Record:
    table_code: str   # normalized ICD code stored in the table (possibly drifted)
    table_desc: str   # the record's diagnosis description
    gold_qid: str     # the KG node it should join to
    drift: bool       # table_code != gold KG node's code
    corrupted: bool = False  # OCR/typo corruption applied to the code


def build(seed: int = 0, drift_frac: float = 0.7, corrupt_frac: float = 0.0):
    """Build the table+KG benchmark. ``corrupt_frac`` of drifted records additionally
    get OCR/typo-corrupted codes (the regime where prefix/rule matching fails)."""
    icd = load_icd10()
    kg = load_kg()
    desc_of = dict(icd)
    by_cat = defaultdict(list)
    for code, _ in icd:
        by_cat[code[:3]].append(code)

    rng = random.Random(seed)
    recs = []
    for node in kg:
        K = node["norm"]
        children = [c for c in by_cat.get(K[:3], []) if c != K and c.startswith(K) and len(c) > len(K)]
        if children and rng.random() < drift_frac:
            child = rng.choice(children)  # a more-specific real billable child (drift)
            desc = desc_of.get(child, node["label"])
            do_corrupt = bool(corrupt_frac) and rng.random() < corrupt_frac
            tc = corrupt_code(child, rng, n_edits=rng.choice([1, 2])) if do_corrupt else child
            recs.append(Record(tc, desc, node["qid"], True, corrupted=do_corrupt))
        else:
            recs.append(Record(K, node["label"], node["qid"], False))
    return recs, kg


def available() -> bool:
    return (DATA / "icd10.csv").exists() and (DATA / "wikidata_icd.json").exists()
