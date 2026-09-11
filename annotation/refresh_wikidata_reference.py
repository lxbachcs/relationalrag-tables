#!/usr/bin/env python3
"""Refresh the optional offline Wikidata reference snapshot.

The annotation workflow itself is offline. This maintenance command contacts the
official Wikidata API and rewrites ``icd_wikidata_reference.csv`` for the QIDs
already present in ``annotation_sample_icd.csv``.
"""

from __future__ import annotations

import csv
import json
import sys
import urllib.parse
import urllib.request
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path


HERE = Path(__file__).resolve().parent
INPUT = HERE / "annotation_sample_icd.csv"
OUTPUT = HERE / "icd_wikidata_reference.csv"
API = "https://www.wikidata.org/w/api.php"
USER_AGENT = "RelationalRAG-Tables-label-audit/1.0 (offline snapshot builder)"


def chunks(values: list[str], size: int) -> list[list[str]]:
    return [values[i : i + size] for i in range(0, len(values), size)]


def text_value(entity: dict, field: str) -> str:
    values = entity.get(field, {})
    if "en" in values:
        return values["en"].get("value", "")
    for value in values.values():
        if isinstance(value, dict) and value.get("value"):
            return value["value"]
    return ""


def aliases(entity: dict) -> str:
    values = entity.get("aliases", {}).get("en", [])
    return " | ".join(sorted({item.get("value", "") for item in values if item.get("value")}))


def icd10_codes(entity: dict) -> str:
    codes: set[str] = set()
    for claim in entity.get("claims", {}).get("P494", []):
        value = claim.get("mainsnak", {}).get("datavalue", {}).get("value")
        if isinstance(value, str) and value.strip():
            codes.add(value.strip())
    return " | ".join(sorted(codes))


def fetch_entities(qids: list[str]) -> dict[str, dict]:
    entities: dict[str, dict] = {}
    for batch in chunks(qids, 50):
        query = urllib.parse.urlencode(
            {
                "action": "wbgetentities",
                "ids": "|".join(batch),
                "props": "labels|descriptions|aliases|claims|info",
                "languages": "en",
                "languagefallback": "1",
                "format": "json",
                "formatversion": "2",
            }
        )
        request = urllib.request.Request(f"{API}?{query}", headers={"User-Agent": USER_AGENT})
        with urllib.request.urlopen(request, timeout=60) as response:
            payload = json.load(response)
        returned = payload.get("entities", {})
        iterator = returned.values() if isinstance(returned, dict) else returned
        for entity in iterator:
            if isinstance(entity, dict) and entity.get("id"):
                entities[entity["id"]] = entity
    return entities


def main() -> int:
    with INPUT.open(newline="", encoding="utf-8-sig") as handle:
        sample = list(csv.DictReader(handle))

    protocol_codes: dict[str, set[str]] = defaultdict(set)
    for row in sample:
        protocol_codes[row["protocol_gold_qid"]].add(row["protocol_gold_code"])
    qids = sorted(protocol_codes, key=lambda qid: int(qid[1:]))

    entities = fetch_entities(qids)
    missing = sorted(set(qids) - set(entities))
    if missing:
        print(f"Wikidata did not return: {', '.join(missing)}", file=sys.stderr)
        return 1

    retrieved = datetime.now(timezone.utc).replace(microsecond=0).isoformat()
    fields = [
        "qid",
        "label_en",
        "description_en",
        "aliases_en",
        "wikidata_icd10_codes",
        "protocol_codes_in_sample",
        "last_revision_id",
        "wikidata_modified_utc",
        "source_url",
        "retrieved_utc",
    ]
    temporary = OUTPUT.with_suffix(".csv.tmp")
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for qid in qids:
            entity = entities[qid]
            writer.writerow(
                {
                    "qid": qid,
                    "label_en": text_value(entity, "labels"),
                    "description_en": text_value(entity, "descriptions"),
                    "aliases_en": aliases(entity),
                    "wikidata_icd10_codes": icd10_codes(entity),
                    "protocol_codes_in_sample": " | ".join(sorted(protocol_codes[qid])),
                    "last_revision_id": entity.get("lastrevid", ""),
                    "wikidata_modified_utc": entity.get("modified", ""),
                    "source_url": f"https://www.wikidata.org/wiki/{qid}",
                    "retrieved_utc": retrieved,
                }
            )
    temporary.replace(OUTPUT)
    print(f"Wrote {len(qids)} entities to {OUTPUT.name} ({retrieved}).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
