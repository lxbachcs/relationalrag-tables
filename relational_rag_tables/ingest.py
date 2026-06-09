"""Fixture loading and validation."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

from .schemas import Corpus, Question, Table, Triple


REQUIRED_CONFIG_KEYS = {"tau_v", "tau_u", "top_k", "index_backend"}


def load_corpus(fixtures_dir: str | Path) -> Corpus:
    root = Path(fixtures_dir)
    if not root.exists():
        raise FileNotFoundError(f"Fixture directory does not exist: {root}")

    config = _load_json(root / "config.json")
    missing_config = REQUIRED_CONFIG_KEYS - set(config)
    if missing_config:
        raise ValueError(f"config.json missing keys: {sorted(missing_config)}")

    tables = _load_tables(root / "tables")
    triples = [
        Triple(subject=item["subject"], relation=item["relation"], object=item["object"])
        for item in _load_json(root / "kg" / "triples.json")
    ]
    questions = [Question(**item) for item in _load_json(root / "questions.json")]
    join_labels = {
        (item["lhs"], item["rhs"]): bool(item["label"])
        for item in _load_json(root / "join_labels.json")
    }

    corpus = Corpus(
        tables=tables,
        triples=triples,
        questions=questions,
        join_labels=join_labels,
        config=config,
    )
    validate_corpus(corpus)
    return corpus


def validate_corpus(corpus: Corpus) -> None:
    if not corpus.tables:
        raise ValueError("At least one table is required")
    if not corpus.questions:
        raise ValueError("At least one question is required")
    if not corpus.triples:
        raise ValueError("At least one KG triple is required")

    for table in corpus.tables.values():
        if not table.rows:
            raise ValueError(f"Table {table.name} has no rows")
        columns = set(table.rows[0])
        for idx, row in enumerate(table.rows):
            if set(row) != columns:
                raise ValueError(f"Table {table.name} row {idx} has inconsistent columns")

    question_ids = set()
    for question in corpus.questions:
        if question.id in question_ids:
            raise ValueError(f"Duplicate question id: {question.id}")
        question_ids.add(question.id)
        if question.table not in corpus.tables:
            raise ValueError(f"Question {question.id} references unknown table {question.table}")
        if not question.answer:
            raise ValueError(f"Question {question.id} has empty answer")


def _load_tables(tables_dir: Path) -> dict[str, Table]:
    if not tables_dir.exists():
        raise FileNotFoundError(f"Tables directory does not exist: {tables_dir}")
    tables: dict[str, Table] = {}
    for csv_path in sorted(tables_dir.glob("*.csv")):
        name = csv_path.stem
        tables[name] = Table(name=name, rows=_load_csv(csv_path))
    if not tables:
        raise ValueError(f"No CSV tables found in {tables_dir}")
    return tables


def _load_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        raise FileNotFoundError(path)
    with path.open(newline="", encoding="utf-8") as handle:
        return [dict(row) for row in csv.DictReader(handle)]


def _load_json(path: Path) -> Any:
    if not path.exists():
        raise FileNotFoundError(path)
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)
