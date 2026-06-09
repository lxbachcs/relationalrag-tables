"""Shared data structures for the reference pipeline."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class Table:
    name: str
    rows: list[dict[str, str]]


@dataclass(frozen=True)
class Triple:
    subject: str
    relation: str
    object: str


@dataclass(frozen=True)
class Question:
    id: str
    text: str
    answer: str
    task: str
    table: str = "accounts"
    lookup_column: str | None = None
    lookup_value: str | None = None
    return_column: str | None = None
    join_column: str | None = None
    traverse_relation: str | None = None
    candidate_order: list[str] = field(default_factory=list)
    gold_join: dict[str, str] | None = None
    # Extended fields for the full benchmark pipeline (default-compatible with smoke).
    traverse_chain: list[str] = field(default_factory=list)
    hops: int = 1
    split: str = "test"
    should_abstain: bool = False


@dataclass(frozen=True)
class Corpus:
    tables: dict[str, Table]
    triples: list[Triple]
    questions: list[Question]
    join_labels: dict[tuple[str, str], bool]
    config: dict[str, Any]


@dataclass(frozen=True)
class IndexEntry:
    kind: str
    id: str
    text: str
    payload: dict[str, Any]
    vector: list[float]


@dataclass(frozen=True)
class Retrieved:
    cells: list[IndexEntry]
    columns: list[IndexEntry]
    edges: list[IndexEntry]


@dataclass(frozen=True)
class Operation:
    type: str
    args: dict[str, Any]


@dataclass
class TraceEvent:
    operation: Operation
    output: Any = None
    verifier_score: float | None = None
    accepted: bool | None = None


@dataclass
class RunResult:
    question_id: str
    question: str
    prediction: str
    gold: str
    status: str
    trace: list[TraceEvent]
    retrieved_values: set[str]
    latency_ms: dict[str, float]
