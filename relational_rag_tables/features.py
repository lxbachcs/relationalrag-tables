"""Feature extraction for the learned join verifier.

The paper's verifier scores a candidate join ``(lhs, rhs)`` with a sigmoid over
a learned combination of representations (Eq. ``ver``). In this self-contained
artifact we materialise an explicit, interpretable feature vector over the two
identifiers and let a small MLP learn the combination. The features deliberately
span the fuzzy-matching regime the paper highlights (e.g. ``ICD-10 J45.901`` vs.
``J45.9``): exact match, containment, digit-suffix agreement, character n-gram
overlap, edit similarity, and an embedding cosine.
"""

from __future__ import annotations

from difflib import SequenceMatcher

from .index import cosine, embed_text, normalize_identifier

FEATURE_NAMES = [
    "exact_match",
    "containment",
    "digit_suffix_match",
    "digit_full_match",
    "char3_jaccard",
    "edit_ratio",
    "len_ratio",
    "prefix_overlap",
    "embed_cosine",
    "token_jaccard",
]


def _digits(text: str) -> str:
    return "".join(char for char in text if char.isdigit())


def _char_ngrams(text: str, n: int = 3) -> set[str]:
    if len(text) < n:
        return {text} if text else set()
    return {text[i : i + n] for i in range(len(text) - n + 1)}


def _jaccard(left: set[str], right: set[str]) -> float:
    if not left and not right:
        return 0.0
    union = left | right
    return len(left & right) / len(union) if union else 0.0


def _prefix_overlap(left: str, right: str) -> float:
    limit = min(len(left), len(right))
    shared = 0
    for i in range(limit):
        if left[i] != right[i]:
            break
        shared += 1
    longest = max(len(left), len(right))
    return shared / longest if longest else 0.0


def join_features(lhs_value: str, rhs_id: str) -> list[float]:
    """Return the fixed-length feature vector for a candidate join pair."""
    lhs = normalize_identifier(lhs_value)
    rhs = normalize_identifier(rhs_id)

    exact = 1.0 if lhs == rhs and lhs else 0.0
    containment = 1.0 if lhs and rhs and (lhs in rhs or rhs in lhs) else 0.0

    lhs_digits, rhs_digits = _digits(lhs), _digits(rhs)
    suffix = lhs_digits[-4:]
    digit_suffix = 1.0 if suffix and suffix == rhs_digits[-4:] else 0.0
    digit_full = 1.0 if lhs_digits and lhs_digits == rhs_digits else 0.0

    char_jac = _jaccard(_char_ngrams(lhs), _char_ngrams(rhs))
    edit_ratio = SequenceMatcher(None, lhs, rhs).ratio()
    len_ratio = (
        min(len(lhs), len(rhs)) / max(len(lhs), len(rhs))
        if max(len(lhs), len(rhs))
        else 0.0
    )
    prefix = _prefix_overlap(lhs, rhs)
    embed_cos = cosine(embed_text(lhs_value), embed_text(rhs_id))
    token_jac = _jaccard(set(lhs_value.lower().split()), set(rhs_id.lower().split()))

    return [
        exact,
        containment,
        digit_suffix,
        digit_full,
        char_jac,
        edit_ratio,
        len_ratio,
        prefix,
        embed_cos,
        token_jac,
    ]
