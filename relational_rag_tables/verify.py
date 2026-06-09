"""Join verifier interfaces, deterministic smoke verifier, and learned verifier.

``DeterministicJoinVerifier`` keeps the dependency-free smoke path working.
``LearnedJoinVerifier`` realises the paper's learned join verifier: a small MLP
trained with cross-entropy on gold-positive join pairs and hard negatives
(near-duplicate identifiers), scoring a candidate pair with a sigmoid match
probability via ``score(lhs, rhs)`` — the exact interface the planner consumes.
"""

from __future__ import annotations

import pickle
from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol

from .features import join_features
from .index import normalize_identifier


class JoinVerifier(Protocol):
    def score(self, lhs_value: str, rhs_id: str) -> float:
        """Return a match probability-like score in [0, 1]."""


@dataclass
class DeterministicJoinVerifier:
    """A small verifier that mimics learned fuzzy matching for fixtures."""

    label_overrides: dict[tuple[str, str], bool]

    def score(self, lhs_value: str, rhs_id: str) -> float:
        key = (lhs_value, rhs_id)
        if key in self.label_overrides:
            return 0.95 if self.label_overrides[key] else 0.05

        lhs = normalize_identifier(lhs_value)
        rhs = normalize_identifier(rhs_id)
        if lhs == rhs or lhs in rhs or rhs in lhs:
            return 0.88
        if _last_digits(lhs) and _last_digits(lhs) == _last_digits(rhs):
            return 0.72
        return 0.15


@dataclass
class ExactMatchVerifier:
    """Brittle hand-crafted string-matching baseline (no learning).

    Scores 1.0 when one normalised identifier contains the other, else 0.0 —
    the kind of exact/substring tool call the paper argues is fragile under
    formatting drift and corrupted identifiers.
    """

    def score(self, lhs_value: str, rhs_id: str) -> float:
        lhs = normalize_identifier(lhs_value)
        rhs = normalize_identifier(rhs_id)
        if not lhs or not rhs:
            return 0.0
        return 1.0 if (lhs == rhs or lhs in rhs or rhs in lhs) else 0.0


@dataclass
class LearnedJoinVerifier:
    """Learned MLP verifier over explicit join-pair features (Eq. ``ver``)."""

    model: object  # a fitted sklearn estimator exposing predict_proba
    seed: int = 0

    def score(self, lhs_value: str, rhs_id: str) -> float:
        features = [join_features(lhs_value, rhs_id)]
        proba = self.model.predict_proba(features)[0]
        # Probability of the positive (match) class.
        classes = list(getattr(self.model, "classes_", [0, 1]))
        pos_index = classes.index(1) if 1 in classes else len(classes) - 1
        return float(proba[pos_index])

    @classmethod
    def train(
        cls,
        pairs: list[tuple[str, str, bool]],
        seed: int = 0,
        hidden: tuple[int, ...] = (32, 16),
        max_iter: int = 800,
    ) -> "LearnedJoinVerifier":
        """Train an MLP verifier on labelled ``(lhs, rhs, is_match)`` pairs."""
        from sklearn.neural_network import MLPClassifier
        from sklearn.pipeline import Pipeline
        from sklearn.preprocessing import StandardScaler

        features = [join_features(lhs, rhs) for lhs, rhs, _ in pairs]
        labels = [1 if is_match else 0 for _, _, is_match in pairs]
        model = Pipeline(
            [
                ("scale", StandardScaler()),
                (
                    "mlp",
                    MLPClassifier(
                        hidden_layer_sizes=hidden,
                        activation="relu",
                        alpha=1e-3,
                        max_iter=max_iter,
                        random_state=seed,
                    ),
                ),
            ]
        )
        model.fit(features, labels)
        return cls(model=model, seed=seed)

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("wb") as handle:
            pickle.dump({"model": self.model, "seed": self.seed}, handle)

    @classmethod
    def load(cls, path: str | Path) -> "LearnedJoinVerifier":
        with Path(path).open("rb") as handle:
            payload = pickle.load(handle)
        return cls(model=payload["model"], seed=int(payload.get("seed", 0)))


@dataclass
class TrainableVerifierStub:
    """Interface placeholder retained for backward compatibility."""

    model_uri: str | None = None

    def fit(self, pairs: list[tuple[str, str, bool]]) -> None:
        raise NotImplementedError("Use LearnedJoinVerifier.train instead")

    def score(self, lhs_value: str, rhs_id: str) -> float:
        raise NotImplementedError("Load a trained verifier or use DeterministicJoinVerifier")


def label_for(labels: dict[tuple[str, str], bool], lhs_value: str, rhs_id: str) -> bool | None:
    return labels.get((lhs_value, rhs_id))


def _last_digits(text: str) -> str:
    digits = "".join(char for char in text if char.isdigit())
    return digits[-4:] if len(digits) >= 4 else digits
