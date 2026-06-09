"""Deep-ensemble uncertainty head.

Realises the paper's deep-ensemble estimator (Eq. ``unc``). We train ``K``
independently seeded verifier members on bootstrap resamples of the join-pair
data. At inference the ensemble exposes:

* ``score`` — the mean match probability (used to gate joins), and
* ``uncertainty`` — the predictive variance over the per-step operation
  distribution ``{accept, reject}`` for the chosen join, which the planner
  thresholds against ``tau_u`` to abstain.

For a binary operation the variance reduces to ``2 * Var_k(p_k)`` where ``p_k``
is member ``k``'s accept probability; we report exactly that quantity.
"""

from __future__ import annotations

import json
import math
import random
from dataclasses import dataclass
from pathlib import Path

from .verify import LearnedJoinVerifier


def _binary_entropy(p: float) -> float:
    p = min(max(p, 1e-12), 1.0 - 1e-12)
    return -(p * math.log(p) + (1.0 - p) * math.log(1.0 - p))


@dataclass
class EnsembleVerifier:
    """An ensemble of learned join verifiers with predictive-variance output."""

    members: list[LearnedJoinVerifier]

    @property
    def k(self) -> int:
        return len(self.members)

    def member_scores(self, lhs_value: str, rhs_id: str) -> list[float]:
        return [member.score(lhs_value, rhs_id) for member in self.members]

    def score(self, lhs_value: str, rhs_id: str) -> float:
        scores = self.member_scores(lhs_value, rhs_id)
        return sum(scores) / len(scores) if scores else 0.0

    def variance(self, lhs_value: str, rhs_id: str) -> float:
        """Epistemic predictive variance over {accept, reject} (paper Eq. ``unc``).

        For a binary operation this equals ``2 * Var_k(p_k)``. It captures
        *disagreement* among members and is reported as the calibration
        diagnostic, but on aleatoric (label-noise) regions members agree, so it
        is a weak error predictor on its own.
        """
        scores = self.member_scores(lhs_value, rhs_id)
        if len(scores) < 2:
            return 0.0
        mean = sum(scores) / len(scores)
        var = sum((value - mean) ** 2 for value in scores) / len(scores)
        return 2.0 * var

    def entropy(self, lhs_value: str, rhs_id: str) -> float:
        """Total predictive uncertainty: entropy of the ensemble-mean decision."""
        return _binary_entropy(self.score(lhs_value, rhs_id))

    def uncertainty(self, lhs_value: str, rhs_id: str) -> float:
        """Abstention signal used by the planner (total uncertainty = entropy)."""
        return self.entropy(lhs_value, rhs_id)

    # Diverse member architectures so independent initialisations disagree in the
    # ambiguous region (epistemic variance), per the deep-ensemble rationale.
    _ARCHS = [(32, 16), (24,), (48, 24), (16, 16), (40,), (32, 32, 16)]

    @classmethod
    def train(
        cls,
        pairs: list[tuple[str, str, bool]],
        k: int = 5,
        seed: int = 0,
        bootstrap: bool = True,
        subsample: float = 0.8,
        diverse: bool = True,
    ) -> "EnsembleVerifier":
        members: list[LearnedJoinVerifier] = []
        for member_idx in range(k):
            member_seed = seed + member_idx
            sample = _resample(pairs, member_seed, bootstrap, subsample)
            hidden = cls._ARCHS[member_idx % len(cls._ARCHS)] if diverse else (32, 16)
            members.append(LearnedJoinVerifier.train(sample, seed=member_seed, hidden=hidden))
        return cls(members=members)

    def save(self, out_dir: str | Path) -> None:
        path = Path(out_dir)
        path.mkdir(parents=True, exist_ok=True)
        for idx, member in enumerate(self.members):
            member.save(path / f"member_{idx}.pkl")
        with (path / "manifest.json").open("w", encoding="utf-8") as handle:
            json.dump({"k": self.k, "members": [f"member_{i}.pkl" for i in range(self.k)]}, handle, indent=2)

    @classmethod
    def load(cls, out_dir: str | Path) -> "EnsembleVerifier":
        path = Path(out_dir)
        with (path / "manifest.json").open(encoding="utf-8") as handle:
            manifest = json.load(handle)
        members = [LearnedJoinVerifier.load(path / name) for name in manifest["members"]]
        return cls(members=members)


def _resample(
    pairs: list[tuple[str, str, bool]], seed: int, bootstrap: bool, subsample: float
) -> list[tuple[str, str, bool]]:
    if len(pairs) <= 1:
        return list(pairs)
    rng = random.Random(seed)
    size = max(2, int(len(pairs) * subsample))
    if bootstrap:
        sample = [rng.choice(pairs) for _ in range(size)]
    else:
        sample = rng.sample(pairs, size)
    if len({label for _, _, label in sample}) < 2:
        return list(pairs)
    return sample
