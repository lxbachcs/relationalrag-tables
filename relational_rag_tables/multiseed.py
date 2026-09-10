"""Multi-seed synthetic study with full risk-coverage curves (rebuttal item E2).

For each seed we (i) synthesise a fresh drifted corpus (corpus seed = verifier seed = s),
(ii) rerun the paper's experiment matrix, and (iii) add the rigor items reviewers asked
for: full 1%-step risk-coverage curves with AURC / E-AURC and bootstrap bands (pairwise
and system level), ECE, cost-aware utility, and thresholds selected on a disjoint
validation half and reported on the test half. Seeds are aggregated as mean, sd, and a
t-based 95% CI.

Run: ``python -m relational_rag_tables.multiseed --seeds 20 --workers 20``
"""

from __future__ import annotations

import os

os.environ.setdefault("OMP_NUM_THREADS", "2")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "2")
os.environ.setdefault("MKL_NUM_THREADS", "2")

import argparse
import json
import math
import random
import time
from multiprocessing import Pool
from pathlib import Path

from . import datagen
from .benchmark import evaluate_verifier, synthesize_corpus
from .calibration import _ece
from .experiments import run_matrix
from .index import build_entries
from .ingest import load_corpus
from .plan_execute import PlannerConfig, build_triple_index, exact_match, run_question
from .uncertainty import EnsembleVerifier
from .verify import ExactMatchVerifier

GRID = [i / 100 for i in range(1, 101)]
COVERAGE_POINTS = (1.0, 0.9, 0.8, 0.7, 0.5)
T975 = {2: 12.706, 3: 4.303, 4: 3.182, 5: 2.776, 6: 2.571, 7: 2.447, 8: 2.365, 9: 2.306,
        10: 2.262, 11: 2.228, 12: 2.201, 13: 2.179, 14: 2.160, 15: 2.145, 16: 2.131,
        17: 2.120, 18: 2.110, 19: 2.101, 20: 2.093, 21: 2.086, 25: 2.064, 30: 2.045}


# ----------------------------------------------------------------------------- curves
def _risk_at_grid(items):
    """items: list of (uncertainty, correct) sorted ascending by uncertainty."""
    n = len(items)
    errs = 0
    risk = []
    for k, (_, c) in enumerate(items, 1):
        errs += 1 - c
        risk.append(errs / k)
    return [risk[max(1, int(round(n * g))) - 1] for g in GRID], risk


def rc_curve(items, rng: random.Random | None = None, boots: int = 500) -> dict:
    items = sorted(enumerate(items), key=lambda t: (t[1][0], t[0]))
    items = [it for _, it in items]
    n = len(items)
    grid_risk, risk = _risk_at_grid(items)
    aurc = sum(risk) / n
    e = n - sum(c for _, c in items)
    opt = sum(max(0, k - (n - e)) / k for k in range(1, n + 1)) / n
    out = {"n": n, "errors": e, "aurc": round(aurc, 4), "e_aurc": round(aurc - opt, 4),
           "curve": [{"coverage": g, "risk": round(r, 4)} for g, r in zip(GRID, grid_risk)],
           "accuracy_at": {str(c): round(1 - grid_risk[int(round(c * 100)) - 1], 4) for c in COVERAGE_POINTS}}
    if rng is not None and boots > 0:
        aurcs, grids = [], []
        for _ in range(boots):
            sample = [items[rng.randrange(n)] for _ in range(n)]
            sample.sort(key=lambda t: t[0])
            g, r = _risk_at_grid(sample)
            aurcs.append(sum(r) / n)
            grids.append(g)
        aurcs.sort()
        lo, hi = int(0.025 * boots), max(0, int(0.975 * boots) - 1)
        out["aurc_ci95"] = [round(aurcs[lo], 4), round(aurcs[hi], 4)]
        band = []
        for gi, g in enumerate(GRID):
            col = sorted(gr[gi] for gr in grids)
            band.append({"coverage": g, "lo": round(col[lo], 4), "hi": round(col[hi], 4)})
        out["band95"] = band
    return out


def _pair_items(verifier, pairs, tau_v):
    items = []
    for lhs, rhs, is_match in pairs:
        p = verifier.score(lhs, rhs)
        pred = 1 if p >= tau_v else 0
        p = min(max(p, 1e-12), 1 - 1e-12)
        entropy = -(p * math.log(p) + (1 - p) * math.log(1 - p))
        items.append((entropy, 1 if pred == (1 if is_match else 0) else 0))
    return items


def _system_items(corpus, entries, ens, tau_v, top_k):
    idx = build_triple_index(corpus)
    cfg = PlannerConfig(tau_v=tau_v, top_k=top_k, use_verifier=True)
    items = []
    n_abstain = 0
    for q in corpus.questions:
        r = run_question(corpus, entries, q, ens, cfg, idx, None)
        if r.status != "completed":
            n_abstain += 1
            continue
        u = 0.0
        for e in r.trace:
            if e.operation.type == "join" and e.accepted:
                u = ens.uncertainty(str(e.operation.args["lhs_value"]), str(e.operation.args["rhs"]))
        items.append((u, 1 if exact_match(r.prediction, r.gold) else 0))
    return items, n_abstain


def _utility(items, lambdas=(1, 2, 5)):
    items = sorted(items, key=lambda t: t[0])
    n = len(items)
    out = {}
    for lam in lambdas:
        best = (-1e9, None)
        for c in (1.0, 0.9, 0.8, 0.7, 0.6, 0.5):
            keep = max(1, int(round(n * c)))
            correct = sum(a for _, a in items[:keep])
            util = (correct - lam * (keep - correct)) / n
            if util > best[0]:
                best = (util, c)
        full = (sum(a for _, a in items) - lam * (n - sum(a for _, a in items))) / n
        out[f"lambda_{lam}"] = {"best_coverage": best[1], "best_utility": round(best[0], 4),
                                "full_utility": round(full, 4)}
    return out


# ------------------------------------------------------------- validation thresholds
def _prf(scores, labels, tau):
    tp = fp = fn = 0
    for s, y in zip(scores, labels):
        p = 1 if s >= tau else 0
        tp += p and y
        fp += p and not y
        fn += (not p) and y
    prec = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
    return {"precision": round(prec, 4), "recall": round(rec, 4), "f1": round(f1, 4)}


def validation_thresholds(ens, pairs, seed: int, alphas=(0.02, 0.05, 0.10)) -> dict:
    rng = random.Random(seed + 12345)
    idx = list(range(len(pairs)))
    rng.shuffle(idx)
    half = len(idx) // 2
    cal, test = idx[:half], idx[half:]
    scores = [ens.score(a, b) for a, b, _ in pairs]
    labels = [1 if m else 0 for _, _, m in pairs]
    grid = [round(0.30 + 0.05 * i, 2) for i in range(9)]
    cal_f1 = {t: _prf([scores[i] for i in cal], [labels[i] for i in cal], t)["f1"] for t in grid}
    tau_sel = max(grid, key=lambda t: (cal_f1[t], -abs(t - 0.5)))
    tv = {"grid_cal_f1": {str(t): cal_f1[t] for t in grid}, "tau_v_selected": tau_sel,
          "test_at_selected": _prf([scores[i] for i in test], [labels[i] for i in test], tau_sel),
          "test_at_0.5": _prf([scores[i] for i in test], [labels[i] for i in test], 0.5)}
    # uncertainty threshold for a target risk, chosen on cal, reported on test
    def conf_err(i):
        p = scores[i]
        pred = 1 if p >= 0.5 else 0
        return max(p, 1 - p), 0 if pred == labels[i] else 1
    cal_items = sorted((conf_err(i) for i in cal), key=lambda t: -t[0])
    tu = []
    for alpha in alphas:
        err = n = 0
        thr = 1.01
        for c, e in cal_items:
            err += e
            n += 1
            if err / n <= alpha:
                thr = c
        acc = [e for c, e in (conf_err(i) for i in test) if c >= thr]
        risk = sum(acc) / len(acc) if acc else 0.0
        tu.append({"alpha": alpha, "confidence_threshold": round(thr, 4),
                   "test_coverage": round(len(acc) / len(test), 4), "test_risk": round(risk, 4),
                   "risk_held": bool(not acc or risk <= alpha)})
    return {"n_cal": len(cal), "n_test": len(test), "tau_v": tv, "risk_targets": tu}


# ------------------------------------------------------------------------ one seed
def run_seed(args) -> dict:
    seed, fixtures_root, out_dir, num_accounts, drift, k, boots = args
    t0 = time.time()
    fx = Path(fixtures_root) / f"seed_{seed}"
    if not (fx / "config.json").exists():
        synthesize_corpus(fx, num_accounts=num_accounts, seed=seed, qa_join_drift=drift)
    matrix = run_matrix(fx, ensemble_k=k, seed=seed)

    corpus = load_corpus(fx)
    entries = build_entries(corpus)
    tau_v = float(corpus.config["tau_v"])
    top_k = int(corpus.config["top_k"])
    train = datagen.load_verifier_pairs(fx)
    ev = datagen.load_verifier_eval(fx)
    ens = EnsembleVerifier.train(train, k=k, seed=seed)
    single = ens.members[0]
    exact = ExactMatchVerifier()
    rng = random.Random(seed + 777)

    ece = {}
    for name, v in (("exact_match", exact), ("learned", single), ("ensemble", ens)):
        s = [v.score(a, b) for a, b, _ in ev]
        y = [1 if m else 0 for _, _, m in ev]
        e, m, _ = _ece(s, y)
        ece[name] = {"ece": e, "mce": m, "brier": evaluate_verifier(v, ev, tau_v=tau_v)["brier"]}

    rc_pairs = {name: rc_curve(_pair_items(v, ev, tau_v), rng, boots)
                for name, v in (("exact_match", exact), ("learned", single), ("ensemble", ens))}
    sys_items, n_abstain = _system_items(corpus, entries, ens, tau_v, top_k)
    rc_system = rc_curve(sys_items, rng, boots)
    rc_system["n_abstained"] = n_abstain

    result = {
        "seed": seed,
        "meta": {"num_accounts": num_accounts, "qa_join_drift": drift, "ensemble_k": k,
                 "num_questions": len(corpus.questions), "n_verifier_train": len(train),
                 "n_verifier_eval": len(ev), "runtime_s": None},
        "matrix": {key: matrix[key] for key in ("qa_by_verifier", "verifier_discrimination",
                                                "selective_prediction", "ensemble_size_sweep")},
        "ece": ece,
        "risk_coverage_pairs": rc_pairs,
        "risk_coverage_system": rc_system,
        "cost_aware_utility": _utility(sys_items),
        "validation_thresholds": validation_thresholds(ens, ev, seed),
    }
    result["meta"]["runtime_s"] = round(time.time() - t0, 1)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / f"seed_{seed}.json").write_text(json.dumps(result, indent=1), encoding="utf-8")
    print(f"[seed {seed}] EM ens={matrix['qa_by_verifier']['ensemble']['em']} "
          f"exact={matrix['qa_by_verifier']['exact_match']['em']} "
          f"AUROC ens={matrix['verifier_discrimination']['ensemble']['auroc']} "
          f"AURC ens={rc_pairs['ensemble']['aurc']} sys={rc_system['aurc']} "
          f"({result['meta']['runtime_s']}s)", flush=True)
    return result


# ------------------------------------------------------------------------ aggregate
_SKIP = {"curve", "band95", "reliability", "confusion", "grid_cal_f1", "corpus", "meta", "seed",
         "aurc_ci95", "risk_targets", "latency_ms_avg", "by_task_total"}


def _flatten(obj, prefix=""):
    if isinstance(obj, dict):
        for key, val in obj.items():
            if key in _SKIP:
                continue
            yield from _flatten(val, f"{prefix}{key}/")
    elif isinstance(obj, list):
        for i, val in enumerate(obj):
            yield from _flatten(val, f"{prefix}{i}/")
    elif isinstance(obj, (int, float)) and not isinstance(obj, bool):
        yield prefix.rstrip("/"), float(obj)


def _stats(vals):
    n = len(vals)
    mean = sum(vals) / n
    sd = math.sqrt(sum((v - mean) ** 2 for v in vals) / (n - 1)) if n > 1 else 0.0
    t = T975.get(n, 1.96)
    half = t * sd / math.sqrt(n) if n > 1 else 0.0
    return {"mean": round(mean, 4), "sd": round(sd, 4), "ci95": [round(mean - half, 4), round(mean + half, 4)],
            "min": round(min(vals), 4), "max": round(max(vals), 4), "n": n}


def aggregate(results: list[dict]) -> dict:
    per_seed = [dict(_flatten(r)) for r in results]
    keys = set(per_seed[0])
    for d in per_seed[1:]:
        keys &= set(d)
    scalars = {k: _stats([d[k] for d in per_seed]) for k in sorted(keys)}
    curves = {}
    for family, getter in (("pairs/exact_match", lambda r: r["risk_coverage_pairs"]["exact_match"]),
                           ("pairs/learned", lambda r: r["risk_coverage_pairs"]["learned"]),
                           ("pairs/ensemble", lambda r: r["risk_coverage_pairs"]["ensemble"]),
                           ("system/ensemble", lambda r: r["risk_coverage_system"])):
        pts = []
        for gi, g in enumerate(GRID):
            vals = [getter(r)["curve"][gi]["risk"] for r in results]
            st = _stats(vals)
            pts.append({"coverage": g, "risk_mean": st["mean"], "risk_sd": st["sd"], "risk_ci95": st["ci95"]})
        curves[family] = pts
    risk_targets = {}
    for alpha_i in range(len(results[0]["validation_thresholds"]["risk_targets"])):
        rows = [r["validation_thresholds"]["risk_targets"][alpha_i] for r in results]
        alpha = rows[0]["alpha"]
        risk_targets[str(alpha)] = {"test_coverage": _stats([x["test_coverage"] for x in rows]),
                                    "test_risk": _stats([x["test_risk"] for x in rows]),
                                    "risk_held_frac": round(sum(x["risk_held"] for x in rows) / len(rows), 4)}
    tau_sel = [r["validation_thresholds"]["tau_v"]["tau_v_selected"] for r in results]
    return {"n_seeds": len(results), "seeds": [r["seed"] for r in results],
            "scalars": scalars, "curves_mean": curves, "risk_targets": risk_targets,
            "tau_v_selected_values": tau_sel}


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=20)
    ap.add_argument("--seed-list", type=str, default=None, help="comma-separated explicit seeds")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--num-accounts", type=int, default=600)
    ap.add_argument("--drift", type=float, default=0.6)
    ap.add_argument("--ensemble-k", type=int, default=5)
    ap.add_argument("--boots", type=int, default=500)
    ap.add_argument("--fixtures-root", default="fixtures/multiseed")
    ap.add_argument("--out", default="artifacts/multiseed")
    a = ap.parse_args(argv)
    seeds = [int(s) for s in a.seed_list.split(",")] if a.seed_list else list(range(a.seeds))
    jobs = [(s, a.fixtures_root, a.out, a.num_accounts, a.drift, a.ensemble_k, a.boots) for s in seeds]
    t0 = time.time()
    if a.workers > 1:
        with Pool(min(a.workers, len(jobs))) as pool:
            results = pool.map(run_seed, jobs)
    else:
        results = [run_seed(j) for j in jobs]
    results.sort(key=lambda r: r["seed"])
    agg = aggregate(results)
    agg["runtime_s"] = round(time.time() - t0, 1)
    Path(a.out).mkdir(parents=True, exist_ok=True)
    (Path(a.out) / "aggregate.json").write_text(json.dumps(agg, indent=1), encoding="utf-8")
    sc = agg["scalars"]
    for key in ("matrix/qa_by_verifier/exact_match/em", "matrix/qa_by_verifier/learned/em",
                "matrix/qa_by_verifier/ensemble/em", "matrix/qa_by_verifier/no_verifier/em",
                "matrix/verifier_discrimination/learned/auroc", "matrix/verifier_discrimination/ensemble/auroc",
                "ece/learned/ece", "ece/ensemble/ece", "risk_coverage_pairs/ensemble/aurc",
                "risk_coverage_pairs/learned/aurc", "risk_coverage_pairs/exact_match/aurc",
                "risk_coverage_system/aurc", "cost_aware_utility/lambda_5/best_utility"):
        if key in sc:
            print(f"{key}: {sc[key]['mean']} ± {sc[key]['sd']}  CI {sc[key]['ci95']}")
    print(f"aggregate -> {Path(a.out) / 'aggregate.json'} ({agg['runtime_s']}s)")
    return agg


if __name__ == "__main__":
    main()
