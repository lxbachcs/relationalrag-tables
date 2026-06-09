"""Fine-tune ONE cross-encoder jointly on BIRD value-linking and HybridQA
evidence-location, then test whether it wins both regimes.

Falsifiable bar for a "universal winner": the same fine-tuned model must reach
BIRD recall@1 >= 0.92 (the feature-MLP verifier) AND HybridQA recall@1 > 0.25
(lexical). We start from the ms-marco MiniLM cross-encoder and fine-tune with a
manual BCE loop on Apple MPS. Run: ``env/bin/python -m relational_rag_real.finetune_ce``.
"""

from __future__ import annotations

import json
import random
import time
from pathlib import Path

import numpy as np

from . import bird as B
from . import hybridqa as H

MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"
RESULTS = Path(__file__).resolve().parents[1] / "artifacts" / "crossencoder" / "results.json"


def _cov(qtok: set, text: str) -> float:
    t = set(H.normalize(text).split())
    return len(qtok & t) / len(qtok) if (qtok and t) else 0.0


def hybridqa_pairs(n_q: int, seed: int = 0, neg: int = 8):
    rng = random.Random(seed)
    pairs = []
    for q in rng.sample(H.questions("train"), n_q):
        try:
            units = H.table_units(q["table_id"])
        except FileNotFoundError:
            continue
        gold = {u.uid for u in H.answer_bearing_units(q, units)}
        if not gold:
            continue
        qtok = set(H.normalize(q["question"]).split())
        negs = sorted((u for u in units if u.uid not in gold), key=lambda u: -_cov(qtok, u.text))[:neg]
        for u in units:
            if u.uid in gold:
                pairs.append((q["question"], u.text, 1.0))
        for u in negs:
            pairs.append((q["question"], u.text, 0.0))
    return pairs


def bird_split(seed: int = 0, test_frac: float = 0.35):
    inst = B.value_link_instances(B.questions())
    rng = random.Random(seed)
    rng.shuffle(inst)
    cut = int(len(inst) * (1 - test_frac))
    return inst[:cut], inst[cut:]


def bird_pairs(instances, neg: int = 12, seed: int = 0):
    rng = random.Random(seed)
    pairs = []
    for i in instances:
        q = f"{i['question']} {i['evidence']}"
        gold = B.normalize(i["gold"])
        qtok = set(B.normalize(i["question"]).split())
        others = [c for c in i["candidates"] if B.normalize(c) != gold]
        others.sort(key=lambda c: -_cov(qtok, c))
        chosen = others[: neg // 2] + (rng.sample(others, min(len(others), neg)) if others else [])
        pairs.append((q, f"{i['column']}: {i['gold']}", 1.0))
        for c in dict.fromkeys(chosen):
            pairs.append((q, f"{i['column']}: {c}", 0.0))
    return pairs


def train(pairs, epochs: int = 2, bs: int = 32, lr: float = 2e-5):
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    dev = "mps" if torch.backends.mps.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL, num_labels=1).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    lossfn = torch.nn.BCEWithLogitsLoss()
    rng = random.Random(0)
    model.train()
    for ep in range(epochs):
        rng.shuffle(pairs)
        tot = 0.0
        for s in range(0, len(pairs), bs):
            b = pairs[s : s + bs]
            enc = tok([a for a, _, _ in b], [c for _, c, _ in b], padding=True,
                      truncation=True, max_length=160, return_tensors="pt").to(dev)
            y = torch.tensor([l for *_, l in b], dtype=torch.float, device=dev)
            loss = lossfn(model(**enc).logits.squeeze(-1), y)
            loss.backward(); opt.step(); opt.zero_grad()
            tot += loss.item()
        print(f"  epoch {ep + 1}/{epochs} loss={tot / (len(pairs) / bs):.4f}", flush=True)
    model.eval()
    return tok, model, dev


def _score(tok, model, dev, pairs, bs: int = 64) -> np.ndarray:
    import torch

    out = []
    with torch.no_grad():
        for s in range(0, len(pairs), bs):
            b = pairs[s : s + bs]
            enc = tok([a for a, _ in b], [c for _, c in b], padding=True,
                      truncation=True, max_length=160, return_tensors="pt").to(dev)
            out.extend(model(**enc).logits.squeeze(-1).float().cpu().numpy().tolist())
    return np.array(out)


def eval_hybridqa(tok, model, dev, n_dev: int = 400, topk: int = 20, seed: int = 0) -> dict:
    rng = random.Random(seed)
    lex = ce = n = 0
    for q in rng.sample(H.questions("dev"), n_dev):
        try:
            units = H.table_units(q["table_id"])
        except FileNotFoundError:
            continue
        gold = {u.uid for u in H.answer_bearing_units(q, units)}
        if not gold:
            continue
        n += 1
        qtok = set(H.normalize(q["question"]).split())
        order = sorted(range(len(units)), key=lambda i: -_cov(qtok, units[i].text))[:topk]
        if units[order[0]].uid in gold:
            lex += 1
        sc = _score(tok, model, dev, [(q["question"], units[i].text) for i in order])
        if units[order[int(np.argmax(sc))]].uid in gold:
            ce += 1
    return {"n": n, "lexical_r1": round(lex / n, 4), "crossenc_r1": round(ce / n, 4)}


def eval_bird(tok, model, dev, test_inst, cap: int = 60, seed: int = 0) -> dict:
    rng = random.Random(seed)
    ce_all = ce_drift = n = nd = 0
    for i in test_inst:
        gold = B.normalize(i["gold"])
        qtok = set(B.normalize(i["question"]).split())
        others = [c for c in i["candidates"] if B.normalize(c) != gold]
        others.sort(key=lambda c: -_cov(qtok, c))
        pool = [i["gold"]] + others[:cap]
        rng.shuffle(pool)
        n += 1
        nd += int(i["drift"])
        q = f"{i['question']} {i['evidence']}"
        sc = _score(tok, model, dev, [(q, f"{i['column']}: {c}") for c in pool])
        if B.normalize(pool[int(np.argmax(sc))]) == gold:
            ce_all += 1
            ce_drift += int(i["drift"])
    return {"n": n, "drift": nd, "crossenc_all": round(ce_all / n, 4),
            "crossenc_drift": round(ce_drift / nd, 4) if nd else 0.0}


def main(n_hq_train: int = 1500, n_dev: int = 400, epochs: int = 2, seed: int = 0) -> dict:
    t0 = time.time()
    bird_train, bird_test = bird_split(seed)
    pairs = hybridqa_pairs(n_hq_train, seed) + bird_pairs(bird_train, seed=seed)
    print(f"train pairs: {len(pairs)} (HybridQA + BIRD), bird_test={len(bird_test)}", flush=True)
    tok, model, dev = train(pairs, epochs=epochs)
    hq = eval_hybridqa(tok, model, dev, n_dev=n_dev, seed=seed)
    bd = eval_bird(tok, model, dev, bird_test, seed=seed)
    results = {"config": {"model": MODEL, "device": dev, "epochs": epochs, "train_pairs": len(pairs),
                          "runtime_s": round(time.time() - t0, 1)},
               "hybridqa": hq, "bird": bd,
               "wins_both": bool(bd["crossenc_all"] >= 0.92 and hq["crossenc_r1"] > 0.25)}
    RESULTS.parent.mkdir(parents=True, exist_ok=True)
    RESULTS.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print("HybridQA:", hq, flush=True)
    print("BIRD:", bd, flush=True)
    print("wins_both (BIRD>=0.92 AND HybridQA>0.25):", results["wins_both"], flush=True)
    return results


if __name__ == "__main__":
    main()
