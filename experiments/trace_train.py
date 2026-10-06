"""Обучение прогнозных моделей на ОБУЧАЮЩИХ окнах реальной трассы (первые 60 % суток).
Валидационные окна (следующие 20 %) используются для выбора эпохи. Тестовые окна не трогаются.

Правило отбора окон (задано заранее и применяется только к обучающей и валидационной частям):
все окна, где максимум 5-секундного бина > 1.5 x медианы бина, плюс случайные 30 % остальных.

Запуск: python experiments/trace_train.py [ar lstm2 gru2]
"""
import json
import pickle
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from forecast.models import DT, ARRidge, NNForecaster2  # noqa: E402
from sim import trace  # noqa: E402
from sim.profiles import make_arrivals  # noqa: E402

OUT = trace.OUT
TAG = trace.TAG


def window_bins(idx):
    lam, _ = trace.trace_window(idx)
    return make_arrivals(lam, idx).reshape(-1, DT).mean(axis=1)


def select(indices, seed):
    rng = np.random.default_rng(seed)
    out = []
    for i in indices:
        b = window_bins(i)
        if b.max() > 1.5 * np.median(b) or rng.random() < 0.3:
            out.append(b)
    return out


def main(which):
    OUT.mkdir(parents=True, exist_ok=True)
    sp = trace.split_indices()
    tr, va = select(sp["train"], 1), select(sp["val"], 2)
    print(f"train windows {len(tr)}/{len(sp['train'])}, val {len(va)}/{len(sp['val'])}; models {which}",
          flush=True)
    info = {}
    if "ar" in which:
        with open(ROOT / "results" / "cache" / f"{TAG}_ar.pkl", "wb") as f:
            pickle.dump(ARRidge().fit(tr), f)
    for name in [m for m in which if m in ("lstm2", "gru2")]:
        t0 = time.time()
        m = NNForecaster2(kind=name[:-1], seed=0).fit(tr, va)
        m.save(ROOT / "forecast" / f"{TAG}_{name}.pt")
        h = np.array(m.history)
        np.savetxt(OUT / f"train_history_{name}.csv", h, delimiter=",",
                   header="member,epoch,train_loss,val_loss", comments="")
        best = [float(h[h[:, 0] == i][:, 3].min()) for i in range(m.n_members)]
        info[name] = {"seconds": round(time.time() - t0, 1), "best_val_loss_per_member": best,
                      "train_windows": len(tr), "val_windows": len(va)}
        print(name, info[name], flush=True)
    p = OUT / "train_info.json"
    cur = json.load(open(p)) if p.exists() else {}
    cur.update(info)
    json.dump(cur, open(p, "w"), indent=2)


if __name__ == "__main__":
    main(sys.argv[1:] or ["ar", "lstm2", "gru2"])
