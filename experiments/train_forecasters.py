"""Обучение прогнозных моделей на синтетических трассах (сиды 1000+, отдельно от тестовых 0-9
и валидационных 2000-2009).

Каждая обучающая трасса делится хронологически: первые 60 % - обучение, следующие 20 % -
валидация (выбор лучшей эпохи), последние 20 % - отложенный тест качества прогноза.

Запуск: python experiments/train_forecasters.py [ar lstm gru lstm2 gru2]   (по умолчанию все)
"""
import json
import pickle
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from forecast.models import DT, ARRidge, NNForecaster, NNForecaster2  # noqa: E402
from sim.profiles import PROFILES, make_arrivals  # noqa: E402

TRAIN_PROFILES = ["S1", "S2", "S3", "S4", "S5", "S6"]
N_TRAIN_SEEDS = 30
SEED0 = 1000
CACHE = ROOT / "results" / "cache"


def bins_of(profile, seed):
    lam, _ = PROFILES[profile](seed)
    return make_arrivals(lam, seed).reshape(-1, DT).mean(axis=1)


def load_splits():
    tr, va, te = [], [], []
    for p in TRAIN_PROFILES:
        for s in range(SEED0, SEED0 + N_TRAIN_SEEDS):
            b = bins_of(p, s)
            n = len(b)
            tr.append(b[: int(0.6 * n)])
            va.append(b[int(0.6 * n): int(0.8 * n)])
            te.append(b[int(0.8 * n):])
    return tr, va, te


def main(which):
    CACHE.mkdir(parents=True, exist_ok=True)
    tr, va, te = load_splits()
    print(f"train traces: {len(tr)}, val {len(va)}, holdout {len(te)}; models: {which}", flush=True)
    info_path = ROOT / "results" / "train_info.json"
    info = json.load(open(info_path)) if info_path.exists() else {}
    if "ar" in which:
        with open(CACHE / "ar.pkl", "wb") as f:
            pickle.dump(ARRidge().fit(tr), f)
    for kind in [m for m in which if m in ("lstm", "gru")]:
        t0 = time.time()
        m = NNForecaster(kind=kind, seed=0, epochs=40).fit(tr, va)
        m.save(ROOT / "forecast" / f"{kind}.pt")
        hist = np.array(m.history)
        np.savetxt(ROOT / "results" / f"train_history_{kind}.csv", hist, delimiter=",",
                   header="epoch,train_loss,val_loss", comments="")
        info[kind] = {"seconds": round(time.time() - t0, 1), "best_val_loss": float(hist[:, 2].min())}
        print(kind, info[kind], flush=True)
    for name in [m for m in which if m in ("lstm2", "gru2")]:
        t0 = time.time()
        m = NNForecaster2(kind=name[:-1], seed=0).fit(tr, va)
        m.save(ROOT / "forecast" / f"{name}.pt")
        hist = np.array(m.history)
        np.savetxt(ROOT / "results" / f"train_history_{name}.csv", hist, delimiter=",",
                   header="member,epoch,train_loss,val_loss", comments="")
        best = [float(hist[hist[:, 0] == i][:, 3].min()) for i in range(m.n_members)]
        info[name] = {"seconds": round(time.time() - t0, 1), "best_val_loss_per_member": best}
        print(name, info[name], flush=True)
    # json может быть записан двумя параллельными процессами: перечитываем и сливаем
    cur = json.load(open(info_path)) if info_path.exists() else {}
    cur.update(info)
    with open(info_path, "w") as f:
        json.dump(cur, f, indent=2)


if __name__ == "__main__":
    main(sys.argv[1:] or ["ar", "lstm", "gru", "lstm2", "gru2"])
