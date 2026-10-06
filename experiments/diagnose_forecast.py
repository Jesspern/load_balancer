"""Диагностика: где LSTM/GRU проигрывают или выигрывают у naive.

Точки прогноза делятся по фазе (по наблюдаемой истории и по тому, что будет дальше):
  quiet   - всплеска впереди нет (будущий максимум <= 1.4 x уровня)
  onset0  - всплеск впереди, но в истории роста ещё нет (принципиально непредсказуем)
  rising  - всплеск впереди и рост в истории уже начался (последний бин > 1.2 x бин 3 шага назад)
Ошибка пика = (предсказанный максимум на горизонте - фактический максимум) / фактический максимум.
Только валидационные сиды 2000-2009.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from experiments import run_experiments as rx  # noqa: E402
from forecast import models as fm  # noqa: E402
from sim.profiles import PROFILES, make_arrivals  # noqa: E402


def main(models=("naive", "ar", "lstm", "gru"), seeds=range(2000, 2010), profiles=("S2", "S3", "S4", "S5")):
    rx._load_models()
    rows = []
    for prof in profiles:
        for seed in seeds:
            lam, _ = PROFILES[prof](seed)
            bins = make_arrivals(lam, seed).reshape(-1, fm.DT).mean(axis=1)
            tr = {m: rx._models[m].track(bins)[0] for m in models}
            for k in range(fm.W, len(bins) - fm.H):
                hist = bins[k - fm.W:k]
                fut = bins[k:k + fm.H]
                lvl = hist[-6:].mean()
                if fut.max() > 1.4 * lvl and fut.max() - lvl > 30:
                    phase = "rising" if hist[-1] > 1.2 * hist[-4] else "onset0"
                else:
                    phase = "quiet"
                for m in models:
                    pk = tr[m][k].max()
                    rows.append(dict(profile=prof, phase=phase, model=m,
                                     mae=np.abs(tr[m][k] - fut).mean(),
                                     peak_err=(pk - fut.max()) / fut.max(),
                                     under=float(pk < fut.max() * 0.85)))
    d = pd.DataFrame(rows)
    pd.set_option("display.width", 200)
    g = d.groupby(["profile", "phase", "model"]).agg(n=("mae", "size"), mae=("mae", "mean"),
                                                        peak_err=("peak_err", "mean"),
                                                        under=("under", "mean")).round(3)
    print(g.to_string())
    return d


if __name__ == "__main__":
    main()
