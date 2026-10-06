"""Калибровка порога fallback на ВАЛИДАЦИОННЫХ сидах 2000-2009 (не пересекаются с тестовыми 0-9).

Для каждой модели прогноза отдельно (масштаб ошибки у моделей разный): порог = 99-й перцентиль
скользящей ошибки на обычных профилях S1-S6 (то есть на обычной нагрузке fallback должен
срабатывать не чаще ~1 % времени). Сдвинутый профиль S5X при калибровке не используется.
"""
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from experiments import run_experiments as rx  # noqa: E402
from forecast import models as fm  # noqa: E402
from sim.cluster import SimParams  # noqa: E402
from sim.profiles import PROFILES, make_arrivals  # noqa: E402

MODELS = ["naive", "linear", "ar", "lstm", "gru", "lstm2", "gru2"]
PROFS = ["S1", "S2", "S3", "S4", "S5", "S6", "S5X"]


def err_series(bins, lam_hat, mu, k_win):
    floor = 2 * mu * 0.4
    e = np.array([abs(lam_hat[k - 1][0] - bins[k - 1]) / max(bins[k - 1], floor)
                  for k in range(2, len(bins))])
    return np.array([e[max(0, i - k_win + 1): i + 1].mean() for i in range(k_win - 1, len(e))])


def main():
    rx._load_models()
    P = SimParams()
    traces = {}
    for prof in PROFS:
        traces[prof] = []
        for seed in range(2000, 2010):
            lam, _ = PROFILES[prof](seed)
            traces[prof].append(make_arrivals(lam, seed).reshape(-1, fm.DT).mean(axis=1))
    eps, detail = {}, {}
    for model in MODELS:
        pooled, per = [], {}
        for prof in PROFS:
            v = np.concatenate([err_series(b, rx._models[model].track(b)[0], P.mu, P.fb_k)
                                for b in traces[prof]])
            per[prof] = {"p50": round(float(np.percentile(v, 50)), 3),
                         "p99": round(float(np.percentile(v, 99)), 3)}
            if prof != "S5X":
                pooled.append(v)
        eps[model] = round(float(np.percentile(np.concatenate(pooled), 99)), 3)
        detail[model] = per
    out = {"fb_eps": eps, "per_model_profile": detail,
           "rule": "99th percentile of S1-S6 validation errors, per model"}
    with open(ROOT / "results" / "fb_calibration.json", "w") as f:
        json.dump(out, f, indent=2)
    print(json.dumps(eps, indent=2))


if __name__ == "__main__":
    main()
