"""Парная статистика при равной стоимости.

Для каждого сида своя кривая каждого семейства (по параметру запаса); потери семейства
считаются при бюджете под-секунд, равном стоимости эталона в этом же сиде (A3 или B8).
Сравнение семейств: парный критерий Уилкоксона по 10 сидам, поправка Холма по всем сравнениям.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from experiments.analyze import holm  # noqa: E402,F401  (повторное использование)

RES = ROOT / "results"
PAIRS = [("ПАБ LSTM v2", "HPA+LC"), ("ПАБ LSTM v2", "ПАБ naive"), ("ПАБ GRU v2", "ПАБ naive"),
         ("ПАБ LSTM v2", "ПАБ AR"), ("ПАБ naive", "HPA+LC")]
PROFS = ["S3", "S4", "S5", "S6"]


def err_at(curve, budget):
    c = curve.sort_values("pod_s")
    x, y = c["pod_s"].values, np.minimum.accumulate(c["err_rate"].values)
    if budget < x.min() or budget > x.max():
        return np.nan
    return float(np.interp(budget, x, y))


def main():
    pr = pd.read_csv(RES / "pareto_runs.csv")
    runs = pd.read_csv(RES / "runs.csv")
    rows = []
    for prof in PROFS:
        for seed in range(10):
            budget = float(runs[(runs.profile == prof) & (runs.seed == seed) & (runs.config == "A3")].pod_s.iloc[0])
            for fam in pr.family.unique():
                cur = pr[(pr.profile == prof) & (pr.seed == seed) & (pr.family == fam)]
                rows.append(dict(profile=prof, seed=seed, family=fam, err=err_at(cur, budget)))
    d = pd.DataFrame(rows)
    out = []
    for prof in PROFS:
        for a, b in PAIRS:
            x = d[(d.profile == prof) & (d.family == a)].sort_values("seed").err.values
            y = d[(d.profile == prof) & (d.family == b)].sort_values("seed").err.values
            ok = ~(np.isnan(x) | np.isnan(y))
            x, y = x[ok], y[ok]
            if len(x) < 6 or np.allclose(x, y):
                p = np.nan
            else:
                p = wilcoxon(x, y).pvalue
            out.append(dict(profile=prof, A=a, B=b, n=len(x), med_A=np.median(x), med_B=np.median(y),
                            rel_diff=(np.median(x) - np.median(y)) / max(np.median(y), 1e-9), p=p))
    t = pd.DataFrame(out)
    valid = t.p.notna()
    t.loc[valid, "p_holm"] = holm(t.loc[valid, "p"].values)
    t.to_csv(RES / "pareto_paired_stats.csv", index=False)
    pd.set_option("display.width", 220)
    print(t.round(4).to_string(index=False))


if __name__ == "__main__":
    main()
