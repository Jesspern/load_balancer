"""Сравнение при РАВНОЙ СТОИМОСТИ: кривые "потери - под-секунды" для каждого семейства методов.

Любой метод можно купить лучшее качество за большее число подов (ниже целевая загрузка HPA,
больше запас rho у ПАБ). Поэтому сравнивать точки с разной стоимостью некорректно. Здесь для каждого
семейства меняется параметр запаса, и все точки показаны на плоскости (под-секунды, потери).
Тестовые сиды 0-9. Параметры не подбираются: показана вся кривая.
"""
import sys
from concurrent.futures import ProcessPoolExecutor
from dataclasses import replace
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from experiments import run_experiments as rx  # noqa: E402
from forecast import models as fm  # noqa: E402
from sim.cluster import Config, ForecastTrack, SimParams, Simulator  # noqa: E402
from sim.metrics import summarize  # noqa: E402
from sim.profiles import PROFILES, make_arrivals, PROFILE_TITLES  # noqa: E402

PROFS = ["S3", "S4", "S5", "S6", "S5X"]
HPA_TARGETS = [0.35, 0.45, 0.55, 0.65, 0.75, 0.85]
RHOS = [0.5, 0.6, 0.7, 0.8, 0.9]
FAMILIES = {"HPA+LC": None, "ПАБ naive": "naive", "ПАБ linear": "linear", "ПАБ AR": "ar",
            "ПАБ LSTM v2": "lstm2", "ПАБ GRU v2": "gru2"}
RES = ROOT / "results"


def work(args):
    prof, seed = args
    if not rx._models:
        rx._load_models()
    P0 = SimParams()
    lam, bursts = PROFILES[prof](seed)
    arr = make_arrivals(lam, seed)
    bins = arr.reshape(-1, fm.DT).mean(axis=1)
    tracks = {m: ForecastTrack(*rx._models[m].track(bins)) for m in set(FAMILIES.values()) if m}
    rows = []

    def run(c, P, fc, fam, par):
        s = summarize(Simulator(c, P, arr, lam, forecast=fc).run(), lam, bursts, P0)
        s.update(profile=prof, seed=seed, family=fam, param=par)
        rows.append(s)

    for t in HPA_TARGETS:
        run(Config("h", "lc", "hpa", hpa_overrides={"hpa_target": t}), P0, None, "HPA+LC", t)
    for fam, m in FAMILIES.items():
        if m is None:
            continue
        for rho in RHOS:
            P = replace(P0, fb_eps=rx._fb_eps(m), rho=rho)
            run(Config("b", "wlc", "pab", m), P, tracks[m], fam, rho)
    return rows


def err_at_budget(curve, budget):
    """Потери семейства при заданных под-секундах (линейная интерполяция по нижней огибающей)."""
    c = curve.sort_values("pod_s")
    x, y = c["pod_s"].values, np.minimum.accumulate(c["err_rate"].values)
    if budget < x.min() or budget > x.max():
        return np.nan
    return float(np.interp(budget, x, y))


def main():
    jobs = [(p, s) for p in PROFS for s in range(10)]
    rows = []
    with ProcessPoolExecutor(max_workers=10) as ex:
        for r in ex.map(work, jobs):
            rows += r
    d = pd.DataFrame(rows)
    d.to_csv(RES / "pareto_runs.csv", index=False)
    med = d.groupby(["profile", "family", "param"])[["err_rate", "slo_viol_s", "pod_s"]].median().reset_index()
    med.to_csv(RES / "pareto_medians.csv", index=False)

    # потери при бюджете, равном стоимости базового A3 (HPA, цель 0.7 ~ 0.65/0.75): берём прогон A3
    runs = pd.read_csv(RES / "runs.csv")
    out = []
    for prof in PROFS:
        a3 = runs[(runs.profile == prof) & (runs.config == "A3")]
        budget = float(a3["pod_s"].median())
        row = {"profile": prof, "budget_pod_s": round(budget), "A3_err": round(float(a3["err_rate"].median()), 4)}
        for fam in FAMILIES:
            cur = med[(med.profile == prof) & (med.family == fam)]
            row[fam] = round(err_at_budget(cur, budget), 4)
        out.append(row)
    tab = pd.DataFrame(out)
    tab.to_csv(RES / "pareto_equal_cost.csv", index=False)
    pd.set_option("display.width", 220)
    print("Потери (доля) при бюджете подов = медиана A3 на профиле:\n", tab.to_string(index=False))

    fig, axes = plt.subplots(1, len(PROFS), figsize=(4.2 * len(PROFS), 4))
    cols = {"HPA+LC": "#d62728", "ПАБ naive": "#bcbd22", "ПАБ linear": "#e377c2", "ПАБ AR": "#2ca02c",
            "ПАБ LSTM v2": "#1f77b4", "ПАБ GRU v2": "#17becf"}
    for ax, prof in zip(axes, PROFS):
        for fam in FAMILIES:
            c = med[(med.profile == prof) & (med.family == fam)].sort_values("pod_s")
            ax.plot(c.pod_s, c.err_rate, "-o", ms=3.5, color=cols[fam], label=fam)
        ax.set_title(PROFILE_TITLES[prof], fontsize=9)
        ax.set_xlabel("под-секунды")
        ax.set_ylabel("доля потерянных запросов")
    axes[0].legend(fontsize=7)
    fig.suptitle("Потери при разной стоимости: каждая точка - значение параметра запаса (медиана по 10 повторам)")
    fig.tight_layout()
    fig.savefig(RES / "figures" / "pareto.png")


if __name__ == "__main__":
    main()
