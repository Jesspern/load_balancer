"""Чувствительность выводов на реальной трассе к масштабу нагрузки K и времени запуска пода T_start.

Варьируется по одному параметру (остальные по умолчанию): целевой p99 = 200/400/800 запр/с
(масштаб K пересчитывается; модели НЕ переобучаются, обучены при 400, это проверка устойчивости
к сдвигу уровня) и T_start = 10/30/60 с. Тестовые окна, конфигурации A3, A6, B6, B8, B12.
Результаты: results/trace/sensitivity.csv и figures/trace_sensitivity.png.
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
from experiments import trace_run as tr  # noqa: E402
from forecast import models as fm  # noqa: E402
from sim import trace  # noqa: E402
from sim.cluster import ForecastTrack, SimParams, Simulator  # noqa: E402
from sim.metrics import summarize  # noqa: E402
from sim.profiles import make_arrivals  # noqa: E402

CFG = ["A3", "A6", "B6", "B8", "B12"]
SETTINGS = [("p99=200", 200.0, 30.0), ("p99=400 (осн.)", 400.0, 30.0), ("p99=800", 800.0, 30.0),
            ("T_start=10", 400.0, 10.0), ("T_start=60", 400.0, 60.0)]
p99_train = None


def work(args):
    idx, label, target, t_start = args
    models = tr.load_models()
    K = trace.scale_constant() * target / trace.TARGET_P99
    lam, bursts = trace.trace_window(idx, K=K)
    arr = make_arrivals(lam, idx)
    bins = arr.reshape(-1, fm.DT).mean(axis=1)
    P0 = SimParams(t_start=t_start)
    rows = []
    cache = {}
    for name in CFG:
        c = tr.CONFIGS[name]
        fc = None
        if c.scaler == "pab":
            if c.forecast not in cache:
                cache[c.forecast] = ForecastTrack(*models[c.forecast].track(bins))
            fc = cache[c.forecast]
        P = replace(P0, fb_eps=tr.fb_eps(c.forecast)) if c.scaler == "pab" else P0
        s = summarize(Simulator(c, P, arr, lam, forecast=fc).run(), lam, bursts, P0)
        s.update(window=idx, config=name, setting=label)
        rows.append(s)
    return rows


def pooled(d, cfg):
    g = d[d.config == cfg]
    return float((g.err_rate * g.arrived).sum() / g.arrived.sum()), float(g.pod_s.mean())


def main():
    test = trace.split_indices()["test"]
    jobs = [(i, lab, tg, ts) for lab, tg, ts in SETTINGS for i in test]
    rows = []
    with ProcessPoolExecutor(max_workers=10) as ex:
        for r in ex.map(work, jobs, chunksize=4):
            rows += r
    d = pd.DataFrame(rows)
    out = tr.OUT
    d.to_csv(out / "sensitivity_runs.csv", index=False)
    rng = np.random.default_rng(7)
    res = []
    for lab, _, _ in SETTINGS:
        g = d[d.setting == lab]
        w = np.array(sorted(g.window.unique()))
        for cfg in CFG:
            loss, pod = pooled(g, cfg)
            # парная разность B8 - A3 (бутстрэп по окнам)
            res.append({"setting": lab, "config": cfg, "loss": loss, "pod_s": pod})
        diffs = []
        for _ in range(1000):
            s = rng.choice(w, size=len(w), replace=True)
            def pl(cfg):
                x = g[g.config == cfg].set_index("window").loc[s]
                return (x.err_rate * x.arrived).sum() / x.arrived.sum()
            diffs.append((pl("B8") - pl("A3"), pl("B8") - pl("B6"), pl("B8") - pl("B12")))
        diffs = np.array(diffs)
        for j, nm in enumerate(("B8-A3", "B8-B6", "B8-B12")):
            res.append({"setting": lab, "config": nm, "loss": float(np.median(diffs[:, j])),
                        "pod_s": np.nan, "ci_lo": float(np.percentile(diffs[:, j], 2.5)),
                        "ci_hi": float(np.percentile(diffs[:, j], 97.5))})
    r = pd.DataFrame(res)
    r.to_csv(out / "sensitivity.csv", index=False)
    pd.set_option("display.width", 200)
    print(r.round(4).to_string(index=False))

    fig, ax = plt.subplots(1, 2, figsize=(12, 4))
    labs = [s[0] for s in SETTINGS]
    for cfg, col in zip(CFG, ["#d62728", "#843c39", "#bcbd22", "#1f77b4", "#ff7f0e"]):
        ax[0].plot(labs, [r[(r.setting == l) & (r.config == cfg)].loss.iloc[0] for l in labs], "-o", color=col, label=cfg)
        ax[1].plot(labs, [r[(r.setting == l) & (r.config == cfg)].pod_s.iloc[0] for l in labs], "-o", color=col, label=cfg)
    ax[0].set_ylabel("пулированная доля потерь")
    ax[1].set_ylabel("под-секунд на окно")
    for a in ax:
        a.tick_params(axis="x", rotation=20)
    ax[0].legend(fontsize=8)
    fig.suptitle("Чувствительность на реальной трассе (модели не переобучались)")
    fig.tight_layout()
    (out / "figures").mkdir(exist_ok=True)
    fig.savefig(out / "figures" / "trace_sensitivity.png")


if __name__ == "__main__":
    main()
