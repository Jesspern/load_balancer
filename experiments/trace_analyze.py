"""Анализ результатов на реальной трассе: сводные метрики, парные сравнения, равная стоимость,
качество прогноза, графики. Единица наблюдения - 30-минутное тестовое окно; доверительные интервалы
бутстрэпом по окнам (окна одной трассы коррелированы, это ограничение).

Все числа считаются здесь; результаты: results/trace/{summary,paired,equal_cost,forecast_quality}.csv,
tables.md, figures/*.png
"""
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from experiments import run_experiments as rx  # noqa: E402
from experiments import trace_run as tr  # noqa: E402
from sim import trace  # noqa: E402

OUT = trace.OUT
FIG = OUT / "figures"
FIG.mkdir(parents=True, exist_ok=True)
plt.rcParams.update({"font.size": 10, "axes.grid": True, "grid.alpha": 0.3, "figure.dpi": 130})
rng = np.random.default_rng(2024)
NB = 2000
CAP = 40 * 50.0  # предельная мощность кластера, запросов/с

runs = pd.read_csv(OUT / "runs.csv")
par = pd.read_csv(OUT / "pareto_runs.csv")
windows = sorted(runs.window.unique())
W = len(windows)
peak = {w: float(trace.trace_window(w)[0].max()) for w in windows}
feasible = [w for w in windows if peak[w] <= 0.8 * CAP]
SUBSETS = {"все тестовые окна": windows, f"окна с пиком <= {int(0.8 * CAP)} запр/с": feasible}
NAMES = {"A1": "A1 RR+фикс.", "A2": "A2 RR+HPA", "A3": "A3 LC+HPA", "A4": "A4 агресс. HPA", "A5": "A5 HPA 50 %",
         "A6": "A6 быстрый HPA", "B3": "B3 AR", "B6": "B6 naive", "B7": "B7 линейный", "B8": "B8 LSTM v2",
         "B12": "B12 пик за 30 с", "B13": "B13 среднее+std", "B9": "B9 GRU v2", "B10": "B10 LSTM, только масштабирование", "B11": "B11 LSTM без fallback"}


def tab(df, cfg, col):
    """Вектор значений по окнам (в порядке windows) для конфигурации."""
    d = df[df.config == cfg].set_index("window")
    return d.loc[windows, col].values


M = {c: {k: tab(runs, c, k) for k in ("err_rate", "late_rate", "arrived", "slo_viol_s", "pod_s", "react_s")}
     for c in runs.config.unique()}


def pooled(cfg, idx, what="err_rate"):
    a, e = M[cfg]["arrived"][idx], M[cfg][what][idx]
    return float((a * e).sum() / a.sum())


def boot(fn, idx_all, n=NB):
    vals = []
    for _ in range(n):
        s = rng.choice(len(idx_all), size=len(idx_all), replace=True)
        vals.append(fn(np.array(idx_all)[s]))
    return np.percentile(vals, [2.5, 97.5])


# ---- сводка
rows = []
pos = {w: i for i, w in enumerate(windows)}
for sub, wl in SUBSETS.items():
    ix = np.array([pos[w] for w in wl])
    for c in M:
        r = {"subset": sub, "n_windows": len(ix), "config": c}
        r["loss"] = pooled(c, ix)
        r["loss_lo"], r["loss_hi"] = boot(lambda i, c=c: pooled(c, i), ix)
        r["late"] = pooled(c, ix, "late_rate")
        r["viol_s_mean"] = float(M[c]["slo_viol_s"][ix].mean())
        r["viol_lo"], r["viol_hi"] = boot(lambda i, c=c: M[c]["slo_viol_s"][i].mean(), ix)
        r["pod_s_mean"] = float(M[c]["pod_s"][ix].mean())
        r["pod_lo"], r["pod_hi"] = boot(lambda i, c=c: M[c]["pod_s"][i].mean(), ix)
        rr = M[c]["react_s"][ix]
        r["react_s"] = float(np.nanmean(rr)) if np.isfinite(rr).any() else np.nan
        rows.append(r)
summ = pd.DataFrame(rows)
summ.to_csv(OUT / "summary.csv", index=False)

# ---- парные сравнения (бутстрэп по окнам): разность пулированных потерь и средних под-секунд
pairs = [("B8", "A3"), ("B8", "A6"), ("B8", "B6"), ("B8", "B3"), ("B8", "B9"), ("B8", "B11"), ("B6", "A3"),
         ("B8", "B12"), ("B8", "B13"), ("B12", "A3"), ("B13", "A3")]
prow = []
for sub, wl in SUBSETS.items():
    ix = np.array([pos[w] for w in wl])
    for a, b in pairs:
        for metric, fn in (("loss", lambda i, a=a, b=b: pooled(a, i) - pooled(b, i)),
                           ("pod_s", lambda i, a=a, b=b: M[a]["pod_s"][i].mean() - M[b]["pod_s"][i].mean()),
                           ("viol_s", lambda i, a=a, b=b: M[a]["slo_viol_s"][i].mean() - M[b]["slo_viol_s"][i].mean())):
            lo, hi = boot(fn, ix)
            prow.append({"subset": sub, "A": a, "B": b, "metric": metric, "diff_A_minus_B": fn(ix),
                         "ci_lo": lo, "ci_hi": hi, "significant_95": bool(lo > 0 or hi < 0)})
paired = pd.DataFrame(prow)
paired.to_csv(OUT / "paired.csv", index=False)

# ---- равная стоимость: потери семейств при бюджете = стоимость A3 (и B8)
FAM = list(tr.FAMILIES)
pos_p = {w: i for i, w in enumerate(windows)}
P = {}  # P[fam][param] -> (err_rate per window, arrived, pod_s)
for fam in FAM:
    for pr_, g in par[par.family == fam].groupby("param"):
        g = g.set_index("window").loc[windows]
        P.setdefault(fam, {})[pr_] = (g.err_rate.values, g.arrived.values, g.pod_s.values)


def curve(fam, idx):
    xs, ys = [], []
    for pr_, (e, a, p) in P[fam].items():
        xs.append(p[idx].mean())
        ys.append((e[idx] * a[idx]).sum() / a[idx].sum())
    o = np.argsort(xs)
    return np.array(xs)[o], np.minimum.accumulate(np.array(ys)[o])


def err_at(fam, idx, budget):
    x, y = curve(fam, idx)
    return float(np.interp(budget, x, y)) if x.min() <= budget <= x.max() else np.nan


erow = []
for sub, wl in SUBSETS.items():
    ix = np.array([pos_p[w] for w in wl])
    for ref in ("A3", "A6", "B8"):
        budget_fn = lambda i, ref=ref: M[ref]["pod_s"][i].mean()
        vals = {f: [] for f in FAM}
        for _ in range(500):
            s = ix[rng.choice(len(ix), size=len(ix), replace=True)]
            b = budget_fn(s)
            for f in FAM:
                vals[f].append(err_at(f, s, b))
        for f in FAM:
            v = np.array(vals[f])
            ok = np.isfinite(v)
            erow.append({"subset": sub, "budget_of": ref, "family": f,
                         "loss_at_budget": err_at(f, ix, budget_fn(ix)),
                         "ci_lo": np.percentile(v[ok], 2.5) if ok.sum() > 50 else np.nan,
                         "ci_hi": np.percentile(v[ok], 97.5) if ok.sum() > 50 else np.nan,
                         "boot_in_range": int(ok.sum())})
        for a, b in (("ПАБ LSTM v2", "HPA+LC"), ("ПАБ LSTM v2", "ПАБ naive"), ("ПАБ LSTM v2", "ПАБ AR"),
                     ("ПАБ naive", "HPA+LC"), ("ПАБ LSTM v2", "ПАБ peak"), ("ПАБ LSTM v2", "ПАБ mean+std"),
                     ("ПАБ peak", "HPA+LC"), ("ПАБ mean+std", "HPA+LC")):
            d = np.array(vals[a]) - np.array(vals[b])
            ok = np.isfinite(d)
            if ok.sum() > 50:
                erow.append({"subset": sub, "budget_of": ref, "family": f"{a} - {b}",
                             "loss_at_budget": err_at(a, ix, budget_fn(ix)) - err_at(b, ix, budget_fn(ix)),
                             "ci_lo": np.percentile(d[ok], 2.5), "ci_hi": np.percentile(d[ok], 97.5),
                             "boot_in_range": int(ok.sum())})
eq = pd.DataFrame(erow)
eq.to_csv(OUT / "equal_cost.csv", index=False)

# ---- качество прогноза на тестовых окнах
models = tr.load_models()
fq = []
for w in windows:
    lam, bursts, arr, bins = tr.window_data(w)
    for k in tr.MODELS:
        lh, p = models[k].track(bins)
        d = rx.forecast_quality(k, bins, lh, p, bursts)
        d["window"] = w
        fq.append(d)
fqd = pd.DataFrame(fq).groupby("model")[["mae", "rmse", "smape", "underest"]].mean().round(3).reset_index()
fqd.to_csv(OUT / "forecast_quality.csv", index=False)

# ---- устойчивость по трети тестового периода и проверка H2 на спокойных окнах
thirds = np.array_split(np.arange(W), 3)
trows = []
for k, ix in enumerate(thirds, 1):
    for a, b in (("B8", "A3"), ("B8", "B6"), ("B8", "B12")):
        fn = lambda i, a=a, b=b: pooled(a, i) - pooled(b, i)
        fp = lambda i, a=a, b=b: M[a]["pod_s"][i].mean() - M[b]["pod_s"][i].mean()
        lo, hi = boot(fn, ix)
        plo, phi = boot(fp, ix)
        trows.append({"third": k, "windows": f"{windows[ix[0]]}-{windows[ix[-1]]}", "A": a, "B": b,
                      "loss_diff": fn(ix), "loss_lo": lo, "loss_hi": hi, "pod_diff": fp(ix),
                      "pod_lo": plo, "pod_hi": phi})
pd.DataFrame(trows).to_csv(OUT / "thirds.csv", index=False)

quiet = [w for w in windows if peak[w] <= 150]
qix = np.array([pos[w] for w in quiet])
p95 = {c: runs[runs.config == c].set_index("window").loc[windows, "p95"].values for c in ("A3", "B8")}
H2 = {"n_quiet_windows": len(quiet), "peak_threshold_rps": 150,
      "pod_s_ratio_B8_over_A3": float(M["B8"]["pod_s"][qix].mean() / M["A3"]["pod_s"][qix].mean()) if len(qix) else None,
      "p95_B8": float(np.mean(p95["B8"][qix])) if len(qix) else None,
      "p95_A3": float(np.mean(p95["A3"][qix])) if len(qix) else None,
      "loss_B8": pooled("B8", qix) if len(qix) else None, "loss_A3": pooled("A3", qix) if len(qix) else None}
json.dump(H2, open(OUT / "h2_quiet_windows.json", "w"), indent=2)

# ---- таблицы
def f3(x):
    return "-" if not np.isfinite(x) else f"{x:.3f}"


md = ["# Реальная трасса Azure Functions 2021: результаты (тестовые окна)\n",
      "Сгенерировано experiments/trace_analyze.py. Единица наблюдения - 30-минутное окно; "
      "доверительные интервалы (95 %) бутстрэпом по окнам. Окна одной трассы коррелированы.\n"]
for sub in SUBSETS:
    md.append(f"\n## {sub}\n")
    md.append("| Конфиг | Потери (пулир.) [95 % ДИ] | Нарушение SLO, с/окно | Под-секунд/окно | Реакция, с |")
    md.append("|---|---|---|---|---|")
    for c in [x for x in tr.MAIN if x in M]:
        r = summ[(summ.subset == sub) & (summ.config == c)].iloc[0]
        md.append(f"| {NAMES[c]} | {r.loss:.4f} [{r.loss_lo:.4f}; {r.loss_hi:.4f}] | "
                  f"{r.viol_s_mean:.0f} [{r.viol_lo:.0f}; {r.viol_hi:.0f}] | "
                  f"{r.pod_s_mean:.0f} [{r.pod_lo:.0f}; {r.pod_hi:.0f}] | {f3(r.react_s) if np.isnan(r.react_s) else f'{r.react_s:.0f}'} |")
md.append("\n## Парные сравнения (A - B, бутстрэп по окнам)\n")
md.append("| Подмножество | A | B | метрика | A - B | 95 % ДИ | значимо |")
md.append("|---|---|---|---|---|---|---|")
for _, r in paired.iterrows():
    md.append(f"| {r.subset} | {r.A} | {r.B} | {r.metric} | {r.diff_A_minus_B:.4g} | "
              f"[{r.ci_lo:.4g}; {r.ci_hi:.4g}] | {'да' if r.significant_95 else 'нет'} |")
md.append("\n## Потери при равной стоимости (бюджет = под-секунды A3 или B8)\n")
md.append("| Подмножество | бюджет | семейство | потери | 95 % ДИ |")
md.append("|---|---|---|---|---|")
for _, r in eq.iterrows():
    md.append(f"| {r.subset} | {r.budget_of} | {r.family} | {f3(r.loss_at_budget)} | "
              f"[{f3(r.ci_lo)}; {f3(r.ci_hi)}] |")
md.append("\n## Качество прогноза (среднее по тестовым окнам)\n")
md.append("| модель | MAE | RMSE | sMAPE % | доля недооценок |")
md.append("|---|---|---|---|---|")
for _, r in fqd.iterrows():
    md.append(f"| {r.model} | {r.mae} | {r.rmse} | {r.smape} | {r.underest} |")
(OUT / "tables.md").write_text("\n".join(md), encoding="utf8")

# ---- графики
ALLC = {"A3": "#d62728", "A6": "#843c39", "B3": "#2ca02c", "B6": "#bcbd22", "B8": "#1f77b4", "B9": "#17becf",
        "B11": "#636363"}
fig, axes = plt.subplots(1, 2, figsize=(13, 4.3))
for ax, (sub, wl) in zip(axes, SUBSETS.items()):
    ix = np.array([pos_p[w] for w in wl])
    for fam, col in zip(FAM, ["#d62728", "#bcbd22", "#ff7f0e", "#9467bd", "#2ca02c", "#1f77b4", "#17becf"]):
        x, y = curve(fam, ix)
        ax.plot(x, y, "-o", ms=4, color=col, label=fam)
    ax.axvline(M["A3"]["pod_s"][ix].mean(), color="gray", ls=":", lw=1)
    ax.set_title(sub, fontsize=10)
    ax.set_xlabel("под-секунд на окно (стоимость)")
    ax.set_ylabel("пулированная доля потерь")
axes[0].legend(fontsize=8)
fig.suptitle("Реальная трасса: потери при разной стоимости (пунктир - стоимость A3)")
fig.tight_layout()
fig.savefig(FIG / "trace_pareto.png")
plt.close(fig)

fig, ax = plt.subplots(figsize=(9, 4))
order = ["A3", "A6", "B3", "B6", "B8", "B9", "B11"]
ix = np.array([pos[w] for w in windows])
v = [summ[(summ.subset == "все тестовые окна") & (summ.config == c)].iloc[0] for c in order]
ax.bar(order, [r.loss for r in v], yerr=[[r.loss - r.loss_lo for r in v], [r.loss_hi - r.loss for r in v]],
       color=[ALLC[c] for c in order], capsize=3)
for i, r in enumerate(v):
    ax.text(i, r.loss_hi, f"{r.pod_s_mean:.0f} п-с", ha="center", va="bottom", fontsize=8)
ax.set_ylabel("пулированная доля потерь")
ax.set_title("Реальная трасса: потери (подпись - под-секунд на окно)")
fig.tight_layout()
fig.savefig(FIG / "trace_bars.png")
plt.close(fig)

# окно для иллюстрации: наибольшие потери A3 среди допустимых окон
loss_a3 = {w: M["A3"]["err_rate"][pos[w]] * M["A3"]["arrived"][pos[w]] for w in feasible}
wsel = max(loss_a3, key=loss_a3.get)
_, _, ts, lam, bins, tracks = tr.run_window(wsel, keep_ts=True, with_pareto=False)
t = np.arange(len(lam))
fig, ax = plt.subplots(3, 1, figsize=(10, 8), sharex=True)
ax[0].plot(t, lam, "k", lw=0.9, label="нагрузка λ(t)")
for c in ("A3", "B8"):
    ax[0].plot(t, ts[c]["cap_ready"], color=ALLC[c], label=f"{c}: мощность готовых подов")
ax[0].set_ylabel("запросов/с")
ax[0].legend(fontsize=8)
for c in ("A3", "B6", "B8"):
    ax[1].plot(t, ts[c]["n_total"], color=ALLC[c], label=c)
ax[1].set_ylabel("число подов")
ax[1].legend(fontsize=8)
for c in ("A3", "B8"):
    bad = (ts[c]["late"] + ts[c]["dropped"]) / np.maximum(ts[c]["arrived"], 1)
    ax[2].plot(t, bad, color=ALLC[c], lw=1, label=c)
ax[2].set_ylabel("доля плохих запросов")
ax[2].set_xlabel("время в окне, с")
ax[2].legend(fontsize=8)
fig.suptitle(f"Реальная трасса, тестовое окно {wsel}: один прогон")
fig.tight_layout()
fig.savefig(FIG / "trace_window.png")
plt.close(fig)

json.dump({"example_window": int(wsel), "n_test_windows": W, "n_feasible_windows": len(feasible),
           "peak_cap_rps": CAP}, open(OUT / "analysis_info.json", "w"), indent=2)
print("done; example window", wsel)
print(summ[summ.subset == "все тестовые окна"][["config", "loss", "loss_lo", "loss_hi", "viol_s_mean", "pod_s_mean"]]
      .round(4).to_string(index=False))
