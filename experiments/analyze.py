"""Анализ результатов: таблицы (медиана + 95 % бутстрэп-ДИ), критерий Манна-Уитни с поправкой
Холма, проверка гипотез H1-H3, графики. Все числа в отчёте берутся отсюда."""
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy.stats import mannwhitneyu  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from sim.profiles import PROFILE_TITLES  # noqa: E402

RES = ROOT / "results"
FIG = RES / "figures"
FIG.mkdir(parents=True, exist_ok=True)
plt.rcParams.update({"font.size": 10, "axes.grid": True, "grid.alpha": 0.3, "figure.dpi": 130})

runs = pd.read_csv(RES / "runs.csv")
fq = pd.read_csv(RES / "forecast_metrics.csv")
CFG_ORDER = ["A1", "A2", "A3", "A4", "A5", "A6", "B1", "B2", "B3", "B4", "B5", "B6", "B7",
             "B8", "B9", "B10", "B11"]
PROFS = ["S1", "S2", "S3", "S4", "S5", "S6", "S5X"]
METRICS = ["p95", "p99", "err_rate", "slo_viol_s", "react_s", "pod_s", "scale_ev", "jain", "cv"]
rng = np.random.default_rng(12345)


def boot_ci(x, n=2000):
    x = np.asarray(x, dtype=float)
    x = x[~np.isnan(x)]
    if len(x) == 0:
        return (np.nan, np.nan)
    med = np.median(rng.choice(x, size=(n, len(x)), replace=True), axis=1)
    return (np.percentile(med, 2.5), np.percentile(med, 97.5))


def holm(pvals):
    p = np.asarray(pvals, dtype=float)
    order = np.argsort(p)
    m = len(p)
    adj = np.empty(m)
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, (m - rank) * p[i])
        adj[i] = min(1.0, running)
    return adj


# ---- сводная таблица
rows = []
for prof in PROFS:
    for cfg in CFG_ORDER:
        d = runs[(runs.profile == prof) & (runs.config == cfg)]
        r = {"profile": prof, "config": cfg, "n": len(d)}
        for m in METRICS:
            lo, hi = boot_ci(d[m])
            r[m] = d[m].median()
            r[m + "_lo"], r[m + "_hi"] = lo, hi
        r["fb_frac"] = d["fb_frac"].median()
        rows.append(r)
summ = pd.DataFrame(rows)
summ.to_csv(RES / "summary.csv", index=False)

# ---- Манн-Уитни: каждая конфигурация против эталона (A3, A6, B6), поправка Холма по семейству
def compare(ref):
    st = []
    for m in ["err_rate", "slo_viol_s", "pod_s", "p95"]:
        batch = []
        for prof in PROFS:
            a = runs[(runs.profile == prof) & (runs.config == ref)].sort_values("seed")[m].values
            for cfg in CFG_ORDER:
                if cfg == ref:
                    continue
                b = runs[(runs.profile == prof) & (runs.config == cfg)].sort_values("seed")[m].values
                p = 1.0 if np.allclose(a, b) else mannwhitneyu(b, a, alternative="two-sided").pvalue
                batch.append({"ref": ref, "metric": m, "profile": prof, "config": cfg,
                              "median_cfg": np.median(b), "median_ref": np.median(a), "p_raw": p})
        adj = holm([x["p_raw"] for x in batch])
        for x, pa in zip(batch, adj):
            x["p_holm"] = pa
            x["signif"] = pa < 0.05
        st += batch
    return st


stats = pd.DataFrame(compare("A3") + compare("A6") + compare("B6"))
stats.to_csv(RES / "stats.csv", index=False)


def sig(metric, prof, cfg, ref="A3"):
    r = stats[(stats.ref == ref) & (stats.metric == metric) & (stats.profile == prof) & (stats.config == cfg)].iloc[0]
    return r


def med(prof, cfg, m):
    return float(summ[(summ.profile == prof) & (summ.config == cfg)][m].iloc[0])


# ---- гипотезы (предлагаемый метод: B8 = взвешенный LC + упреждающее масштабирование на LSTM v2)
PROP = "B8"
H = {}
for ref in ("A3", "A6", "B6"):
    for prof in ("S3", "S5", "S6"):
        for m in ("err_rate", "slo_viol_s"):
            r = sig(m, prof, PROP, ref)
            H[f"H1 {PROP} vs {ref} | {prof} {m}"] = dict(
                ref_median=r.median_ref, B8_median=r.median_cfg, p_holm=r.p_holm,
                better=bool(r.median_cfg < r.median_ref), significant=bool(r.p_holm < 0.05))
p95_ratio = med("S1", PROP, "p95") / med("S1", "A3", "p95") - 1
pod_ratio = med("S1", PROP, "pod_s") / med("S1", "A3", "pod_s") - 1
H["H2 S1 (B8 vs A3)"] = dict(p95_delta=p95_ratio, pod_s_delta=pod_ratio,
                              ok=bool(p95_ratio <= 0.05 and pod_ratio <= 0.15))
e_b, e_a3, e_nf = med("S5X", PROP, "err_rate"), med("S5X", "A3", "err_rate"), med("S5X", "B11", "err_rate")
v_b, v_a3, v_nf = med("S5X", PROP, "slo_viol_s"), med("S5X", "A3", "slo_viol_s"), med("S5X", "B11", "slo_viol_s")
H["H3 S5X"] = dict(err_B8=e_b, err_A3=e_a3, err_B11_no_fallback=e_nf, viol_B8=v_b, viol_A3=v_a3,
                   viol_B11_no_fallback=v_nf, fb_frac_B8=med("S5X", PROP, "fb_frac"),
                   ok_vs_A3=bool(e_b <= 1.10 * e_a3 and v_b <= 1.10 * v_a3),
                   fallback_better_viol=bool(v_b < v_nf), fallback_better_err=bool(e_b < e_nf))
with open(RES / "hypotheses.json", "w", encoding="utf8") as f:
    json.dump(H, f, indent=2, ensure_ascii=False, default=float)

# ---- markdown-таблицы
fmt = {"p95": "{:.2f}", "p99": "{:.2f}", "err_rate": "{:.3f}", "slo_viol_s": "{:.0f}", "react_s": "{:.0f}",
       "pod_s": "{:.0f}", "scale_ev": "{:.0f}", "jain": "{:.3f}", "cv": "{:.3f}"}
names = {"p95": "p95, с", "p99": "p99, с", "err_rate": "потери", "slo_viol_s": "нарушение SLO, с",
         "react_s": "реакция, с", "pod_s": "под-сек", "scale_ev": "событий масштаб.",
         "jain": "Джейн J", "cv": "CV загрузки"}
md = ["# Результаты экспериментов (медиана по 10 повторам [95 % бутстрэп-ДИ])\n",
      "Сгенерировано experiments/analyze.py. Не править вручную.\n"]
for prof in PROFS:
    md.append(f"\n## {PROFILE_TITLES[prof]}\n")
    cols = ["p95", "err_rate", "slo_viol_s", "react_s", "pod_s", "scale_ev", "jain"]
    md.append("| Конфиг | " + " | ".join(names[c] for c in cols) + " | fallback, доля времени |")
    md.append("|---|" + "---|" * (len(cols) + 1))
    for cfg in CFG_ORDER:
        r = summ[(summ.profile == prof) & (summ.config == cfg)].iloc[0]
        cells = []
        for c in cols:
            if np.isnan(r[c]):
                cells.append("-")
            else:
                cells.append(fmt[c].format(r[c]) + f" [{fmt[c].format(r[c + '_lo'])}; {fmt[c].format(r[c + '_hi'])}]")
        md.append(f"| {cfg} | " + " | ".join(cells) + f" | {r['fb_frac']:.2f} |")
md.append("\n## Качество прогноза (медиана по сидам, тестовые трассы)\n")
md.append("Внимание: burst_detect и lead_s неинформативны на S5/S5X (всплески перекрываются, и тревога "
          "соседнего всплеска засчитывается как упреждающая; даже naive получает ненулевые значения). "
          "В выводах не используются. precision/recall/F1 считаются по метке всплеска, заданной тем "
          "же правилом, что и p у простых моделей, поэтому для naive/ma/linear они не сравнимы с LSTM/GRU "
          "напрямую.\n")
fqm = fq.groupby(["profile", "model"])[["mae", "rmse", "smape", "underest", "prec", "rec", "f1",
                                        "burst_detect", "lead_s"]].median().round(2).reset_index()
md.append("| " + " | ".join(fqm.columns) + " |")
md.append("|" + "---|" * len(fqm.columns))
for _, r in fqm.iterrows():
    md.append("| " + " | ".join("-" if (isinstance(v, float) and np.isnan(v)) else str(v) for v in r) + " |")
md.append("\n## Проверка гипотез\n```json\n" + json.dumps(H, indent=2, ensure_ascii=False, default=float) + "\n```")
(RES / "tables.md").write_text("\n".join(md), encoding="utf8")

# ---- графики
COL = {"A5": "#e6550d", "A6": "#843c39", "B8": "#08519c", "B9": "#3182bd", "B10": "#756bb1", "B11": "#636363",
       "A1": "#7f7f7f", "A2": "#c49c94", "A3": "#d62728", "A4": "#ff7f0e", "B1": "#1f77b4",
       "B2": "#17becf", "B3": "#2ca02c", "B4": "#9467bd", "B5": "#8c564b", "B6": "#bcbd22", "B7": "#e377c2"}


def load_ts(prof):
    return np.load(RES / "cache" / f"ts_{prof}.npz", allow_pickle=True)


# 1: временные ряды (S3 и S5)
for prof, (t0, t1) in {"S3": (300, 700), "S5": (0, 700), "S6": (0, 900)}.items():
    ts = load_ts(prof)
    fig, ax = plt.subplots(3, 1, figsize=(9, 8), sharex=True)
    t = np.arange(len(ts["lam"]))
    sl = slice(t0, t1)
    ax[0].plot(t[sl], ts["lam"][sl], "k", lw=1, label="нагрузка λ(t)")
    for cfg in ("A3", "B8"):
        ax[0].plot(t[sl], ts[f"{cfg}__cap_ready"][sl], color=COL[cfg], lw=1.4,
                   label=f"{cfg}: мощность готовых подов")
    ax[0].set_ylabel("запросов/с")
    ax[0].legend(fontsize=8)
    for cfg in ("A3", "B6", "B8"):
        ax[1].plot(t[sl], ts[f"{cfg}__n_total"][sl], color=COL[cfg], label=f"{cfg}")
    ax[1].set_ylabel("число подов (с запускаемыми)")
    ax[1].legend(fontsize=8)
    for cfg in ("A3", "B8"):
        bad = (ts[f"{cfg}__late"] + ts[f"{cfg}__dropped"]) / np.maximum(ts[f"{cfg}__arrived"], 1)
        ax[2].plot(t[sl], bad[sl], color=COL[cfg], lw=1, label=f"{cfg}")
    ax[2].set_ylabel("доля плохих запросов")
    ax[2].set_xlabel("время, с")
    ax[2].legend(fontsize=8)
    for b in ts["bursts"]:
        if t0 <= b <= t1:
            for a in ax:
                a.axvline(b, color="gray", ls=":", lw=0.8)
    fig.suptitle(f"{PROFILE_TITLES[prof]}: один прогон (seed 0), пунктир - начало всплеска")
    fig.tight_layout()
    fig.savefig(FIG / f"timeseries_{prof}.png")
    plt.close(fig)

# 2: потери и под-секунды по профилям
fig, ax = plt.subplots(1, 2, figsize=(12, 4.2))
w = 0.09
show = ["A1", "A3", "A6", "B6", "B1", "B8", "B11"]
for j, cfg in enumerate(show):
    for a, m, lab in ((ax[0], "err_rate", "доля потерянных запросов"), (ax[1], "pod_s", "под-секунды")):
        vals = [med(p, cfg, m) for p in PROFS]
        lo = [med(p, cfg, m) - float(summ[(summ.profile == p) & (summ.config == cfg)][m + "_lo"].iloc[0]) for p in PROFS]
        hi = [float(summ[(summ.profile == p) & (summ.config == cfg)][m + "_hi"].iloc[0]) - med(p, cfg, m) for p in PROFS]
        a.bar(np.arange(len(PROFS)) + (j - len(show) / 2) * w, vals, w, yerr=[lo, hi], color=COL[cfg],
              label=cfg, capsize=1.5)
        a.set_ylabel(lab)
        a.set_xticks(range(len(PROFS)))
        a.set_xticklabels(PROFS)
ax[0].legend(ncol=2, fontsize=8)
fig.suptitle("Медиана по 10 повторам и 95 % бутстрэп-ДИ")
fig.tight_layout()
fig.savefig(FIG / "bars_err_pods.png")
plt.close(fig)

# 3: компромисс потери - ресурсы (профили со всплесками)
fig, ax = plt.subplots(1, 4, figsize=(17, 4))
for a, prof in zip(ax, ["S3", "S5", "S6", "S5X"]):
    for cfg in CFG_ORDER:
        a.scatter(med(prof, cfg, "pod_s"), med(prof, cfg, "err_rate"), color=COL[cfg], s=45)
        a.annotate(cfg, (med(prof, cfg, "pod_s"), med(prof, cfg, "err_rate")), fontsize=8,
                   xytext=(3, 3), textcoords="offset points")
    a.set_title(PROFILE_TITLES[prof])
    a.set_xlabel("под-секунды (ресурсы)")
    a.set_ylabel("доля потерянных запросов")
fig.tight_layout()
fig.savefig(FIG / "tradeoff.png")
plt.close(fig)

# 4: прогноз LSTM на S5 (seed 0)
ts = load_ts("S5")
bins = ts["bins"]
fc1 = ts["fc_lstm"][:, 0]
fch = np.array([ts["fc_lstm"][k, 11] if k + 11 < len(bins) else np.nan for k in range(len(bins))])
tb = np.arange(len(bins)) * 5
fig, ax = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
ax[0].plot(tb, bins, "k", lw=0.9, label="факт (бины 5 с)")
ax[0].plot(tb, fc1, color=COL["B1"], lw=1, label="LSTM, шаг +1 (5 с)")
ax[0].plot(tb + 55, fch, color=COL["A3"], lw=1, label="LSTM, шаг +12 (60 с), сдвинут на 55 с")
ax[0].set_ylabel("запросов/с")
ax[0].legend(fontsize=8)
ax[1].plot(tb, ts["p_lstm"], color=COL["B3"])
ax[1].axhline(0.5, color="gray", ls="--", lw=0.8)
for b in ts["bursts"]:
    ax[1].axvline(b, color="gray", ls=":", lw=0.8)
ax[1].set_ylabel("вероятность всплеска p")
ax[1].set_xlabel("время, с")
fig.suptitle("Прогноз LSTM на профиле S5 (seed 0)")
fig.tight_layout()
fig.savefig(FIG / "forecast_S5.png")
plt.close(fig)

# 5: ошибки прогноза по моделям
fig, ax = plt.subplots(1, 2, figsize=(11, 4))
models = ["naive", "ma", "linear", "ar", "lstm", "gru", "lstm2", "gru2"]
for a, m, lab in ((ax[0], "mae", "MAE, запросов/с"), (ax[1], "smape", "sMAPE, %")):
    for j, mod in enumerate(models):
        v = [fq[(fq.profile == p) & (fq.model == mod)][m].median() for p in PROFS]
        a.bar(np.arange(len(PROFS)) + (j - 3.5) * 0.11, v, 0.11, label=mod)
    a.set_xticks(range(len(PROFS)))
    a.set_xticklabels(PROFS)
    a.set_ylabel(lab)
ax[0].legend(fontsize=8, ncol=2)
fig.tight_layout()
fig.savefig(FIG / "forecast_quality.png")
plt.close(fig)

# 6: кривые обучения
fig, ax = plt.subplots(figsize=(6.5, 4))
for kind in ("lstm", "gru"):
    h = np.loadtxt(RES / f"train_history_{kind}.csv", delimiter=",", skiprows=1)
    ax.plot(h[:, 0], h[:, 2], "--", label=f"{kind} v1 val")
for kind in ("lstm2", "gru2"):
    h = np.loadtxt(RES / f"train_history_{kind}.csv", delimiter=",", skiprows=1)
    ep = np.unique(h[:, 1])
    ax.plot(ep, [h[h[:, 1] == e, 3].mean() for e in ep], label=f"{kind} (среднее по членам) val")
ax.set_xlabel("эпоха")
ax.set_ylabel("потеря на валидации")
ax.legend(fontsize=8)
fig.tight_layout()
fig.savefig(FIG / "training_loss.png")
plt.close(fig)

print(json.dumps(H, indent=2, ensure_ascii=False, default=float))
print("figures:", sorted(p.name for p in FIG.glob("*.png")))
