"""Первичный разбор трассы Azure Functions 2021: сколько нагрузки, какие приложения,
есть ли суточный цикл и всплески. Результаты: results/trace_overview.json и
results/figures/trace_overview.png.

Источник: Zhang et al., "Faster and Cheaper Serverless Computing on Harvested Resources",
SOSP 2021 (лицензия CC-BY). Издатель указывает, что временные метки изменены относительно
реального продакшена.
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
SRC = ROOT / "data" / "AzureFunctionsInvocationTraceForTwoWeeksJan2021.txt"
OUT = ROOT / "results"


def load():
    d = pd.read_csv(SRC)
    d["start"] = d["end_timestamp"] - d["duration"]
    return d


def per_window(starts, width, T):
    n = int(np.ceil(T / width))
    return np.bincount((starts // width).astype(int), minlength=n)[:n] / width


def main():
    d = load()
    T = float(d.end_timestamp.max())
    tot = per_window(d.start.values, 1, T)
    info = {"invocations": int(len(d)), "apps": int(d.app.nunique()), "days": round(T / 86400, 2),
            "mean_rps": float(tot.mean()), "max_rps_1s": float(tot.max())}
    for w in (5, 60, 300):
        r = per_window(d.start.values, w, T)
        info[f"max_rps_{w}s"] = float(r.max())
        info[f"p99_rps_{w}s"] = float(np.percentile(r, 99))
        info[f"cv_{w}s"] = float(r.std() / r.mean())
    top = d.groupby("app").size().sort_values(ascending=False)
    info["top_apps_share"] = {f"top{k}": round(float(top.iloc[:k].sum() / len(d)), 3) for k in (1, 3, 10)}
    info["top5_apps"] = [{"app": a[:8], "n": int(n), "mean_rps": round(n / T, 3)} for a, n in top.iloc[:5].items()]
    # суточная форма (по часам суток; "сутки" = 86400 с от начала трассы)
    hour = ((d.start.values % 86400) // 3600).astype(int)
    prof = np.bincount(hour, minlength=24) / (T / 86400) / 3600
    info["hourly_mean_rps"] = [round(float(x), 3) for x in prof]
    info["daily_peak_to_trough"] = float(prof.max() / max(prof.min(), 1e-9))
    day = (d.start.values // 86400).astype(int)
    info["invocations_per_day"] = [int(x) for x in np.bincount(day, minlength=14)]
    OUT.mkdir(exist_ok=True)
    with open(OUT / "trace_overview.json", "w") as f:
        json.dump(info, f, indent=2)
    print(json.dumps(info, indent=2))

    fig, ax = plt.subplots(3, 1, figsize=(11, 8))
    r60 = per_window(d.start.values, 60, T)
    ax[0].plot(np.arange(len(r60)) / 1440, r60, lw=0.5)
    ax[0].set_xlabel("день трассы")
    ax[0].set_ylabel("запросов/с (окно 60 с)")
    ax[0].set_title("Весь поток (все 119 приложений)")
    app1 = d[d.app == top.index[0]]
    r1 = per_window(app1.start.values, 60, T)
    ax[1].plot(np.arange(len(r1)) / 1440, r1, lw=0.5, color="tab:red")
    ax[1].set_xlabel("день трассы")
    ax[1].set_ylabel("запросов/с (окно 60 с)")
    ax[1].set_title(f"Крупнейшее приложение ({top.iloc[0] / len(d):.0%} всех вызовов)")
    ax[2].bar(range(24), prof)
    ax[2].set_xlabel("час суток (от начала трассы)")
    ax[2].set_ylabel("средний rps")
    ax[2].set_title("Средний суточный профиль")
    fig.tight_layout()
    (OUT / "figures").mkdir(exist_ok=True)
    fig.savefig(OUT / "figures" / "trace_overview.png", dpi=130)


if __name__ == "__main__":
    main()
