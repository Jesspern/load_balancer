"""Метрики качества одного прогона симуляции."""
import numpy as np

from .cluster import HIST_EDGES


def hist_percentile(hist, q):
    c = np.cumsum(hist)
    if c[-1] == 0:
        return float("nan")
    i = int(np.searchsorted(c, q * c[-1]))
    return float(HIST_EDGES[min(i, len(HIST_EDGES) - 1)])


def reaction_times(res, lam, bursts, mu, window=120):
    """Секунд от начала всплеска до момента, когда мощность готовых подов
    окончательно покрывает нагрузку с запасом 10 % (0 - если пода хватало заранее)."""
    out = []
    cap = res["cap_ready"]
    T = len(cap)
    for b in bursts:
        hi = min(b + window, T)
        deficit = np.where(cap[b:hi] < lam[b:hi] / 0.9)[0]
        out.append(float(deficit[-1] + 1) if len(deficit) else 0.0)
    return float(np.mean(out)) if out else float("nan")


def summarize(res, lam, bursts, params):
    arrived = res["arrived"].sum()
    bad = res["late"] + res["dropped"]
    viol = (bad / np.maximum(res["arrived"], 1)) > 0.05
    pod_sec = res["n_total"].sum()
    return {
        "arrived": float(arrived),
        "p50": hist_percentile(res["hist"], 0.50),
        "p95": hist_percentile(res["hist"], 0.95),
        "p99": hist_percentile(res["hist"], 0.99),
        "err_rate": float(res["dropped"].sum() / max(arrived, 1)),
        "late_rate": float(res["late"].sum() / max(arrived, 1)),
        "slo_viol_s": float(viol.sum()),
        "react_s": reaction_times(res, lam, bursts, params.mu),
        "jain": float(res["jain"].mean()),
        "cv": float(res["cv"].mean()),
        "pod_s": float(pod_sec),
        "scale_ev": float(res["scale_events"]),
        "fb_frac": float(res["fb"].mean()),
    }
