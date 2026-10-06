"""Серия экспериментов: все конфигурации x все профили x N повторов (общие seed и арrivals).

Результаты: results/runs.csv (метрики прогонов), results/forecast_metrics.csv (качество прогноза),
results/cache/ts_*.npz (временные ряды seed 0 для графиков).
"""
import sys
import time
from dataclasses import replace
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from forecast import models as fm  # noqa: E402
from sim.cluster import Config, ForecastTrack, SimParams, Simulator  # noqa: E402
from sim.metrics import summarize  # noqa: E402
from sim.profiles import PROFILES, make_arrivals  # noqa: E402

PROFILE_LIST = ["S1", "S2", "S3", "S4", "S5", "S6", "S5X"]
N_SEEDS = 10
FORECAST_MODELS = ["naive", "ma", "linear", "ar", "lstm", "gru", "lstm2", "gru2"]

CONFIGS = [
    Config("A1", "rr", "fixed"),
    Config("A2", "rr", "hpa"),
    Config("A3", "lc", "hpa"),
    Config("A4", "lc", "hpa_aggr"),
    Config("A5", "lc", "hpa", hpa_overrides={"hpa_target": 0.5}),
    Config("A6", "lc", "hpa", hpa_overrides={"hpa_target": 0.5, "hpa_period": 5, "hpa_down_window": 60,
                                              "metric_delay": 5, "metric_window": 5}),
    Config("B1", "wlc", "pab", "lstm"),
    Config("B2", "wlc", "pab", "gru"),
    Config("B3", "wlc", "pab", "ar"),
    Config("B4", "lc", "pab", "lstm"),
    Config("B5", "wlc", "pab", "lstm", fallback=False),
    Config("B6", "wlc", "pab", "naive"),
    Config("B7", "wlc", "pab", "linear"),
    Config("B8", "wlc", "pab", "lstm2"),
    Config("B9", "wlc", "pab", "gru2"),
    Config("B10", "lc", "pab", "lstm2"),
    Config("B11", "wlc", "pab", "lstm2", fallback=False),
]
TS_KEEP = ["n_total", "n_ready", "arrived", "dropped", "late", "util", "cap_ready", "fb"]

_models = {}


def _load_models():
    import pickle
    import torch
    torch.set_num_threads(1)
    for name in ("naive", "ma", "linear"):
        _models[name] = fm.make_model(name)
    with open(ROOT / "results" / "cache" / "ar.pkl", "rb") as f:
        _models["ar"] = pickle.load(f)
    for kind in ("lstm2", "gru2"):
        _models[kind] = fm.NNForecaster2.load(ROOT / "forecast" / f"{kind}.pt")
    for kind in ("lstm", "gru"):
        ck = torch.load(ROOT / "forecast" / f"{kind}.pt", weights_only=False)
        m = fm.NNForecaster(kind=kind, hidden=ck["hidden"])
        m.net = m._build()
        m.net.load_state_dict(ck["state"])
        m.net.eval()
        _models[kind] = m


def _fb_eps(model="lstm2"):
    """Порог fallback для модели прогноза из калибровки на валидационных сидах."""
    import json
    path = ROOT / "results" / "fb_calibration.json"
    if not path.exists():
        return SimParams().fb_eps
    return json.load(open(path))["fb_eps"].get(model, SimParams().fb_eps)


def forecast_quality(name, bins, lam_hat, p, bursts):
    K = len(bins)
    ks = range(fm.W, K - fm.H)
    err = np.array([lam_hat[k] - bins[k:k + fm.H] for k in ks])
    real = np.array([bins[k:k + fm.H] for k in ks])
    mae = np.abs(err).mean()
    rmse = np.sqrt((err ** 2).mean())
    smape = (2 * np.abs(err) / np.maximum(np.abs(lam_hat[list(ks)]) + np.abs(real), 1e-9)).mean() * 100
    under = (err < 0).mean()
    y = np.array([fm.burst_label(fm._pad_hist(bins, k), real[i]) for i, k in enumerate(ks)])
    pred = np.array([p[k] >= 0.5 for k in ks])
    tp = float(((pred == 1) & (y == 1)).sum())
    fp = float(((pred == 1) & (y == 0)).sum())
    fn = float(((pred == 0) & (y == 1)).sum())
    prec = tp / (tp + fp) if tp + fp else float("nan")
    rec = tp / (tp + fn) if tp + fn else float("nan")
    f1 = 2 * prec * rec / (prec + rec) if tp > 0 else (0.0 if tp + fp + fn > 0 else float("nan"))
    leads, det = [], 0
    for onset in bursts:
        b = onset // fm.DT
        base = float(np.median(bins[max(0, b - 6):b])) if b > 0 else 0.0
        # упреждающей считается тревога, поданная, пока нагрузка ещё на докризисном уровне
        window = [k for k in range(max(1, b - fm.H), min(b + 1, K))
                  if p[k] >= 0.5 and bins[k - 1] <= 1.2 * base]
        if window:
            det += 1
            leads.append((b - window[0]) * fm.DT)
    return dict(model=name, mae=mae, rmse=rmse, smape=smape, underest=under, prec=prec,
                rec=rec, f1=f1, burst_detect=det / len(bursts) if bursts else float("nan"),
                lead_s=float(np.mean(leads)) if leads else float("nan"))


def work(args):
    profile, seed = args
    if not _models:
        _load_models()
    P0 = SimParams()
    lam, bursts = PROFILES[profile](seed)
    arr = make_arrivals(lam, seed)
    bins = arr.reshape(-1, fm.DT).mean(axis=1)
    tracks, fq = {}, []
    for name in FORECAST_MODELS:
        lam_hat, p = _models[name].track(bins)
        tracks[name] = ForecastTrack(lam_hat, p)
        d = forecast_quality(name, bins, lam_hat, p, bursts)
        d.update(profile=profile, seed=seed)
        fq.append(d)
    n_peak = int(np.ceil(lam.max() / (P0.mu * 0.8)))
    rows, ts = [], {}
    for c in CONFIGS:
        fc = tracks[c.forecast] if c.scaler == "pab" else None
        P = replace(P0, fb_eps=_fb_eps(c.forecast)) if c.scaler == "pab" else P0
        res = Simulator(c, P, arr, lam, forecast=fc, n_peak=n_peak).run()
        s = summarize(res, lam, bursts, P0)
        s.update(profile=profile, seed=seed, config=c.name)
        rows.append(s)
        if seed == 0:
            ts[c.name] = {k: res[k] for k in TS_KEEP}
    if seed == 0:
        ts["lam"] = lam
        ts["bursts"] = np.array(bursts)
        ts["bins"] = bins
        ts["fc_lstm"] = tracks["lstm2"].lam_hat
        ts["p_lstm"] = tracks["lstm2"].p
    return profile, seed, rows, fq, ts


def main():
    cache = ROOT / "results" / "cache"
    cache.mkdir(parents=True, exist_ok=True)
    jobs = [(p, s) for p in PROFILE_LIST for s in range(N_SEEDS)]
    t0 = time.time()
    rows, fq = [], []
    with ProcessPoolExecutor(max_workers=10) as ex:
        for profile, seed, r, f, ts in ex.map(work, jobs):
            rows += r
            fq += f
            if ts:
                flat = {}
                for k, v in ts.items():
                    if isinstance(v, dict):
                        for kk, vv in v.items():
                            flat[f"{k}__{kk}"] = vv
                    else:
                        flat[k] = v
                np.savez_compressed(cache / f"ts_{profile}.npz", **flat)
            print(f"{profile} seed {seed} done ({time.time() - t0:.0f}s)", flush=True)
    pd.DataFrame(rows).to_csv(ROOT / "results" / "runs.csv", index=False)
    pd.DataFrame(fq).to_csv(ROOT / "results" / "forecast_metrics.csv", index=False)
    print("saved", len(rows), "runs")


if __name__ == "__main__":
    main()
