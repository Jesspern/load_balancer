"""Прогон всех конфигураций на ТЕСТОВЫХ окнах реальной трассы + кривые равной стоимости.

python experiments/trace_run.py calibrate   # порог fallback по валидационным окнам (по моделям)
python experiments/trace_run.py run         # основной прогон + развёртка параметров запаса
Результаты: results/trace/runs.csv, results/trace/pareto_runs.csv, results/trace/fb_calibration.json
"""
import json
import pickle
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from experiments import run_experiments as rx  # noqa: E402
from experiments.calibrate_fallback import err_series  # noqa: E402
from forecast import models as fm  # noqa: E402
from sim import trace  # noqa: E402
from sim.cluster import Config, ForecastTrack, SimParams, Simulator  # noqa: E402
from sim.metrics import summarize  # noqa: E402
from sim.profiles import make_arrivals  # noqa: E402

OUT = trace.OUT
MTAG = trace.MODEL_TAG
MODELS = ["naive", "ma", "linear", "peak", "msd", "ar", "lstm2", "gru2"]
MAIN = ["A1", "A2", "A3", "A4", "A5", "A6", "B3", "B6", "B7", "B8", "B9", "B10", "B11", "B12", "B13"]
HPA_TARGETS = [0.25, 0.35, 0.45, 0.55, 0.65, 0.75, 0.85]
RHOS = [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
FAMILIES = {"HPA+LC": None, "ПАБ naive": "naive", "ПАБ peak": "peak", "ПАБ mean+std": "msd",
            "ПАБ AR": "ar", "ПАБ LSTM v2": "lstm2", "ПАБ GRU v2": "gru2"}
CONFIGS = {c.name: c for c in rx.CONFIGS if c.name in MAIN}
CONFIGS["B12"] = Config("B12", "wlc", "pab", "peak")        # простая эвристика: пик за 30 с
CONFIGS["B13"] = Config("B13", "wlc", "pab", "msd")         # простая эвристика: среднее + std за 60 с
_m = {}


def load_models():
    if _m:
        return _m
    import torch
    torch.set_num_threads(1)
    _m.update({k: fm.make_model(k) for k in ("naive", "ma", "linear", "peak", "msd")})
    with open(ROOT / "results" / "cache" / f"{MTAG}_ar.pkl", "rb") as f:
        _m["ar"] = pickle.load(f)
    for k in ("lstm2", "gru2"):
        _m[k] = fm.NNForecaster2.load(ROOT / "forecast" / f"{MTAG}_{k}.pt")
    return _m


def fb_eps(model):
    p = OUT / "fb_calibration.json"
    return json.load(open(p))["fb_eps"].get(model, SimParams().fb_eps) if p.exists() else SimParams().fb_eps


def window_data(idx):
    lam, bursts = trace.trace_window(idx)
    arr = make_arrivals(lam, idx)
    bins = arr.reshape(-1, fm.DT).mean(axis=1)
    return lam, bursts, arr, bins


def run_window(idx, keep_ts=False, with_pareto=True):
    models = load_models()
    P0 = SimParams()
    lam, bursts, arr, bins = window_data(idx)
    tracks = {k: ForecastTrack(*models[k].track(bins)) for k in MODELS}
    n_peak = int(min(np.ceil(lam.max() / (P0.mu * 0.8)), P0.n_max))
    rows, par, ts = [], [], {}

    def go(c, P, fc):
        res = Simulator(c, P, arr, lam, forecast=fc, n_peak=n_peak).run()
        return res, summarize(res, lam, bursts, P0)

    for name in MAIN:
        c = CONFIGS[name]
        P = replace(P0, fb_eps=fb_eps(c.forecast)) if c.scaler == "pab" else P0
        res, s = go(c, P, tracks[c.forecast] if c.scaler == "pab" else None)
        s.update(window=idx, config=name)
        rows.append(s)
        if keep_ts:
            ts[name] = {k: res[k] for k in ("n_total", "n_ready", "arrived", "dropped", "late", "cap_ready", "fb")}
    if with_pareto:
        for t in HPA_TARGETS:
            _, s = go(Config("h", "lc", "hpa", hpa_overrides={"hpa_target": t}), P0, None)
            s.update(window=idx, family="HPA+LC", param=t)
            par.append(s)
        for fam, mname in FAMILIES.items():
            if mname is None:
                continue
            for rho in RHOS:
                P = replace(P0, fb_eps=fb_eps(mname), rho=rho)
                _, s = go(Config("b", "wlc", "pab", mname), P, tracks[mname])
                s.update(window=idx, family=fam, param=rho)
                par.append(s)
    return rows, par, ts, lam, bins, tracks


def _work(idx):
    r, p, _, _, _, _ = run_window(idx)
    return r, p


def calibrate():
    models = load_models()
    P = SimParams()
    sp = trace.split_indices()
    eps, detail = {}, {}
    wb = [window_data(i)[3] for i in sp["val"]]
    for k in MODELS:
        v = np.concatenate([err_series(b, models[k].track(b)[0], P.mu, P.fb_k) for b in wb])
        eps[k] = round(float(np.percentile(v, 99)), 3)
        detail[k] = {"p50": round(float(np.percentile(v, 50)), 3), "p99": eps[k]}
    OUT.mkdir(parents=True, exist_ok=True)
    json.dump({"fb_eps": eps, "detail": detail, "rule": "99th percentile on validation windows, per model"},
              open(OUT / "fb_calibration.json", "w"), indent=2)
    print(json.dumps(eps, indent=2))


def main():
    mode = sys.argv[1] if len(sys.argv) > 1 else "run"
    if mode == "calibrate":
        calibrate()
        return
    test = trace.split_indices()["test"]
    t0 = time.time()
    rows, par = [], []
    with ProcessPoolExecutor(max_workers=10) as ex:
        for i, (r, p) in enumerate(ex.map(_work, test)):
            rows += r
            par += p
            if i % 15 == 0:
                print(f"window {i + 1}/{len(test)} ({time.time() - t0:.0f}s)", flush=True)
    pd.DataFrame(rows).to_csv(OUT / "runs.csv", index=False)
    pd.DataFrame(par).to_csv(OUT / "pareto_runs.csv", index=False)
    print("saved", len(rows), "main runs,", len(par), "pareto runs")


if __name__ == "__main__":
    main()
