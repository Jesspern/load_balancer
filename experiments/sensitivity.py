"""Чувствительность вывода к времени запуска пода T_start (от этого параметра зависит выигрыш
упреждающего масштабирования). Профили S3 и S5, конфигурации A3, B1, B6, 10 сидов."""
import sys
from dataclasses import replace
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from experiments import run_experiments as rx  # noqa: E402
from forecast import models as fm  # noqa: E402
from sim.cluster import ForecastTrack, SimParams, Simulator  # noqa: E402
from sim.metrics import summarize  # noqa: E402
from sim.profiles import PROFILES, make_arrivals  # noqa: E402

T_STARTS = [10, 20, 30, 60]
CFGS = {c.name: c for c in rx.CONFIGS if c.name in ("A3", "A5", "A6", "B6", "B8")}


def work(args):
    prof, seed, ts_ = args
    if not rx._models:
        rx._load_models()
    P0 = SimParams(t_start=float(ts_))
    lam, bursts = PROFILES[prof](seed)
    arr = make_arrivals(lam, seed)
    bins = arr.reshape(-1, fm.DT).mean(axis=1)
    rows = []
    for name, c in CFGS.items():
        fc = None
        if c.scaler == "pab":
            lh, p = rx._models[c.forecast].track(bins)
            fc = ForecastTrack(lh, p)
        P = replace(P0, fb_eps=rx._fb_eps(c.forecast)) if c.scaler == "pab" else P0
        s = summarize(Simulator(c, P, arr, lam, forecast=fc).run(), lam, bursts, P0)
        s.update(profile=prof, seed=seed, config=name, t_start=ts_)
        rows.append(s)
    return rows


def main():
    jobs = [(p, s, t) for p in ("S3", "S5") for s in range(10) for t in T_STARTS]
    rows = []
    with ProcessPoolExecutor(max_workers=10) as ex:
        for r in ex.map(work, jobs):
            rows += r
    d = pd.DataFrame(rows)
    d.to_csv(ROOT / "results" / "sensitivity_tstart.csv", index=False)
    g = d.groupby(["profile", "t_start", "config"])[["err_rate", "slo_viol_s", "pod_s"]].median().round(3)
    print(g.to_string())


if __name__ == "__main__":
    main()
