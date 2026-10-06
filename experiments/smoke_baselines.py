"""Быстрая проверка симулятора: базовые конфигурации A1-A4 на одном профиле."""
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from sim.cluster import Config, SimParams, Simulator
from sim.metrics import summarize
from sim.profiles import PROFILES, make_arrivals

P = SimParams()
prof = sys.argv[1] if len(sys.argv) > 1 else "S3"
lam, bursts = PROFILES[prof](0)
arr = make_arrivals(lam, 0)
n_peak = int(np.ceil(lam.max() / (P.mu * 0.8)))
cfgs = [Config("A1", "rr", "fixed"), Config("A2", "rr", "hpa"),
        Config("A3", "lc", "hpa"), Config("A4", "lc", "hpa_aggr")]
for c in cfgs:
    t0 = time.time()
    res = Simulator(c, P, arr, lam, n_peak=n_peak).run()
    s = summarize(res, lam, bursts, P)
    print(c.name, f"{time.time() - t0:.1f}s", {k: round(v, 3) for k, v in s.items()})
