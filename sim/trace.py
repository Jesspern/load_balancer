"""Адаптер реальной трассы Azure Functions 2021 для симулятора.

Правила (зафиксированы до просмотра результатов):
1. Интенсивность lam(t) = суммарный поток всех 119 приложений по секундам, сглаженный окном 5 с
   (шаг управления), умноженный на постоянный масштаб K. K подбирается ТОЛЬКО по обучающим суткам:
   99-й перцентиль 5-секундных бинов обучающей части переводится в TARGET_P99 запросов/с.
2. 14 суток режутся на непересекающиеся окна по WINDOW секунд в реальном времени, без отбора.
3. Хронологическое разбиение окон: первые 60 % - обучение, следующие 20 % - валидация, последние
   20 % - тест.
4. Внутри окна запросы приходят пуассоновским потоком с интенсивностью lam(t) (make_arrivals).

Источник данных: Zhang et al., SOSP 2021 (CC-BY). Издатель изменил временные метки.
"""
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
# Какая трасса и какие модели используются, задаётся переменными окружения:
#   NIR_TRACE = azure (по умолчанию) | huawei
#   NIR_MODEL_TAG = метка набора моделей (по умолчанию = метка трассы); "trace" = модели, обученные на Azure
TRACE = os.environ.get("NIR_TRACE", "azure")
TAG = {"azure": "trace", "huawei": "huawei"}[TRACE]      # метка файлов моделей
MODEL_TAG = os.environ.get("NIR_MODEL_TAG", TAG)
TITLE = {"azure": "Azure Functions 2021", "huawei": "Huawei Private Cloud 2023"}[TRACE]
SRC = ROOT / "data" / "AzureFunctionsInvocationTraceForTwoWeeksJan2021.txt"
CACHE = ROOT / "results" / "cache" / ("azure_rps_1s.npy" if TRACE == "azure" else "huawei_rps_1s.npy")
_out = {"azure": "trace", "huawei": "trace_huawei"}[TRACE]
if MODEL_TAG != TAG:
    _out += "_xfer"          # перенос: модели обучены на другой трассе
OUT = ROOT / "results" / _out

WINDOW = 1800
TARGET_P99 = 400.0     # запросов/с, цель для 99-го перцентиля обучающей части
SMOOTH = 5             # секунд
SPLIT = (0.6, 0.8)     # границы обучение/валидация/тест по окнам

_cache = {}


def counts_per_second():
    """Число вызовов в каждую секунду трассы (кэшируется на диске)."""
    if "c" in _cache:
        return _cache["c"]
    if CACHE.exists():
        c = np.load(CACHE)
    elif TRACE == "huawei":
        raise FileNotFoundError(f"нет кэша {CACHE}: сначала python experiments/huawei_prepare.py")
    else:
        d = pd.read_csv(SRC)
        start = (d["end_timestamp"] - d["duration"]).values
        n = int(np.ceil(d["end_timestamp"].max()))
        c = np.bincount(start.astype(int), minlength=n)[:n].astype(np.float32)
        CACHE.parent.mkdir(parents=True, exist_ok=True)
        np.save(CACHE, c)
    _cache["c"] = c
    return c


def smoothed():
    if "s" not in _cache:
        c = counts_per_second().astype(np.float64)
        k = np.ones(SMOOTH) / SMOOTH
        _cache["s"] = np.convolve(c, k, mode="same")
    return _cache["s"]


def n_windows():
    return len(counts_per_second()) // WINDOW


def split_indices():
    n = n_windows()
    a, b = int(SPLIT[0] * n), int(SPLIT[1] * n)
    return {"train": list(range(0, a)), "val": list(range(a, b)), "test": list(range(b, n))}


def scale_constant():
    """K по обучающим окнам: p99 5-секундных бинов -> TARGET_P99. Сохраняется в results/trace/scale.json."""
    path = OUT / "scale.json"
    if path.exists():
        return json.load(open(path))["K"]
    c = counts_per_second()
    tr = split_indices()["train"]
    train = c[: (tr[-1] + 1) * WINDOW].astype(np.float64)
    bins = train[: len(train) // 5 * 5].reshape(-1, 5).mean(axis=1)
    p99 = float(np.percentile(bins, 99))
    K = TARGET_P99 / p99
    OUT.mkdir(parents=True, exist_ok=True)
    json.dump({"K": K, "train_p99_rps_5s": p99, "target_p99": TARGET_P99, "window_s": WINDOW,
               "smooth_s": SMOOTH, "n_windows": n_windows(),
               "n_train": len(tr), "n_val": len(split_indices()["val"]),
               "n_test": len(split_indices()["test"])}, open(path, "w"), indent=2)
    return K


def detect_bursts(lam, min_gap=120):
    """Моменты начала всплесков для метрики времени реакции: интенсивность > 3x медианы
    предыдущих 300 с и > 150 запросов/с; не чаще раза в min_gap секунд."""
    s = pd.Series(lam)
    base = s.rolling(300, min_periods=60).median().shift(1)
    cond = (s > 3 * base) & (s > 150)
    onsets, last = [], -10 ** 9
    for t in np.where(cond.values)[0]:
        if t - last >= min_gap:
            onsets.append(int(t))
            last = t
    return onsets


def trace_window(idx, K=None):
    """Профиль для симулятора: (lam, bursts) для окна с номером idx. seed = номер окна.
    K - масштаб интенсивности (по умолчанию определён по обучающим окнам; переопределяется для проверки
    чувствительности)."""
    K = scale_constant() if K is None else K
    s = smoothed()
    lam = K * s[idx * WINDOW:(idx + 1) * WINDOW]
    lam = np.maximum(lam, 1.0)
    return lam, detect_bursts(lam)
