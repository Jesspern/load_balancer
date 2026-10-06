"""Генераторы профилей нагрузки S1-S5.

Каждый генератор возвращает (lam, bursts): lam - массив интенсивности (запросов/с)
по секундам, bursts - список моментов начала всплесков (для метрики времени реакции).
Все профили воспроизводимы по seed.
"""
import numpy as np

T_DEFAULT = 1800  # длительность одного прогона, с


def _noise(rng, n, sigma=0.04, rho=0.95):
    """Медленный AR(1)-шум: множитель около 1 (реальная нагрузка не гладкая)."""
    e = rng.normal(0, sigma * np.sqrt(1 - rho ** 2), n)
    x = np.zeros(n)
    for i in range(1, n):
        x[i] = rho * x[i - 1] + e[i]
    return np.exp(x)


def _smooth_step(n, t0, dur, ramp):
    """Единичный импульс с линейными фронтами длиной ramp секунд."""
    t = np.arange(n)
    up = np.clip((t - t0) / ramp, 0, 1)
    down = np.clip((t0 + dur - t) / ramp, 0, 1)
    return np.minimum(up, down)


def s1_stationary(seed, T=T_DEFAULT, base=150.0):
    rng = np.random.default_rng(seed)
    return base * _noise(rng, T), []


def s2_ramp(seed, T=T_DEFAULT, lo=100.0, hi=450.0):
    rng = np.random.default_rng(seed)
    t = np.arange(T)
    t0, t1 = 300, 1200
    shape = np.clip((t - t0) / (t1 - t0), 0, 1)
    return (lo + (hi - lo) * shape) * _noise(rng, T), []


def s3_step_burst(seed, T=T_DEFAULT, base=100.0, amp=3.5, dur=150):
    """Три ступенчатых всплеска с быстрым фронтом (5 с)."""
    rng = np.random.default_rng(seed)
    lam = np.full(T, base)
    bursts = []
    for t0 in (400, 900, 1400):
        t0j = t0 + int(rng.integers(-30, 30))
        lam = lam + base * (amp - 1) * _smooth_step(T, t0j, dur, 5)
        bursts.append(t0j)
    return lam * _noise(rng, T), bursts


def s4_daily(seed, T=T_DEFAULT, base=250.0, amp=150.0, period=600):
    """Суточная цикличность в сжатом времени: сутки = period секунд."""
    rng = np.random.default_rng(seed)
    t = np.arange(T)
    phase = rng.uniform(0, 2 * np.pi)
    lam = base + amp * np.sin(2 * np.pi * t / period + phase)
    return lam * _noise(rng, T), []


def s5_random_bursts(seed, T=T_DEFAULT, base=100.0, rate=1 / 250.0,
                     amp_median=2.5, amp_sigma=0.4, dur_range=(50, 140)):
    """Пуассоновский поток всплесков, амплитуда логнормальная."""
    rng = np.random.default_rng(seed)
    lam = np.full(T, base)
    bursts = []
    t = 150.0
    while True:
        t += rng.exponential(1 / rate)
        if t > T - 100:
            break
        amp = amp_median * np.exp(rng.normal(0, amp_sigma))
        amp = min(amp, 6.0)
        dur = rng.uniform(*dur_range)
        ramp = rng.uniform(5, 25)
        lam = lam + base * (amp - 1) * _smooth_step(T, int(t), int(dur), max(1, int(ramp)))
        bursts.append(int(t))
    return lam * _noise(rng, T), bursts


def _trapezoid(n, t0, ramp_up, plateau, ramp_down):
    """Импульс: плавный рост (smoothstep), плато, плавный спад."""
    t = np.arange(n, dtype=float)
    up = np.clip((t - t0) / max(ramp_up, 1), 0, 1)
    up = up * up * (3 - 2 * up)
    t1 = t0 + ramp_up + plateau
    down = np.clip((t1 + ramp_down - t) / max(ramp_down, 1), 0, 1)
    down = down * down * (3 - 2 * down)
    return np.minimum(up, down)


def s6_flash_crowds(seed, T=T_DEFAULT, base=150.0, daily_amp=60.0, period=900,
                    rate=1 / 300.0, amp_median=2.0, amp_sigma=0.35, spike_rate=1 / 700.0):
    """Синтетический аналог реальной трассы: суточный цикл + нарастающие "флэш-краудс"
    (рост 40-120 с, т.е. у всплеска есть предвестники) + редкие внезапные скачки (непредсказуемые).

    Профиль добавлен после диагностики прогнозиста (см. README), а не заранее."""
    rng = np.random.default_rng(seed)
    t = np.arange(T)
    phase = rng.uniform(0, 2 * np.pi)
    lvl = base + daily_amp * np.sin(2 * np.pi * t / period + phase)
    lam = lvl.copy()
    bursts = []
    tt = 100.0
    while True:
        tt += rng.exponential(1 / rate)
        if tt > T - 250:
            break
        amp = amp_median * np.exp(rng.normal(0, amp_sigma))
        ru, pl, rd = rng.uniform(40, 120), rng.uniform(40, 100), rng.uniform(40, 100)
        lam = lam + lvl * (amp - 1) * _trapezoid(T, tt, ru, pl, rd)
        bursts.append(int(tt))
    tt = 100.0
    while True:  # внезапные скачки без предвестников
        tt += rng.exponential(1 / spike_rate)
        if tt > T - 100:
            break
        lam = lam + lvl * 0.9 * _smooth_step(T, int(tt), int(rng.uniform(20, 40)), 5)
        bursts.append(int(tt))
    return lam * _noise(rng, T), sorted(bursts)


def s5x_shifted(seed, T=T_DEFAULT):
    """Сдвиг распределения (для проверки H3): всплески чаще, выше и короче, чем в обучении."""
    return s5_random_bursts(seed, T, rate=1 / 130.0, amp_median=4.5, amp_sigma=0.3,
                            dur_range=(25, 70))


def t1_trace(seed):
    """Реальная трасса Azure Functions 2021: seed = номер 30-минутного окна (см. sim/trace.py)."""
    from .trace import trace_window
    return trace_window(seed)


PROFILES = {
    "S1": s1_stationary,
    "S2": s2_ramp,
    "S3": s3_step_burst,
    "S4": s4_daily,
    "S5": s5_random_bursts,
    "S5X": s5x_shifted,
    "S6": s6_flash_crowds,
    "T1": t1_trace,
}

PROFILE_TITLES = {
    "S1": "S1 стационарная",
    "S2": "S2 рамп",
    "S3": "S3 ступенчатые всплески",
    "S4": "S4 суточная цикличность",
    "S5": "S5 случайные всплески",
    "S5X": "S5X сдвиг распределения",
    "S6": "S6 нарастающие всплески + суточный цикл",
    "T1": "T1 реальная трасса Azure Functions 2021",
}


def make_arrivals(lam, seed):
    """Число поступивших запросов в каждую секунду (Пуассон). Общие для всех конфигураций."""
    rng = np.random.default_rng(seed + 7919)
    return rng.poisson(np.maximum(lam, 0))
