"""Дискретный симулятор кластера (шаг 1 с): поды, очереди, балансировщики, HPA, ПАБ.

Модель (все допущения явно перечислены в README):
- под обслуживает mu запросов/с; очередь ограничена qmax, лишние запросы теряются;
- новый под готов через t_start секунд и ещё t_warm секунд "прогревается"
  (пропускная способность растёт с cold_frac до 1);
- запросы приходят Пуассоновским потоком по профилю lam(t) (общие для всех конфигураций);
- задержка запроса = (длина очереди перед ним + 1) / пропускная способность пода;
- метрики для HPA запаздывают на metric_delay секунд.
"""
from collections import deque
from dataclasses import dataclass, field, replace

import numpy as np

# Границы гистограммы задержек, с (лог-шкала): точность ~2.4 % на бин
HIST_EDGES = np.concatenate([[0.0], np.logspace(-2, 1.3, 241)])


@dataclass
class SimParams:
    mu: float = 50.0            # запросов/с на под
    qmax: float = 100.0         # длина очереди (макс. ожидание ~2 с)
    t_start: float = 30.0       # запуск пода: планирование + старт контейнера
    t_warm: float = 20.0        # прогрев после готовности
    cold_frac: float = 0.3      # доля пропускной способности у холодного пода
    n_min: int = 2
    n_max: int = 40
    slo: float = 0.5            # SLO по задержке, с
    # HPA
    hpa_period: int = 15
    hpa_target: float = 0.7
    hpa_tol: float = 0.1
    hpa_down_window: int = 120  # окно стабилизации вниз (в K8s 300 с, сжато вместе со временем)
    metric_delay: int = 10      # T_scrape
    metric_window: int = 15
    # ПАБ
    dt_ctrl: int = 5
    rho: float = 0.7            # целевая загрузка при расчёте N*
    p_thr: float = 0.5
    down_hold: int = 60         # выдержка перед уменьшением числа подов
    t_ramp: float = 20.0        # время разгона веса нового пода
    eps_w: float = 1.0
    fb_k: int = 24              # окно оценки ошибки прогноза (шагов dt_ctrl)
    fb_eps: float = 0.5         # порог относительной ошибки для fallback


@dataclass
class Config:
    name: str
    balancer: str               # 'rr' | 'lc' | 'wlc'
    scaler: str                 # 'fixed' | 'hpa' | 'hpa_aggr' | 'pab'
    forecast: str = ""          # имя модели прогноза для 'pab'
    fallback: bool = True
    hpa_overrides: dict = field(default_factory=dict)


def count_latency_le(e, q0, cap, acc):
    """Сколько из acc запросов пода имеют задержку <= e (e - вектор порогов, форма (E,)).

    Запросы равномерно приходят в течение секунды. Для j-го запроса
    задержка = (max(0, q0 + s*j) + 1) / cap, где s = 1 - cap/acc
    (очередь растёт при s>0 и рассасывается при s<0). Возвращает массив (m, E).
    """
    q0 = q0[:, None]
    cap = cap[:, None]
    acc = acc[:, None]
    R = e[None, :] * cap - 1.0                      # допустимый "хвост очереди" перед запросом
    with np.errstate(divide="ignore", invalid="ignore"):
        s = 1.0 - cap / np.maximum(acc, 1e-9)
        up = np.clip(np.floor((R - q0) / s), 0, acc)
        j0 = np.maximum(1.0, np.ceil((q0 - R) / np.abs(s)))
        down = np.clip(acc - j0 + 1, 0, acc)
    flat = np.where(q0 <= R, acc, 0.0)
    cnt = np.where(s > 1e-12, up, np.where(s < -1e-12, down, flat))
    cnt = np.where(R < 0, 0.0, cnt)
    return np.where(acc > 0, cnt, 0.0)


# ---------------------------------------------------------------- балансировка

def _integerize(x, n):
    """Округляет неотрицательный вектор x (сумма n) до целых с той же суммой."""
    base = np.floor(x).astype(np.int64)
    rest = int(n - base.sum())
    if rest > 0:
        order = np.argsort(-(x - base))
        base[order[:rest]] += 1
    return base


def distribute_rr(n, m, rr_state):
    base = np.full(m, n // m, dtype=np.int64)
    rem = n - base.sum()
    for j in range(rem):
        base[(rr_state + j) % m] += 1
    return base, (rr_state + rem) % max(m, 1)


def distribute_lc(n, q, w):
    """Взвешенный Least Connections: минимизируем max (q_i + a_i)/w_i (водозаполнение)."""
    m = len(q)
    if n == 0:
        return np.zeros(m, dtype=np.int64)
    r = q / w
    order = np.argsort(r)
    rs, ws, qs = r[order], w[order], q[order]
    cw, cq = np.cumsum(ws), np.cumsum(qs)
    level = (n + cq) / cw
    k = m - 1
    for i in range(m - 1):
        if level[i] <= rs[i + 1]:
            k = i
            break
    L = level[k]
    a = np.maximum(0.0, L * w - q)
    s = a.sum()
    if s <= 0:
        a = np.full(m, n / m)
    else:
        a = a * (n / s)
    return _integerize(a, n)


# ---------------------------------------------------------------- прогнозы

class ForecastTrack:
    """Предвычисленные прогнозы: lam_hat[k, h] (запр/с) и p[k] для шага k (начало k-го бина)."""

    def __init__(self, lam_hat, p):
        self.lam_hat = np.asarray(lam_hat)
        self.p = np.asarray(p)

    def at(self, k):
        k = min(k, len(self.p) - 1)
        return self.lam_hat[k], float(self.p[k])


# ---------------------------------------------------------------- симулятор

class Simulator:
    def __init__(self, cfg, params, arrivals, lam, forecast=None, n_peak=None):
        self.cfg = cfg
        self.P = replace(params, **cfg.hpa_overrides) if cfg.hpa_overrides else params
        self.arr = np.asarray(arrivals)
        self.lam = lam
        self.T = len(arrivals)
        self.fc = forecast
        self.n_peak = n_peak
        P = self.P

        n0 = max(P.n_min, int(np.ceil(lam[:30].mean() / (P.mu * P.hpa_target))))
        if cfg.scaler == "fixed":
            n0 = n_peak
        self.ready_at = np.full(n0, -1e9)
        self.q = np.zeros(n0)
        self.rr_state = 0
        # история для контроллеров
        self.util_hist = np.zeros(self.T)           # средняя загрузка готовых подов
        self.hpa_recs = deque()                      # (t, desired) для окна стабилизации
        self.pab_recs = deque()
        self.err_hist = deque(maxlen=P.fb_k)
        self.fallback = False
        self.last_up = -1e9
        self.up_streak = 0
        self.scale_events = 0
        self.fb_seconds = 0
        self.bins = None  # наблюдаемые бины rps, заполняются по ходу
        self._obs = []

    # -- состояние
    def n_total(self):
        return len(self.q)

    def ready_mask(self, t):
        return self.ready_at <= t

    def cap(self, t):
        age = t - self.ready_at
        f = self.P.cold_frac + (1 - self.P.cold_frac) * np.clip(age / self.P.t_warm, 0, 1)
        return np.where(age >= 0, self.P.mu * f, 0.0)

    # -- масштабирование
    def scale_to(self, t, target):
        P = self.P
        target = int(min(max(target, P.n_min), P.n_max))
        n = self.n_total()
        if target > n:
            add = target - n
            self.ready_at = np.concatenate([self.ready_at, np.full(add, t + P.t_start)])
            self.q = np.concatenate([self.q, np.zeros(add)])
            self.scale_events += 1
            self.last_up = t
        elif target < n:
            remove = n - target
            # сначала убираем неготовые (ещё стартующие), затем самые незагруженные
            idx = np.arange(n)
            pending = idx[self.ready_at > t]
            ready = idx[self.ready_at <= t]
            ready = ready[np.argsort(self.q[ready])]
            order = np.concatenate([pending[np.argsort(-self.ready_at[pending])], ready])
            drop = order[:remove]
            keep = np.ones(n, dtype=bool)
            keep[drop] = False
            moved = self.q[drop][self.ready_at[drop] <= t].sum()
            self.ready_at, self.q = self.ready_at[keep], self.q[keep]
            rd = self.ready_at <= t
            if moved > 0 and rd.any():
                self.q[rd] += moved / rd.sum()
            self.scale_events += 1

    # -- HPA
    def hpa_desired(self, t):
        P = self.P
        hi = t - P.metric_delay
        lo = max(0, hi - P.metric_window)
        if hi <= 0:
            return self.n_total()
        m = self.util_hist[lo:hi].mean()
        n = self.n_total()
        n_ready = int((self.ready_at <= t).sum())
        ratio = m / P.hpa_target
        if abs(ratio - 1.0) <= P.hpa_tol:
            return n
        if ratio > 1.0 and n_ready < n:
            # как в Kubernetes: при росте нагрузки неготовые поды считаются с нулевой загрузкой
            ratio_all = m * n_ready / (n * P.hpa_target)
            if ratio_all < 1.0 or abs(ratio_all - 1.0) <= P.hpa_tol:
                return n
            return int(np.ceil(n * ratio_all))
        return int(np.ceil(n * ratio))

    def hpa_step(self, t, down_window, period_up=True):
        P = self.P
        desired = self.hpa_desired(t)
        n = self.n_total()
        self.hpa_recs.append((t, desired))
        while self.hpa_recs and self.hpa_recs[0][0] < t - down_window:
            self.hpa_recs.popleft()
        if desired > n:
            limit = n + max(n, 4)
            self.scale_to(t, min(desired, limit))
        elif desired < n:
            stab = max(d for _, d in self.hpa_recs)
            if stab < n:
                self.scale_to(t, max(stab, P.n_min))

    # -- ПАБ
    def forecast_error_update(self, k):
        """Ошибка прогноза бина k-1, сделанного на шаге k-1."""
        if k < 2 or self.fc is None:
            return
        lam_hat_prev, _ = self.fc.at(k - 1)
        real = self._obs[k - 1]
        err = abs(lam_hat_prev[0] - real) / max(real, 2 * self.P.mu * 0.4)
        self.err_hist.append(err)

    def pab_step(self, t):
        P = self.P
        k = t // P.dt_ctrl
        self.forecast_error_update(k)
        if self.cfg.fallback and len(self.err_hist) >= P.fb_k:
            e = float(np.mean(self.err_hist))
            if not self.fallback and e > P.fb_eps:
                self.fallback = True
            elif self.fallback and e < P.fb_eps / 2:
                self.fallback = False
        if self.fallback:
            if t % P.hpa_period == 0:
                self.hpa_step(t, P.hpa_down_window)
            return
        lam_hat, p = self.fc.at(k)
        n_star = max(P.n_min, int(np.ceil(np.max(lam_hat) / (P.mu * P.rho))))
        self.pab_recs.append((t, n_star))
        while self.pab_recs and self.pab_recs[0][0] < t - P.down_hold:
            self.pab_recs.popleft()
        n = self.n_total()
        if n_star > n:
            if p >= P.p_thr:
                self.scale_to(t, n_star)          # упреждающий запуск при ожидаемом всплеске
                self.up_streak = 0
            else:
                self.up_streak += 1               # плавный рост: подтверждение 2 шага подряд
                if self.up_streak >= 2:
                    self.scale_to(t, n_star)
                    self.up_streak = 0
        else:
            self.up_streak = 0
            if p < P.p_thr and t - self.last_up >= P.down_hold:
                stab = max(d for _, d in self.pab_recs)
                if stab < n:
                    self.scale_to(t, stab)

    # -- веса
    def weights(self, t, ready_idx):
        P = self.P
        if self.cfg.balancer != "wlc":
            return np.ones(len(ready_idx))
        if self.fallback:
            return np.ones(len(ready_idx))
        k = max(t // P.dt_ctrl - 1, 0)
        obs = self._obs[k] if self._obs else self.lam[0]
        lam_i = obs / max(len(ready_idx), 1)
        age = t - self.ready_at[ready_idx]
        ramp = np.clip(age / P.t_ramp, 0.1, 1.0)
        return np.maximum(P.eps_w, P.mu - lam_i) * ramp

    # -- основной цикл
    def run(self):
        P, T = self.P, self.T
        hist = np.zeros(len(HIST_EDGES))
        arrived = np.zeros(T)
        dropped = np.zeros(T)
        late = np.zeros(T)
        n_tot = np.zeros(T)
        n_rdy = np.zeros(T)
        jain = np.ones(T)
        cv = np.zeros(T)
        cap_rdy = np.zeros(T)
        fb_flag = np.zeros(T)

        for t in range(T):
            # обновляем наблюдаемый бин (rps), когда завершается бин dt_ctrl
            if t % P.dt_ctrl == 0 and t > 0:
                self._obs.append(self.arr[t - P.dt_ctrl:t].sum() / P.dt_ctrl)
            # контроль
            sc = self.cfg.scaler
            if sc in ("hpa", "hpa_aggr"):
                per = P.hpa_period if sc == "hpa" else 5
                win = P.hpa_down_window if sc == "hpa" else 30
                if t % per == 0 and t > 0:
                    self.hpa_step(t, win)
            elif sc == "pab" and t % P.dt_ctrl == 0:
                self.pab_step(t)

            fb_flag[t] = self.fallback
            cap = self.cap(t)
            ready_idx = np.where(self.ready_at <= t)[0]
            m = len(ready_idx)
            n = int(self.arr[t])
            arrived[t] = n
            n_tot[t] = self.n_total()
            n_rdy[t] = m
            cap_rdy[t] = cap[ready_idx].sum() if m else 0.0
            if m == 0:
                dropped[t] = n
                continue

            qr = self.q[ready_idx]
            cr = cap[ready_idx]
            if self.cfg.balancer == "rr":
                a, self.rr_state = distribute_rr(n, m, self.rr_state)
            else:
                a = distribute_lc(n, qr, self.weights(t, ready_idx))
            # в очередь помещается qmax; за секунду ещё cr запросов успевает уйти на обработку
            acc = np.minimum(a, np.maximum(P.qmax - qr, 0) + cr).astype(np.float64)
            dropped[t] = n - acc.sum()

            # гистограмма задержек и счётчик нарушений SLO
            cle = count_latency_le(HIST_EDGES, qr, cr, acc)       # (m, E)
            hist[1:] += np.diff(cle, axis=1).sum(axis=0)
            hist[0] += cle[:, 0].sum()
            hist[-1] += (acc - cle[:, -1]).sum()
            late[t] = (acc - count_latency_le(np.array([P.slo]), qr, cr, acc)[:, 0]).sum()

            busy = np.minimum(1.0, (qr + acc) / np.maximum(cr, 1e-9))
            served = np.minimum(qr + acc, cr)
            self.q[ready_idx] = np.maximum(qr + acc - cr, 0)
            self.util_hist[t] = busy.mean()
            x = busy
            if x.sum() > 0:
                jain[t] = x.sum() ** 2 / (m * (x ** 2).sum())
                cv[t] = x.std() / x.mean()

        return dict(hist=hist, arrived=arrived, dropped=dropped, late=late, n_total=n_tot,
                    n_ready=n_rdy, cap_ready=cap_rdy, jain=jain, cv=cv, fb=fb_flag,
                    scale_events=self.scale_events, util=self.util_hist)
