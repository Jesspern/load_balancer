"""Модели прогнозирования интенсивности запросов.

Все модели работают на бинах rps длиной dt=5 с. На шаге k по бинам 0..k-1 строится прогноз
на H бинов вперёд (бины k..k+H-1) и вероятность всплеска p.

Простые модели (naive, ma, linear, ar) и нейросетевые (lstm, gru) имеют одинаковый интерфейс:
    model.fit(bins_list)               - обучение на списке рядов
    model.track(bins) -> (lam_hat, p)  - прогнозы для всех шагов k = 0..K-1
"""
import numpy as np

W = 24        # окно истории, бинов (120 с)
H = 12        # горизонт, бинов (60 с >= T_start)
DT = 5
PERIOD = 600  # период "суток" для признаков sin/cos, с
BURST_RATIO = 1.4   # метка всплеска: будущий максимум > 1.4 x недавний уровень
BURST_MIN = 30.0    # и рост не менее 30 запросов/с
RECENT = 6


def recent_level(hist):
    return float(np.mean(hist[-RECENT:])) if len(hist) else 0.0


def burst_label(hist, future):
    """1, если в будущем окне есть всплеск относительно недавнего уровня."""
    lvl = recent_level(hist)
    mx = float(np.max(future))
    return float(mx > BURST_RATIO * lvl and mx - lvl > BURST_MIN)


def _pad_hist(bins, k, w=W):
    """История длины w, последние k бинов (если истории мало - дополняем первым значением)."""
    h = np.asarray(bins[max(0, k - w):k], dtype=float)
    if len(h) == 0:
        return np.full(w, 100.0)
    if len(h) < w:
        h = np.concatenate([np.full(w - len(h), h[0]), h])
    return h


def _future(bins, k, h=H):
    f = np.asarray(bins[k:k + h], dtype=float)
    if len(f) < h:
        f = np.concatenate([f, np.full(h - len(f), f[-1] if len(f) else 0.0)])
    return f


def _rule_p(hist, lam_hat):
    lvl = recent_level(hist)
    mx = float(np.max(lam_hat))
    return float(mx > BURST_RATIO * lvl and mx - lvl > BURST_MIN)


class _Simple:
    name = "simple"

    def fit(self, bins_list):
        return self

    def predict_one(self, hist):
        raise NotImplementedError

    def track(self, bins):
        K = len(bins)
        out = np.zeros((K, H))
        p = np.zeros(K)
        for k in range(K):
            hist = _pad_hist(bins, k)
            out[k] = np.maximum(self.predict_one(hist), 0)
            p[k] = _rule_p(hist, out[k])
        return out, p


class Naive(_Simple):
    name = "naive"

    def predict_one(self, hist):
        return np.full(H, hist[-1])


class MovingAverage(_Simple):
    name = "ma"

    def predict_one(self, hist):
        return np.full(H, hist[-RECENT:].mean())


class Linear(_Simple):
    """Линейный тренд по последним 6 бинам, экстраполяция с затуханием."""
    name = "linear"

    def predict_one(self, hist):
        y = hist[-RECENT:]
        x = np.arange(RECENT)
        b, a = np.polyfit(x, y, 1)
        steps = np.arange(1, H + 1)
        tau = 4.0
        level = a + b * (RECENT - 1)                      # значение тренда в последнем бине
        pred = level + b * tau * (1 - np.exp(-steps / tau))  # тренд затухает
        return np.clip(pred, 0.3 * y[-1], 4.0 * y[-1])


class PeakHold(_Simple):
    """Простая эвристика без обучения: прогноз = максимум за последние 6 бинов (30 с)."""
    name = "peak"

    def predict_one(self, hist):
        return np.full(H, hist[-RECENT:].max())


class MeanStd(_Simple):
    """Простая эвристика с запасом: среднее + 1 std за последние 12 бинов (60 с)."""
    name = "msd"

    def predict_one(self, hist):
        h = hist[-12:]
        return np.full(H, h.mean() + h.std())


class ARRidge(_Simple):
    """Линейная авторегрессия (ridge) на нормированном окне; замена SARIMA."""
    name = "ar"

    def __init__(self, alpha=1.0):
        self.alpha = alpha
        self.coef = None

    def fit(self, bins_list):
        X, Y = [], []
        for b in bins_list:
            for k in range(W, len(b) - H):
                h = _pad_hist(b, k)
                s = h.mean()
                X.append(np.concatenate([h / s, [1.0]]))
                Y.append(_future(b, k) / s)
        X, Y = np.array(X), np.array(Y)
        A = X.T @ X + self.alpha * np.eye(X.shape[1])
        self.coef = np.linalg.solve(A, X.T @ Y)
        return self

    def predict_one(self, hist):
        s = hist.mean()
        return (np.concatenate([hist / s, [1.0]]) @ self.coef) * s


# ---------------------------------------------------------------- нейросети

def _features(hist, k):
    """Вход модели: окно rps, нормированное на среднее, + sin/cos времени суток."""
    s = hist.mean() + 1e-6
    t = (k - np.arange(W)[::-1] - 1) * DT
    ang = 2 * np.pi * t / PERIOD
    return np.stack([hist / s, np.sin(ang), np.cos(ang)], axis=1), s


class NNForecaster:
    """LSTM / GRU с двумя выходами: прогноз H значений и вероятность всплеска."""

    def __init__(self, kind="lstm", hidden=48, alpha=0.7, beta=0.5, epochs=40, seed=0, lr=3e-3):
        self.kind, self.hidden, self.alpha, self.beta = kind, hidden, alpha, beta
        self.epochs, self.seed, self.lr = epochs, seed, lr
        self.name = kind
        self.net = None
        self.history = []

    def _build(self):
        import torch.nn as nn

        class Net(nn.Module):
            def __init__(s, kind, hidden):
                super().__init__()
                rnn = nn.LSTM if kind == "lstm" else nn.GRU
                s.rnn = rnn(3, hidden, batch_first=True)
                s.head_lam = nn.Linear(hidden, H)
                s.head_p = nn.Linear(hidden, 1)

            def forward(s, x):
                out, _ = s.rnn(x)
                h = out[:, -1]
                return s.head_lam(h), s.head_p(h).squeeze(-1)

        return Net(self.kind, self.hidden)

    def _dataset(self, bins_list):
        X, Y, L = [], [], []
        for b in bins_list:
            for k in range(W, len(b) - H):
                h = _pad_hist(b, k)
                f, s = _features(h, k)
                fut = _future(b, k)
                X.append(f)
                Y.append(fut / s)
                L.append(burst_label(h, fut))
        return (np.array(X, dtype=np.float32), np.array(Y, dtype=np.float32),
                np.array(L, dtype=np.float32))

    def _loss(self, lam_hat, p_logit, y, lab):
        import torch
        import torch.nn.functional as F
        e = y - lam_hat                      # e>0: недооценка (прогноз ниже факта)
        under = torch.clamp(e, min=0)
        over = torch.clamp(-e, min=0)
        reg = (self.alpha * under + (1 - self.alpha) * over).mean()
        bce = F.binary_cross_entropy_with_logits(p_logit, lab)
        return reg + self.beta * bce

    def fit(self, bins_list, val_list=None):
        import torch
        torch.manual_seed(self.seed)
        np.random.seed(self.seed)
        torch.set_num_threads(4)
        self.net = self._build()
        X, Y, L = self._dataset(bins_list)
        Xt, Yt, Lt = map(torch.tensor, (X, Y, L))
        if val_list:
            Xv, Yv, Lv = map(torch.tensor, self._dataset(val_list))
        opt = torch.optim.Adam(self.net.parameters(), lr=self.lr)
        best, best_state = 1e9, None
        n = len(Xt)
        self.history = []
        for ep in range(self.epochs):
            self.net.train()
            perm = torch.randperm(n)
            tot = 0.0
            for i in range(0, n, 256):
                idx = perm[i:i + 256]
                opt.zero_grad()
                lam_hat, pl = self.net(Xt[idx])
                loss = self._loss(lam_hat, pl, Yt[idx], Lt[idx])
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.net.parameters(), 1.0)
                opt.step()
                tot += loss.item() * len(idx)
            tr = tot / n
            if val_list:
                self.net.eval()
                with torch.no_grad():
                    lh, pl = self.net(Xv)
                    vl = float(self._loss(lh, pl, Yv, Lv))
            else:
                vl = tr
            self.history.append((ep, tr, vl))
            if vl < best:
                best = vl
                best_state = {k: v.clone() for k, v in self.net.state_dict().items()}
        self.net.load_state_dict(best_state)
        self.net.eval()
        return self

    def track(self, bins):
        import torch
        K = len(bins)
        feats, scales = [], []
        for k in range(K):
            h = _pad_hist(bins, k)
            f, s = _features(h, k)
            feats.append(f)
            scales.append(s)
        with torch.no_grad():
            lh, pl = self.net(torch.tensor(np.array(feats, dtype=np.float32)))
        lam = np.maximum(lh.numpy() * np.array(scales)[:, None], 0)
        p = torch.sigmoid(pl).numpy()
        return lam, p

    def save(self, path):
        import torch
        torch.save({"state": self.net.state_dict(), "kind": self.kind, "hidden": self.hidden,
                    "history": self.history}, path)


# ---------------------------------------------------------------- улучшенная версия (v2)

def _features2(hist, k, w):
    """Признаки v2: нормированный ряд, его разности, лог-уровень, sin/cos времени суток."""
    s = hist.mean() + 1e-6
    x = hist / s
    dx = np.concatenate([[0.0], np.diff(x)])
    t = (k - np.arange(w)[::-1] - 1) * DT
    ang = 2 * np.pi * t / PERIOD
    lvl = np.full(w, np.log(s / 100.0))
    return np.stack([x, dx * 5, lvl, np.sin(ang), np.cos(ang)], axis=1), s


class NNForecaster2:
    """Ансамбль LSTM/GRU: окно 36 бинов, признаки v2, асимметричная потеря + BCE.

    Прогноз ансамбля - среднее по членам, p - среднее вероятностей."""

    def __init__(self, kind="lstm", hidden=64, window=36, alpha=0.7, beta=0.5, epochs=40,
                 n_members=5, lr=3e-3, seed=0):
        self.kind, self.hidden, self.window = kind, hidden, window
        self.alpha, self.beta, self.epochs = alpha, beta, epochs
        self.n_members, self.lr, self.seed = n_members, lr, seed
        self.nets = []
        self.history = []
        self.name = kind + "2"

    def _build(self):
        import torch.nn as nn

        class Net(nn.Module):
            def __init__(s, kind, hidden):
                super().__init__()
                rnn = nn.LSTM if kind == "lstm" else nn.GRU
                s.rnn = rnn(5, hidden, batch_first=True)
                s.head_lam = nn.Linear(hidden, H)
                s.head_p = nn.Linear(hidden, 1)

            def forward(s, x):
                out, _ = s.rnn(x)
                h = out[:, -1]
                return s.head_lam(h), s.head_p(h).squeeze(-1)

        return Net(self.kind, self.hidden)

    def _dataset(self, bins_list):
        X, Y, L = [], [], []
        w = self.window
        for b in bins_list:
            for k in range(w, len(b) - H):
                h = _pad_hist(b, k, w)
                f, s = _features2(h, k, w)
                fut = _future(b, k)
                X.append(f)
                Y.append(fut / s)
                L.append(burst_label(h, fut))
        return (np.array(X, dtype=np.float32), np.array(Y, dtype=np.float32),
                np.array(L, dtype=np.float32))

    def _loss(self, lam_hat, p_logit, y, lab):
        import torch
        import torch.nn.functional as F
        e = y - lam_hat
        reg = (self.alpha * torch.clamp(e, min=0) + (1 - self.alpha) * torch.clamp(-e, min=0)).mean()
        return reg + self.beta * F.binary_cross_entropy_with_logits(p_logit, lab)

    def fit(self, bins_list, val_list):
        import torch
        torch.set_num_threads(4)
        X, Y, L = map(torch.tensor, self._dataset(bins_list))
        Xv, Yv, Lv = map(torch.tensor, self._dataset(val_list))
        self.nets, self.history = [], []
        n = len(X)
        for m in range(self.n_members):
            torch.manual_seed(self.seed * 100 + m)
            net = self._build()
            opt = torch.optim.Adam(net.parameters(), lr=self.lr)
            sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=self.epochs)
            best, best_state, hist = 1e9, None, []
            for ep in range(self.epochs):
                net.train()
                perm = torch.randperm(n)
                tot = 0.0
                for i in range(0, n, 256):
                    idx = perm[i:i + 256]
                    opt.zero_grad()
                    lh, pl = net(X[idx])
                    loss = self._loss(lh, pl, Y[idx], L[idx])
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
                    opt.step()
                    tot += loss.item() * len(idx)
                sched.step()
                net.eval()
                with torch.no_grad():
                    lh, pl = net(Xv)
                    vl = float(self._loss(lh, pl, Yv, Lv))
                hist.append((m, ep, tot / n, vl))
                if vl < best:
                    best, best_state = vl, {k_: v.clone() for k_, v in net.state_dict().items()}
            net.load_state_dict(best_state)
            net.eval()
            self.nets.append(net)
            self.history += hist
        return self

    def track(self, bins):
        import torch
        K, w = len(bins), self.window
        feats, scales = [], []
        for k in range(K):
            f, s = _features2(_pad_hist(bins, k, w), k, w)
            feats.append(f)
            scales.append(s)
        x = torch.tensor(np.array(feats, dtype=np.float32))
        lam, ps = [], []
        with torch.no_grad():
            for net in self.nets:
                lh, pl = net(x)
                lam.append(lh.numpy())
                ps.append(torch.sigmoid(pl).numpy())
        lam = np.maximum(np.mean(lam, axis=0) * np.array(scales)[:, None], 0)
        return lam, np.mean(ps, axis=0)

    def save(self, path):
        import torch
        torch.save({"states": [n.state_dict() for n in self.nets], "kind": self.kind,
                    "hidden": self.hidden, "window": self.window, "history": self.history}, path)

    @classmethod
    def load(cls, path):
        import torch
        ck = torch.load(path, weights_only=False)
        m = cls(kind=ck["kind"], hidden=ck["hidden"], window=ck["window"])
        for st in ck["states"]:
            net = m._build()
            net.load_state_dict(st)
            net.eval()
            m.nets.append(net)
        return m


def make_model(name, seed=0):
    if name == "naive":
        return Naive()
    if name == "ma":
        return MovingAverage()
    if name == "linear":
        return Linear()
    if name == "peak":
        return PeakHold()
    if name == "msd":
        return MeanStd()
    if name == "ar":
        return ARRidge()
    if name in ("lstm", "gru"):
        return NNForecaster(kind=name, seed=seed)
    if name in ("lstm2", "gru2"):
        return NNForecaster2(kind=name[:-1], seed=seed)
    raise ValueError(name)
