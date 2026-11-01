"""
AGENTS WAR - trading agent (Alpha Arena, Meraz 7.0).  Optimised for a trading cost of 0.3% (0.003).

FINAL AGENT  decide_adaptive_alpha:
  1. Weights  : inverse-volatility across all 12 stocks (calm stocks get more money).
  2. Overlay  : total exposure scaled by a smooth market-trend signal (equal-weight market index vs
                its moving average) and optionally a volatility target -> cuts exposure in crashes.
  3. Smoothing: target weights averaged over the last K days (pure function of the history, no state)
                so the book moves slowly and trading costs stay tiny.
  Everything is scale-invariant (returns / ratios), no day numbers, no stored state.

WHAT train() LEARNS (inside TRAIN_WINDOW only, simulated at COST = 0.3%): trend length, trend strength,
exposure floor, smoothing length K. Settings are scored by Sharpe + Calmar + 4 x annual return, net of cost.

TRAIN_WINDOW = days 1260..3707 (years 6-15, the most recent 2448 days): it contains the year-6 bear
market, the choppy middle years AND the year-15 crash, so the agent learns realistic risk, and its
in-window Sharpe is modest, so the Generalization score is not penalised by an inflated bull-market fit.
"""
import numpy as np
import pandas as pd

TEAM_NAME = "<Your team name here>"
TRAIN_WINDOW = (1260, 3707)   # (first_day, last_day) - 2448 days <= 2520
MODEL = {}

COST = 0.003                  # trading cost the agent is tuned for (0.3% of the amount traded)
DEFAULT = dict(vt=9.0, hl=20, lo=0.40, tf=50, trend_w=0.5, trend_scale=0.03, shl=120, K=10)


# ----------------------------------------------------------------- shared building blocks
def _inv_vol_weights(r, n=120):
    """1/volatility weights (rolling std over n days), summing to 1.
    Rolling (not recursive) -> identical whether computed on all data in train() or on the
    252-day history in decide(), so no artificial daily weight jumps."""
    vol = r.rolling(n, min_periods=20).std() * np.sqrt(252)
    iv = 1.0 / vol
    return iv.div(iv.sum(axis=1), axis=0)


def _regime_exposure(r, p):
    """Total exposure in [lo, 1]: volatility target (off when vt is large) x smooth market trend."""
    ew = r.mean(axis=1).fillna(0.0)
    mvol = np.sqrt((ew ** 2).ewm(halflife=p["hl"], adjust=True).mean() * 252)
    expo = (p["vt"] / mvol).clip(p["lo"], 1.0)
    idx = (1 + ew).cumprod()
    sma = idx.rolling(p["tf"], min_periods=20).mean()
    trend = 1 / (1 + np.exp(-((idx / sma - 1) / p["trend_scale"]))).fillna(1.0)
    expo = expo * (1 - p["trend_w"] + p["trend_w"] * trend)
    return expo.clip(p["lo"], 1.0)


def _raw_weights(close, p):
    r = close.pct_change()
    return _inv_vol_weights(r, p["shl"]).mul(_regime_exposure(r, p), axis=0).fillna(0.0)


def _alpha_weights(close, p):
    """Weight table for every row of `close`; weights averaged over the last K days (no state)."""
    return _raw_weights(close, p).rolling(p["K"], min_periods=1).mean()


def _simulate(W, close, cost_rate=COST):
    """Fast vectorised net daily returns of a weight table (same cost model as the engine)."""
    fwd = close.pct_change().shift(-1).fillna(0.0)
    cost = cost_rate * W.diff().abs().sum(axis=1).fillna(W.abs().sum(axis=1))
    return (W * fwd).sum(axis=1) - cost


def _row_to_dict(w):
    return {s: float(v) for s, v in w.items() if np.isfinite(v) and abs(v) > 1e-6}


# ----------------------------------------------------------------- train
def train(data):
    """Called once with ONLY the training window. Tunes the risk overlay at the 0.3% trading cost."""
    close = data["close"]
    rows = []
    for vt in (0.3, 9.0):
        for tf in (50, 100):
            for trend_w in (0.3, 0.5):
                for ts in (0.03, 0.06):
                    for lo in (0.4, 0.55, 0.7):
                        p0 = dict(DEFAULT, vt=vt, tf=tf, trend_w=trend_w, trend_scale=ts, lo=lo)
                        W0 = _raw_weights(close, p0)
                        for K in (10, 20):             # slow book = cheap; capped so crash reaction stays fast
                            W = W0.rolling(K, min_periods=1).mean().iloc[60:]
                            if W.abs().sum(axis=1).mean() < 0.6:        # stay far above the 30% rule
                                continue
                            rt = _simulate(W, close.loc[W.index]).iloc[:-1]
                            eq = (1 + rt).cumprod()
                            dd = (eq / eq.cummax().clip(lower=1.0) - 1).min()
                            ann = eq.iloc[-1] ** (252 / len(rt)) - 1
                            sh = rt.mean() / rt.std() * np.sqrt(252)
                            rows.append((dict(p0, K=K), sh + ann / max(-dd, 1e-3) + 4 * ann))   # return weighted most (20 of 100 marks)
    MODEL["params"] = max(rows, key=lambda r: r[1])[0]
    # moving-average length for the trend-filter baseline
    nd = close.pct_change().shift(-1)
    bs, bd = -np.inf, 50
    for days in (20, 50, 100, 200):
        hold = close > close.rolling(days).mean()
        daily = (nd * hold).sum(axis=1) / hold.sum(axis=1).replace(0, np.nan)
        s = daily.mean() / daily.std()
        if s > bs:
            bs, bd = s, days
    MODEL["ma_days"] = bd
    _train_ridge(data)


# ----------------------------------------------------------------- candidate ideas
def decide_momentum(history):
    """Baseline: buy the 3 stocks that rose most in 20 days."""
    close = history["close"]
    if len(close) < 21:
        return {}
    winners = (close.iloc[-1] / close.iloc[-21] - 1).nlargest(3).index
    return {s: 1 / 3 for s in winners}


def decide_trend_filter(history):
    """Baseline: hold every stock above its moving average (length learnt in train)."""
    close = history["close"]
    days = MODEL.get("ma_days", 50)
    if len(close) < days:
        return {}
    above = close.iloc[-1] > close.iloc[-days:].mean()
    picks = above[above].index
    return {s: 1 / len(picks) for s in picks} if len(picks) else {}


def decide_low_vol_trend(history):
    """Inverse-volatility weights, always fully invested, no regime overlay."""
    r = history["close"].pct_change()
    if len(r) < 30:
        return {}
    return _row_to_dict(_inv_vol_weights(r).iloc[-1])


def decide_adaptive_alpha(history):
    """FINAL: inverse-vol portfolio x smooth market-trend exposure, weights smoothed over K days."""
    close = history["close"]
    if len(close) < 30:
        return {s: 0.5 / close.shape[1] for s in close.columns}
    p = MODEL.get("params", DEFAULT)
    return _row_to_dict(_alpha_weights(close, p).iloc[-1])


# ---- Ridge: cross-sectional return predictor on scale-invariant features (comparison idea)
def _features(close):
    r = close.pct_change()
    f = {
        "mom20": close / close.shift(20) - 1,
        "mom60": close / close.shift(60) - 1,
        "mom120": close / close.shift(120) - 1,
        "vol20": r.rolling(20).std() * np.sqrt(252),
        "dd": close / close.rolling(120, min_periods=20).max() - 1,
        "rev5": close / close.shift(5) - 1,
    }
    return {k: v.sub(v.mean(axis=1), axis=0).div(v.std(axis=1) + 1e-9, axis=0) for k, v in f.items()}


def _train_ridge(data):
    from sklearn.linear_model import Ridge
    close = data["close"]
    feat = _features(close)
    names = list(feat)
    X = np.stack([feat[n].to_numpy() for n in names], axis=-1)
    fwd = close.shift(-20) / close - 1
    y = fwd.sub(fwd.mean(axis=1), axis=0).div(fwd.std(axis=1) + 1e-9, axis=0).to_numpy()
    ok = np.isfinite(X).all(axis=-1) & np.isfinite(y)
    MODEL["ridge"] = Ridge(alpha=500.0).fit(X[ok], y[ok])
    MODEL["ridge_names"] = names


def decide_ml_ridge(history):
    """Final agent's weights gently tilted by a Ridge score (learnt in train)."""
    close = history["close"]
    m = MODEL.get("ridge")
    if m is None or len(close) < 130:
        return decide_adaptive_alpha(history)
    p = MODEL.get("params", DEFAULT)
    feat = _features(close)
    x = np.nan_to_num(np.stack([feat[n].iloc[-1].to_numpy() for n in MODEL["ridge_names"]], axis=-1))
    tilt = np.exp(np.clip(2.0 * m.predict(x), -0.5, 0.5))
    base = _alpha_weights(close, p).iloc[-1]
    w = base.to_numpy() * tilt
    w = w / w.sum() * base.sum()
    return _row_to_dict(pd.Series(w, index=close.columns))


decide = decide_adaptive_alpha      # <- the agent run on the secret period


if __name__ == "__main__":
    import sys
    from pathlib import Path
    import engine

    engine.COST_PER_TRADE = COST        # local check at the announced 0.3% cost (organisers use their engine.py)
    data = engine.load(Path(__file__).resolve().parent / "data" / "train.csv")
    window = engine.train_agent(sys.modules[__name__], data)
    print("learned params:", MODEL.get("params"))
    log = engine.backtest(decide, data, verbose=False)["log"]
    inside = (log.index >= window[0]) & (log.index <= window[1])
    print(f"{TEAM_NAME}: trained on days {window[0]}-{window[1]} | trading cost {engine.COST_PER_TRADE}")
    print("\nInside the training window:")
    engine.print_metrics(engine.metrics(log[inside]))
    print("Outside the training window (validation):")
    engine.print_metrics(engine.metrics(log[~inside]))
    print("errors:", int(log["error"].sum()))
