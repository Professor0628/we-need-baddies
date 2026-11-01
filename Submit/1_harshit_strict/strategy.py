"""
AGENTS WAR - your trading agent.  This is the ONLY code file you submit.

How to work:
  1. Choose TRAIN_WINDOW: the block of days your agent learns from. It can be ANY block of
     at most 10 years (2520 days) inside days 0..3707, e.g. (0, 2519), (252, 2771), (1188, 3707).
  2. train(data) is called ONCE with only those days. Learn your parameters / fit your models there.
  3. Write every trading idea as its own function named  decide_<something>(history).
     decide only sees the last year of prices, so all long-term learning must happen in train().
  4. python compare.py          -> tests all your decide_ ideas side by side
     python compare.py windows  -> tests your final agent trained on 6 different 10-year windows
  5. Point  decide  (bottom of this file) at the one you want us to run on the secret period.

Rules for this file:
  * TRAIN_WINDOW, train (optional) and decide must exist with these names.
  * A TRAIN_WINDOW longer than 2520 days is rejected (disqualified).
  * Allowed imports: python standard library, numpy, pandas, scikit-learn.
  * No internet, no reading other files, no hard-coded prices or day numbers in decide,
    and no collecting prices day by day inside decide to get around the 1-year limit.
  * On average your agent must keep at least 30% of the money invested.
  * A full run (training + secret days) must finish within 3 minutes.
"""
import numpy as np
import pandas as pd

TEAM_NAME = "We Need Baddies"
TRAIN_WINDOW = (1188, 3707)      # (first_day, last_day) your agent learns from - at most 2520 days
MODEL = {}                    # whatever train() learns is stored here

# data / history: dict with keys "open", "high", "low", "close", "volume".
#          Each one is a pandas DataFrame: rows = days (last row = today),
#          columns = stocks (STOCK_01 ... STOCK_12).
# decide returns: dict {stock: weight}
#          weight =  0.25 -> put 25% of your money in that stock (buy)
#          weight = -0.10 -> bet 10% against that stock (short sell)
#          stocks you leave out get 0; whatever is unused stays in cash.
#          The sum of |weights| must be <= 1 (scaled down for you if it is more).


def train(data):
    """Called once, with ONLY the days in TRAIN_WINDOW.
    Example: find which moving-average length worked best inside the training window."""
    close = data["close"]
    next_day = close.pct_change().shift(-1)
    best_sharpe, best_days = -np.inf, 50
    for days in [20, 50, 100, 200]:
        hold = close > close.rolling(days).mean()                  # stocks above their average
        daily = (next_day * hold).sum(axis=1) / hold.sum(axis=1).replace(0, np.nan)
        sharpe = daily.mean() / daily.std()
        if sharpe > best_sharpe:
            best_sharpe, best_days = sharpe, days
    MODEL["ma_days"] = best_days

    # 1A. Fixed per-stock weights (inverse volatility)
    rets = close.pct_change().iloc[1:]
    vol = rets.std()
    if vol.isna().any() or (vol <= 0).any():
        MODEL["iv_weights"] = {stock: 1.0 / 12.0 for stock in close.columns}
    else:
        inv = 1.0 / vol
        w = inv / inv.sum()
        w = w.clip(upper=0.15)
        w = w / w.sum()
        MODEL["iv_weights"] = {stock: float(weight) for stock, weight in w.items()}

    # 1B. Market volatility thresholds
    idx = rets.mean(axis=1)
    for L in [10, 20, 40, 60]:
        roll = idx.rolling(L).std().dropna()
        if not roll.empty:
            MODEL[f"vol_q75_{L}"] = float(roll.quantile(0.75))
            MODEL[f"vol_q90_{L}"] = float(roll.quantile(0.90))
            MODEL[f"vol_q95_{L}"] = float(roll.quantile(0.95))

    # --- Apoorva Model 1: training logic ---
    ret_m1 = close.pct_change()
    fwd_ret_m1 = ret_m1.shift(-1)
    best_score_m1 = -np.inf
    best_ma_m1 = 40
    m63_m1 = close / close.shift(63) - 1.0
    m126_m1 = close / close.shift(126) - 1.0
    m200_m1 = close / close.shift(200) - 1.0
    mom_m1 = 0.15 * m63_m1 + 0.40 * m126_m1 + 0.45 * m200_m1
    vol_m1 = ret_m1.rolling(42).std().replace(0, np.nan)
    risk_adj_m1 = (mom_m1 / vol_m1).clip(lower=0.0)

    for ma_m1 in [35, 40, 45, 50]:
        c_ma = close.rolling(ma_m1).mean()
        trend_ratio = (close / c_ma - 1.0)
        trend_factor = ((trend_ratio + 0.015) / 0.03).clip(lower=0.0, upper=1.0)
        breadth = (close > c_ma).mean(axis=1)

        score = trend_factor * risk_adj_m1 / vol_m1
        score_sum = score.sum(axis=1).replace(0, np.nan)
        norm_w = score.div(score_sum, axis=0).fillna(0.0)

        scale = 0.35 + (0.95 - 0.35) * ((breadth - 0.2) / 0.5).clip(lower=0.0, upper=1.0)
        target_w = norm_w.mul(scale, axis=0)

        smoothed_w = target_w.ewm(alpha=0.10, adjust=False).mean()
        delta_w = smoothed_w.diff().abs().sum(axis=1).fillna(0.0)

        cost = 0.003 * delta_w
        daily_net = (smoothed_w * fwd_ret_m1).sum(axis=1) - cost
        std_m1 = daily_net.std()
        sh_m1 = float(daily_net.mean() / std_m1 * np.sqrt(252)) if std_m1 > 0 else 0.0

        eq = (1 + daily_net).cumprod()
        dd = float((eq / eq.cummax().clip(lower=1.0) - 1.0).min())
        calmar = ((1 + eq.iloc[-1]) ** (252 / len(daily_net)) - 1) / -dd if dd < 0 else 0.0
        score_metric = sh_m1 + 0.3 * calmar
        if score_metric > best_score_m1:
            best_score_m1 = score_metric
            best_ma_m1 = ma_m1

    MODEL["ap_m1_ma_days"] = best_ma_m1
    MODEL["ap_m1_alpha"] = 0.10
    MODEL["ap_prev_w"] = None

    # --- Apoorva Model 2: training logic ---
    rows_m2 = []
    for vt in (0.3, 9.0):
        for tf in (50, 100):
            for trend_w in (0.3, 0.5):
                for ts in (0.03, 0.06):
                    for lo in (0.4, 0.55, 0.7):
                        p0 = dict(AP_DEFAULT, vt=vt, tf=tf, trend_w=trend_w, trend_scale=ts, lo=lo)
                        W0 = _ap_raw_weights(close, p0)
                        for K in (10, 20):
                            W = W0.rolling(K, min_periods=1).mean().iloc[60:]
                            if W.abs().sum(axis=1).mean() < 0.6:
                                continue
                            rt = _ap_simulate(W, close.loc[W.index]).iloc[:-1]
                            eq = (1 + rt).cumprod()
                            dd = (eq / eq.cummax().clip(lower=1.0) - 1).min()
                            ann = eq.iloc[-1] ** (252 / len(rt)) - 1
                            sh = rt.mean() / rt.std() * np.sqrt(252)
                            rows_m2.append((dict(p0, K=K), sh + ann / max(-dd, 1e-3) + 4 * ann))
    if rows_m2:
        MODEL["ap_params"] = max(rows_m2, key=lambda r: r[1])[0]
    else:
        MODEL["ap_params"] = AP_DEFAULT



# --- Helper functions ---

def _base_weights(stocks, kind):
    if kind == "iv" and "iv_weights" in MODEL:
        return {s: MODEL["iv_weights"].get(s, 1.0 / len(stocks)) for s in stocks}
    return {s: 1.0 / len(stocks) for s in stocks}


def _market_returns(history):
    close = history["close"]
    rets = close.pct_change().iloc[1:]
    return rets.mean(axis=1)


def _vol_level(history, L):
    idx = _market_returns(history)
    if len(idx) < L:
        return 1.0
    if f"vol_q75_{L}" not in MODEL or f"vol_q90_{L}" not in MODEL:
        return 1.0
    v = idx.iloc[-L:].std()
    if pd.isna(v):
        return 1.0
    if v > MODEL[f"vol_q90_{L}"]:
        return 0.4
    elif v > MODEL[f"vol_q75_{L}"]:
        return 0.7
    else:
        return 1.0


def _vol_level_strict(history, L):
    idx = _market_returns(history)
    if len(idx) < L:
        return 1.0
    if f"vol_q90_{L}" not in MODEL or f"vol_q95_{L}" not in MODEL:
        return 1.0
    v = idx.iloc[-L:].std()
    if pd.isna(v):
        return 1.0
    if v > MODEL[f"vol_q95_{L}"]:
        return 0.4
    elif v > MODEL[f"vol_q90_{L}"]:
        return 0.7
    else:
        return 1.0


def _trend_level(history):
    idx = _market_returns(history)
    if len(idx) < 200:
        return 1.0
    level = (1 + idx).cumprod()
    if level.iloc[-1] < level.iloc[-200:].mean():
        return 0.5
    else:
        return 1.0


def _scaled(history, kind, exposure):
    stocks = list(history["close"].columns)
    base = _base_weights(stocks, kind)
    return {s: base[s] * exposure for s in stocks}


# --- Apoorva helper functions ---
AP_DEFAULT = dict(vt=9.0, hl=20, lo=0.40, tf=50, trend_w=0.5, trend_scale=0.03, shl=120, K=10)


def _ap_inv_vol_weights(r, n=120):
    vol = r.rolling(n, min_periods=20).std() * np.sqrt(252)
    iv = 1.0 / vol
    return iv.div(iv.sum(axis=1), axis=0).fillna(0.0)


def _ap_regime_exposure(r, p):
    ew = r.mean(axis=1).fillna(0.0)
    mvol = np.sqrt((ew ** 2).ewm(halflife=p["hl"], adjust=True).mean() * 252)
    expo = (p["vt"] / mvol).clip(p["lo"], 1.0)
    idx = (1 + ew).cumprod()
    sma = idx.rolling(p["tf"], min_periods=20).mean()
    trend = 1 / (1 + np.exp(-((idx / sma - 1) / p["trend_scale"]))).fillna(1.0)
    expo = expo * (1 - p["trend_w"] + p["trend_w"] * trend)
    return expo.clip(p["lo"], 1.0)


def _ap_raw_weights(close, p):
    r = close.pct_change()
    return _ap_inv_vol_weights(r, p["shl"]).mul(_ap_regime_exposure(r, p), axis=0).fillna(0.0)


def _ap_alpha_weights(close, p):
    return _ap_raw_weights(close, p).rolling(p["K"], min_periods=1).mean().fillna(0.0)


def _ap_simulate(W, close, cost_rate=0.003):
    fwd = close.pct_change().shift(-1).fillna(0.0)
    cost = cost_rate * W.diff().abs().sum(axis=1).fillna(W.abs().sum(axis=1))
    return (W * fwd).sum(axis=1) - cost


def _ap_row_to_dict(w):
    return {s: float(v) for s, v in w.items() if np.isfinite(v) and v > 1e-6}


def decide_momentum(history):
    """Buy the 3 stocks that went up the most in the last 20 days."""
    close = history["close"]
    if len(close) < 21:
        return {}                                   # not enough data yet -> stay in cash
    past_return = close.iloc[-1] / close.iloc[-21] - 1
    winners = past_return.nlargest(3).index
    return {stock: 1 / 3 for stock in winners}


def decide_trend_filter(history):
    """Hold every stock that is above its moving average (length learnt in train), equally weighted."""
    close = history["close"]
    days = MODEL.get("ma_days", 50)
    if len(close) < days:
        return {}
    above = close.iloc[-1] > close.iloc[-days:].mean()
    picks = above[above].index
    return {stock: 1 / len(picks) for stock in picks} if len(picks) else {}


# Ideas to try next: mean reversion, RSI, volatility filters, stop-losses,
# position sizing by risk, ML models fitted in train(), ...


def decide_inv_vol(history):
    """Hold all 12 stocks with fixed inverse-volatility weights learned from the training window."""
    return _scaled(history, "iv", 1.0)


def decide_ew_brake20(history):
    """Equal weight across all stocks with a 20-day market volatility brake."""
    return _scaled(history, "ew", _vol_level(history, 20))


def decide_iv_brake10(history):
    """Inverse-volatility weights with a 10-day market volatility brake."""
    return _scaled(history, "iv", _vol_level(history, 10))


def decide_iv_brake10_strict(history):
    """Inverse-volatility weights with strict 10-day market volatility brake (q90/q95)."""
    return _scaled(history, "iv", _vol_level_strict(history, 10))


def decide_iv_brake20(history):
    """Inverse-volatility weights with a 20-day market volatility brake."""
    return _scaled(history, "iv", _vol_level(history, 20))


def decide_iv_brake40(history):
    """Inverse-volatility weights with a 40-day market volatility brake."""
    return _scaled(history, "iv", _vol_level(history, 40))


def decide_iv_brake60(history):
    """Inverse-volatility weights with a 60-day market volatility brake."""
    return _scaled(history, "iv", _vol_level(history, 60))


def decide_iv_trend(history):
    """Inverse-volatility weights cut in half when market is below 200-day trend."""
    return _scaled(history, "iv", _trend_level(history))


def decide_iv_combo(history):
    """Inverse-volatility weights combining 20-day volatility brake and 200-day trend filter."""
    return _scaled(history, "iv", min(_vol_level(history, 20), _trend_level(history)))


def decide_ap_cost_opt(history):
    """Apoorva Model 1: Multi-horizon momentum with MA trend filter, inverse-volatility sizing, breadth scaling & EMA smoothing."""
    close = history["close"]
    if len(close) <= 51:
        MODEL["ap_prev_w"] = None

    ma_days = MODEL.get("ap_m1_ma_days", 40)
    alpha = MODEL.get("ap_m1_alpha", 0.10)

    req_len = 205
    if len(close) < req_len:
        return {}

    curr = close.iloc[-1]
    ma = close.iloc[-ma_days:].mean()
    trend_ratio = (curr / ma - 1.0)
    trend_factor = ((trend_ratio + 0.015) / 0.03).clip(lower=0.0, upper=1.0)
    breadth = float((curr > ma).mean())

    m63 = curr / close.iloc[-63] - 1.0
    m126 = curr / close.iloc[-126] - 1.0
    m200 = curr / close.iloc[-200] - 1.0
    mom = 0.15 * m63 + 0.40 * m126 + 0.45 * m200

    pct = close.pct_change()
    vol = pct.iloc[-42:].std().replace(0, np.nan)
    risk_adj = (mom / vol).clip(lower=0.0)

    score = trend_factor * risk_adj / vol
    s_sum = float(score.sum())

    if s_sum <= 0 or np.isnan(s_sum):
        inv_vol = 1.0 / vol
        target_w = (inv_vol / inv_vol.sum()) * 0.35
    else:
        norm_w = score / s_sum
        gross = 0.35 + (0.95 - 0.35) * max(0.0, min(1.0, (breadth - 0.2) / 0.5))
        target_w = norm_w * gross

    target_w = target_w.fillna(0.0)
    prev_w = MODEL.get("ap_prev_w", None)
    if prev_w is None or not isinstance(prev_w, pd.Series):
        w = target_w
    else:
        w = (1.0 - alpha) * prev_w + alpha * target_w

    MODEL["ap_prev_w"] = w
    return {s: float(v) for s, v in w.fillna(0.0).items() if np.isfinite(v) and v >= 0}


def decide_ap_adaptive(history):
    """Apoorva Model 2: Inverse-volatility portfolio with smooth market trend & volatility targeting, smoothed over K days."""
    close = history["close"]
    if len(close) < 30:
        return {s: 0.5 / close.shape[1] for s in close.columns}
    p = MODEL.get("ap_params", AP_DEFAULT)
    return _ap_row_to_dict(_ap_alpha_weights(close, p).iloc[-1])


FINAL_IDEA = decide_iv_brake10_strict
_ERRORS = {"count": 0}


def decide_final(history):
    try:
        return FINAL_IDEA(history)
    except Exception:
        _ERRORS["count"] += 1
        try:
            return _base_weights(list(history["close"].columns), "iv")
        except Exception:
            return {}


decide = decide_final      # <- the agent we will run on the secret period


if __name__ == "__main__":
    # python strategy.py -> trains on TRAIN_WINDOW and prints the performance metrics.
    # (Only runs when you start this file yourself; our evaluation just imports it.)
    import sys
    from pathlib import Path
    import engine

    data = engine.load(Path(__file__).resolve().parent / "data" / "train.csv")
    window = engine.train_agent(sys.modules[__name__], data)
    log = engine.backtest(decide, data, verbose=False)["log"]
    inside = (log.index >= window[0]) & (log.index <= window[1])
    print(f"{TEAM_NAME}: trained on days {window[0]}-{window[1]}")
    print("\nInside the training window:")
    engine.print_metrics(engine.metrics(log[inside]))
    print("Outside the training window (validation):")
    engine.print_metrics(engine.metrics(log[~inside]))
    print("decide errors caught:", _ERRORS["count"])
