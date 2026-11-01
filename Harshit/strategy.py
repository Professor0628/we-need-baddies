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
