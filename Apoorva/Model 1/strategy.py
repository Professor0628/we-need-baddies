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

TEAM_NAME = "we-need-baddies"
TRAIN_WINDOW = (1188, 3707)   # (first_day, last_day) - most recent 10 years (2520 days), directly preceding secret period
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
    Finds optimal moving-average trend filter and momentum lookback length inside the training window
    specifically accounting for the official 0.3% transaction fee."""
    close = data["close"]
    ret = close.pct_change()
    fwd_ret = ret.shift(-1)
    
    best_score = -np.inf
    best_ma = 40
    
    m63 = close / close.shift(63) - 1.0
    m126 = close / close.shift(126) - 1.0
    m200 = close / close.shift(200) - 1.0
    mom = 0.15 * m63 + 0.40 * m126 + 0.45 * m200
    vol = ret.rolling(42).std().replace(0, np.nan)
    risk_adj = (mom / vol).clip(lower=0.0)
    
    for ma in [35, 40, 45, 50]:
        c_ma = close.rolling(ma).mean()
        trend_ratio = (close / c_ma - 1.0)
        trend_factor = ((trend_ratio + 0.015) / 0.03).clip(lower=0.0, upper=1.0)
        breadth = (close > c_ma).mean(axis=1)
        
        score = trend_factor * risk_adj / vol
        score_sum = score.sum(axis=1).replace(0, np.nan)
        norm_w = score.div(score_sum, axis=0).fillna(0.0)
        
        scale = 0.35 + (0.95 - 0.35) * ((breadth - 0.2) / 0.5).clip(lower=0.0, upper=1.0)
        target_w = norm_w.mul(scale, axis=0)
        
        # Simulate EMA turnover smoothing with alpha = 0.10
        smoothed_w = target_w.ewm(alpha=0.10, adjust=False).mean()
        delta_w = smoothed_w.diff().abs().sum(axis=1).fillna(0.0)
        
        # Official 0.3% trading cost
        cost = 0.003 * delta_w
        daily_net = (smoothed_w * fwd_ret).sum(axis=1) - cost
        std = daily_net.std()
        sh = float(daily_net.mean() / std * np.sqrt(252)) if std > 0 else 0.0
        
        eq = (1 + daily_net).cumprod()
        dd = float((eq / eq.cummax().clip(lower=1.0) - 1.0).min())
        calmar = ((1 + eq.iloc[-1]) ** (252 / len(daily_net)) - 1) / -dd if dd < 0 else 0.0
        score_metric = sh + 0.3 * calmar
        if score_metric > best_score:
            best_score = score_metric
            best_ma = ma
            
    MODEL["ma_days"] = best_ma
    MODEL["alpha"] = 0.10
    MODEL["prev_w"] = None


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


def decide_adaptive_momentum(history):
    """Adaptive Alpha (Base 0.1% Model): Multi-horizon trend-filtered, inverse-volatility weighted momentum
    with market-breadth exposure scaling and downside crash protection."""
    close = history["close"]
    ma_days = MODEL.get("ma_days", 50)
    mom_days = 150
    vol_days = 21
    req_len = max(ma_days, mom_days, vol_days) + 2
    if len(close) < req_len:
        return {}
    
    curr_price = close.iloc[-1]
    ma = close.iloc[-ma_days:].mean()
    above_ma = curr_price > ma
    breadth = float(above_ma.mean())
    
    # Scale-invariant features: multi-horizon momentum and 21-day return volatility
    m63 = curr_price / close.iloc[-63] - 1.0
    m126 = curr_price / close.iloc[-126] - 1.0
    m_long = curr_price / close.iloc[-mom_days] - 1.0
    mom_ret = 0.2 * m63 + 0.4 * m126 + 0.4 * m_long
    
    pct = close.pct_change()
    vol = pct.iloc[-vol_days:].std().replace(0, np.nan)
    risk_adj_mom = (mom_ret / vol).clip(lower=0.0)
    
    # Continuous inverse-volatility weighting on trending positive-momentum assets
    raw_score = above_ma.astype(float) * risk_adj_mom / vol
    score_sum = float(raw_score.sum())
    
    if score_sum <= 0 or np.isnan(score_sum):
        # Crash/defensive regime: allocate minimum exposure across lowest-volatility stocks
        inv_vol = 1.0 / vol
        low_vol_w = (inv_vol / inv_vol.sum()) * 0.35
        return low_vol_w.to_dict()
    
    norm_w = raw_score / score_sum
    # Dynamic exposure based on market breadth (ranges from 0.35 in deep bear to 0.95 in bull)
    gross_exp = 0.35 + (0.95 - 0.35) * max(0.0, min(1.0, (breadth - 0.2) / 0.5))
    final_w = norm_w * gross_exp
    return final_w.to_dict()


def decide_cost_optimized_03(history):
    """Cost-Optimized Alpha (0.3% Fee Champion):
    Slow-moving multi-horizon momentum, continuous non-binary trend factor,
    inverse-volatility position sizing, dynamic breadth scaling, and EMA turnover damping."""
    close = history["close"]
    if len(close) <= 51:
        MODEL["prev_w"] = None
        
    ma_days = MODEL.get("ma_days", 40)
    alpha = MODEL.get("alpha", 0.10)
    
    req_len = 205
    if len(close) < req_len:
        return {}
        
    curr = close.iloc[-1]
    ma = close.iloc[-ma_days:].mean()
    trend_ratio = (curr / ma - 1.0)
    trend_factor = ((trend_ratio + 0.015) / 0.03).clip(lower=0.0, upper=1.0)
    breadth = float((curr > ma).mean())
    
    # Scale-invariant multi-horizon momentum: 63d, 126d, 200d
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
        
    prev_w = MODEL.get("prev_w", None)
    if prev_w is None or not isinstance(prev_w, pd.Series):
        w = target_w
    else:
        w = (1.0 - alpha) * prev_w + alpha * target_w
        
    MODEL["prev_w"] = w
    return w.to_dict()


# Ideas to try next: mean reversion, RSI, volatility filters, stop-losses,
# position sizing by risk, ML models fitted in train(), ...


decide = decide_cost_optimized_03      # <- the agent we will run on the secret period


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
