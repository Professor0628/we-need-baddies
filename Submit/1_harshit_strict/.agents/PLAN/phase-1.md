# Phase 1: Step 1 — Extend `train(data)` in `strategy.py`

## Instructions:
Keep all existing code in `train()`. Add the following at the end of `train()`:

### 1A. Fixed per-stock weights (inverse volatility)
1. `rets = close.pct_change().iloc[1:]`
2. `vol = rets.std()` -> one number per stock (daily standard deviation).
3. If any value of `vol` is NaN or <= 0: set every stock's weight to `1/12` and skip to item 7.
4. `inv = 1 / vol`
5. `w = inv / inv.sum()`
6. Cap any single stock at 15%: `w = w.clip(upper=0.15)`, then normalise again: `w = w / w.sum()`
7. Save as a plain dict of Python floats: `MODEL["iv_weights"] = {stock: float(weight) for each stock}`

### 1B. Market volatility thresholds
1. `idx = rets.mean(axis=1)` -> equal-weight market's daily return.
2. For `L` in `[20, 60]`:
   - `roll = idx.rolling(L).std().dropna()`
   - If `roll` is not empty:
     - `MODEL[f"vol_q75_{L}"] = float(roll.quantile(0.75))`
     - `MODEL[f"vol_q90_{L}"] = float(roll.quantile(0.90))`

## Command to run:
`python strategy.py` inside `Harshit`.
Must finish without errors.

