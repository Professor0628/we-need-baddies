# Phase 4 Extra Check: Volatility Lookback Variations

- Extended `train()` Step 1B thresholds for lookbacks: `[10, 20, 40, 60]`.
- Added strategies:
  - `decide_iv_brake10`: 10-day volatility brake.
  - `decide_iv_brake40`: 40-day volatility brake.
- Retained `TRAIN_WINDOW = (0, 2519)`.
- Reran `compare.py` and overwrote `comparison_forward_0_2519.csv`.
- `decide_iv_brake10` achieved the highest validation Sharpe (1.01), highest total return (84.82%), and lowest drawdown among pure brakes (-21.08%).

