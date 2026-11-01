# Phase 2: Steps 2 & 3 Implementation Log

- Added helper functions above `decide_` functions:
  - `_base_weights(stocks, kind)`: supports inverse-volatility (`"iv"`) and equal-weights (`"ew"`).
  - `_market_returns(history)`: calculates equal-weighted market returns across all stocks.
  - `_vol_level(history, L)`: multi-tier volatility brake returning 1.0, 0.7, or 0.4.
  - `_trend_level(history)`: 200-day trend filter returning 1.0 or 0.5.
  - `_scaled(history, kind, exposure)`: scales base weights by exact exposure constants.
- Implemented 6 new trading ideas:
  - `decide_inv_vol`
  - `decide_ew_brake20`
  - `decide_iv_brake20`
  - `decide_iv_brake60`
  - `decide_iv_trend`
  - `decide_iv_combo`
- Preserved existing ideas `decide_momentum` and `decide_trend_filter`.

