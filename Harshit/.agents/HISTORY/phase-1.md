# Phase 1: Step 1 Implementation Log

- Extended `train(data)` in `Harshit/strategy.py`:
  - Implemented 1A: Fixed per-stock inverse volatility weights saved to `MODEL["iv_weights"]`.
  - Implemented 1B: Rolling market return volatility thresholds saved to `MODEL["vol_q75_L"]` and `MODEL["vol_q90_L"]` for `L in [20, 60]`.
- Verified execution:
  - `python strategy.py` ran without errors.
  - Numbers are identical to baseline as expected because new keys are not yet consumed by decision functions.

