# Alpha Arena Strategy Submission

## 1. Team and AI Disclosure
- **Team Name:** We Need Baddies
- **AI Disclosure:** Code written with help from Gemini Flash (Antigravity). Problem analysis and strategy discussion with Claude.

---

## 2. Our Agent in One Sentence
Holds all 12 stocks with inverse-volatility weights learned from the training window, and cuts total exposure to 70% or 40% when the market's 10-day volatility rises above the 90th or 95th percentile of the training window.

---

## 3. Ideas We Tried

*Note: This is a forward test: trained on days 0–2519, measured on days 2520–3707 (with 0.3% trading cost).*

| Strategy | total_return_% | sharpe | calmar | max_drawdown_% | costs_paid_% | avg_exposure_% | generalization |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `decide_iv_brake10` | 80.446 | 0.976 | 0.630 | -21.176 | 3.600 | 95.530 | -0.581 |
| `decide_final` *(submitted)* | 84.715 | 0.966 | 0.568 | -24.456 | 1.080 | 98.030 | -0.560 |
| `decide_iv_brake10_strict` | 84.715 | 0.966 | 0.568 | -24.456 | 1.080 | 98.030 | -0.560 |
| `decide_iv_brake20` | 69.410 | 0.870 | 0.484 | -24.456 | 1.800 | 95.758 | -0.792 |
| `decide_ew_brake20` | 69.023 | 0.850 | 0.477 | -24.666 | 1.800 | 95.758 | -0.781 |
| `decide_iv_trend` | 61.343 | 0.848 | 0.487 | -21.950 | 4.650 | 89.857 | -0.692 |
| `decide_iv_combo` | 57.790 | 0.839 | 0.493 | -20.621 | 4.830 | 88.813 | -0.756 |
| `decide_iv_brake40` | 67.436 | 0.834 | 0.435 | -26.550 | 1.080 | 96.162 | -0.741 |
| `decide_inv_vol` | 83.985 | 0.821 | 0.407 | -33.926 | 0.000 | 100.000 | -0.822 |
| `BENCHMARK equal weight` | 82.805 | 0.801 | 0.397 | -34.387 | 0.000 | 100.000 | -0.818 |
| `decide_iv_brake60` | 62.522 | 0.784 | 0.399 | -27.198 | 0.540 | 95.833 | -0.808 |
| `decide_trend_filter` | 13.623 | 0.242 | 0.085 | -32.453 | 41.135 | 99.916 | -0.889 |
| `decide_momentum` | -64.199 | -0.894 | -0.270 | -72.421 | 137.400 | 100.000 | -0.615 |

Brake speed test (trained 0-2519, tested 2520-3707): Sharpe 0.98 (10-day), 0.87 (20-day), 0.83 (40-day), 0.78 (60-day) vs 0.80 for equal weight. Faster brakes worked better.

We first used the 75th/90th percentiles. With thresholds learned from the recent decade, that brake was on for about a quarter of days, so we raised them to the 90th/95th to keep it for genuine volatility spikes.

---

## 4. Training Window: (1188, 3707), the Most Recent 10 Years
We chose `TRAIN_WINDOW = (1188, 3707)` for three key reasons:
1. The secret period starts right after day 3707, so the most recent decade is closest to it.
2. The market changed: the equal-weight benchmark had a Sharpe of 1.62 inside days 0–2519 but only 0.80 on days 2520–3707. Weights and volatility thresholds learned from the calmer early years would be miscalibrated for the later market.
3. In `windows.csv` each row is validated on different days. The (1188, 3707) row is validation on days 51–1187, an earlier rising market (Sharpe 2.04, total return 331.84%), so its high validation numbers are not presented as expected performance. The real evidence is the forward test (trained 0–2519, tested 2520–3707).

---

## 5. Features Used
- Rolling 10-day market volatility (final agent); strict version (q90/q95) and 20, 40, and 60-day versions were tested as alternatives.
- Daily close-to-close returns of each stock.
- Each stock's volatility over the training window (to establish inverse-volatility base allocation).
- The equal-weight market's daily return.
- The 200-day trend of equal-weight market returns.

---

## 6. Risk Control
- Exposure never below 40% and never above 100%.
- No individual stock exceeds 15% (weights capped and re-normalized).
- No shorting (weights >= 0) and no leverage (sum of weights <= 1.0).
- Strict volatility brake steps down total exposure to 70% or 40% whenever turbulent regimes appear.
- Fallback wrapper catches any exception, returning base inverse-volatility weights or cash so the agent never crashes.

---

## 7. Assumptions
- 0.3% cost per change in weights (as announced by the organisers).
- Trades happen at the close.
- Cash earns 0.
- High-volatility periods tend to persist for a while (volatility clustering).

---

## 8. How to Run
```bash
python strategy.py
python compare.py
python compare.py windows
python visualization.py
python checks.py
```
