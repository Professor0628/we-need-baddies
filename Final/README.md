# Alpha Arena Strategy Submission

## 1. Team and AI Disclosure
- **Team Name:** We Need Baddies
- **AI Disclosure:** Code written with help from Gemini Flash (Antigravity) and Claude. Apoorva's bots were also developed with AI assistance.

---

## 2. Our Three Strategies

We package three complementary strategies developed by our team:

1. **Harshit: Inverse-Volatility + Volatility Brake (`decide_final` / `decide_iv_brake10_strict`)**:
   Holds all 12 stocks with inverse-volatility weights learned from the training window, and cuts total exposure to 70% or 40% when the market's 10-day volatility rises above the 90th or 95th percentile of the training window.
2. **Apoorva: Adaptive Alpha (`decide_ap_adaptive`)**:
   Allocates capital across stocks using an ensemble of trend, momentum, and mean-reversion signals weighted by inverse volatility, scaling overall equity exposure via a multi-window volatility and trend regime filter.
3. **Apoorva: Cost-Optimized Momentum (`decide_ap_cost_opt`)**:
   Combines long-term momentum with inverse-volatility sizing while penalizing turnover through a proportional cost penalty, adjusting market exposure dynamically based on 20-day trend and volatility regimes.

### Forward Test Performance (Trained: Days 0–2519, Tested: Days 2520–3707 at 0.3% Cost)

| Strategy | total_return_% | sharpe | sortino | calmar | max_drawdown_% | trading_accuracy_% | costs_paid_% | avg_exposure_% | fit_sharpe | generalization |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `decide_final` (`decide_iv_brake10_strict`) | 84.715 | 0.966 | 1.383 | 0.568 | -24.456 | 49.972 | 1.080 | 98.030 | 1.526 | -0.560 |
| `decide_ap_adaptive` | 79.595 | 0.955 | 1.354 | 0.494 | -26.782 | 49.972 | 1.793 | 87.932 | 1.617 | -0.662 |
| `decide_ap_cost_opt` | 41.420 | 0.677 | 0.995 | 0.440 | -17.342 | 49.972 | 11.308 | 76.457 | 1.400 | -0.723 |
| `decide_inv_vol` | 83.985 | 0.821 | 1.150 | 0.407 | -33.926 | 49.972 | 0.000 | 100.000 | 1.643 | -0.822 |
| `BENCHMARK equal weight` | 82.805 | 0.801 | 1.123 | 0.397 | -34.387 | 49.972 | 0.000 | 100.000 | 1.619 | -0.818 |

---

## 3. Ideas We Tried

*Full forward test comparison table (trained on days 0–2519, tested on days 2520–3707 with 0.3% trading cost):*

| Strategy | total_return_% | sharpe | sortino | calmar | max_drawdown_% | costs_paid_% | avg_exposure_% | fit_sharpe | generalization |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `decide_iv_brake10` | 80.446 | 0.976 | 1.422 | 0.630 | -21.176 | 3.600 | 95.530 | 1.557 | -0.581 |
| `decide_final` *(submitted)* | 84.715 | 0.966 | 1.383 | 0.568 | -24.456 | 1.080 | 98.030 | 1.526 | -0.560 |
| `decide_iv_brake10_strict` | 84.715 | 0.966 | 1.383 | 0.568 | -24.456 | 1.080 | 98.030 | 1.526 | -0.560 |
| `decide_ap_adaptive` | 79.595 | 0.955 | 1.354 | 0.494 | -26.782 | 1.793 | 87.932 | 1.617 | -0.662 |
| `decide_iv_brake20` | 69.410 | 0.870 | 1.242 | 0.484 | -24.456 | 1.800 | 95.758 | 1.662 | -0.792 |
| `decide_ew_brake20` | 69.023 | 0.850 | 1.217 | 0.477 | -24.666 | 1.800 | 95.758 | 1.631 | -0.781 |
| `decide_iv_trend` | 61.343 | 0.848 | 1.209 | 0.487 | -21.950 | 4.650 | 89.857 | 1.540 | -0.692 |
| `decide_iv_combo` | 57.790 | 0.839 | 1.198 | 0.493 | -20.621 | 4.830 | 88.813 | 1.595 | -0.756 |
| `decide_iv_brake40` | 67.436 | 0.834 | 1.175 | 0.435 | -26.550 | 1.080 | 96.162 | 1.575 | -0.741 |
| `decide_inv_vol` | 83.985 | 0.821 | 1.150 | 0.407 | -33.926 | 0.000 | 100.000 | 1.643 | -0.822 |
| `BENCHMARK equal weight` | 82.805 | 0.801 | 1.123 | 0.397 | -34.387 | 0.000 | 100.000 | 1.619 | -0.818 |
| `decide_iv_brake60` | 62.522 | 0.784 | 1.099 | 0.399 | -27.198 | 0.540 | 95.833 | 1.592 | -0.808 |
| `decide_ap_cost_opt` | 41.420 | 0.677 | 0.995 | 0.440 | -17.342 | 11.308 | 76.457 | 1.400 | -0.723 |
| `decide_trend_filter` | 13.623 | 0.242 | 0.343 | 0.085 | -32.453 | 41.135 | 99.916 | 1.131 | -0.889 |
| `decide_momentum` | -64.199 | -0.894 | -1.221 | -0.270 | -72.421 | 137.400 | 100.000 | -0.279 | -0.615 |

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
- Momentum and multi-window trend signals across assets.

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
