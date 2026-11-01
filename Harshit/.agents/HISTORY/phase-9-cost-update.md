# Cost Update to 0.3% & Final Strict Strategy Log

- Updated `COST_PER_TRADE = 0.003` in `engine.py` (root) and `Harshit/engine.py`.
- Selected `FINAL_IDEA = decide_iv_brake10_strict`.
- Reran forward test on `TRAIN_WINDOW = (0, 2519)` with 0.3% cost -> saved `comparison_forward_0_2519.csv`.
- Set `TRAIN_WINDOW = (1188, 3707)`.
- Reran pipeline in order:
  - `python strategy.py`: `decide errors caught: 0`.
  - `python compare.py windows`: saved `windows.csv`.
  - `python compare.py`: saved `comparison.csv`.
  - `python checks.py`: `PASS` (2.58s, 0 errors, 91.32% avg exposure).
  - `python visualization.py`: regenerated `portfolio_growth.png` and `forward_test.png`.
- Updated `Harshit/README.md` with:
  - New 0.3% cost statistics.
  - New brake speed-test observations.
  - Updated one-sentence description and 90th/95th threshold rationale.
  - Correct validation label for days 51-1187.

