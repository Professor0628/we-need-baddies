# Phase 5–8 Implementation Log

## Step 7
- Configured `FINAL_IDEA = decide_iv_brake10`.
- Updated `TRAIN_WINDOW = (1188, 3707)`.
- Verified `python strategy.py`: `decide errors caught: 0`.
- Ran `python compare.py windows` -> generated `windows.csv`.
- Ran `python compare.py` -> generated `comparison.csv`.

## Step 8 (`checks.py`)
- Created `Harshit/checks.py` testing short history lengths, NaN handling on STOCK_03, execution runtime, error count, and weight constraints.
- Output: `PASS` in 2.64s with 0 errors and 84.15% average exposure.

## Step 9 (`visualization.py`)
- Implemented `Harshit/visualization.py`.
- Generated `portfolio_growth.png` for training window (1188-3707).
- Generated `forward_test.png` for forward test (train 0-2519, test 2520-3707).
- Both charts correctly rebase to Rs 1,00,000 on day 0.

## Step 10 & 11 (README & Checklist)
- Preserved original kit README as `Harshit/README_kit.md`.
- Wrote full team `Harshit/README.md` with all required sections, including the brake speed test observations and exact figures.
- Completed final checklist verification.

