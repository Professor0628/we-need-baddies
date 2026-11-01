import sys
import time
from pathlib import Path
import numpy as np
import pandas as pd

import engine
import strategy

def check_weights(w, context=""):
    if not isinstance(w, dict):
        print(f"FAIL: {context} result is not a dict: {type(w)}")
        return False
    total = 0.0
    for k, v in w.items():
        if not np.isfinite(v):
            print(f"FAIL: {context} stock {k} has non-finite weight: {v}")
            return False
        if v < 0:
            print(f"FAIL: {context} stock {k} has negative weight: {v}")
            return False
        total += float(v)
    if total > 1.0 + 1e-9:
        print(f"FAIL: {context} sum of weights exceeds 1.0: {total}")
        return False
    return True

def main():
    if len(sys.argv) > 1:
        fn_name = sys.argv[1]
        target_fn = getattr(strategy, fn_name)
    else:
        fn_name = strategy.FINAL_IDEA.__name__
        target_fn = strategy.FINAL_IDEA

    print(f"Testing function: {fn_name}")
    all_passed = True

    # 1. Load data and train agent
    data = engine.load("data/train.csv")
    engine.train_agent(strategy, data)
    print("Agent trained on TRAIN_WINDOW:", strategy.TRAIN_WINDOW)

    # 2. Short-history test
    print("\n--- Short-history test ---")
    short_ns = [1, 2, 5, 20, 21, 59, 60, 61, 199, 200, 201, 252]
    for n in short_ns:
        # first n days
        h_first = {f: data[f].iloc[:n] for f in engine.FIELDS}
        w_first = target_fn(h_first)
        ok_first = check_weights(w_first, f"first {n}")

        # last n days
        h_last = {f: data[f].iloc[-n:] for f in engine.FIELDS}
        w_last = target_fn(h_last)
        ok_last = check_weights(w_last, f"last {n}")

        if ok_first and ok_last:
            print(f"OK n={n}")
        else:
            all_passed = False
            print(f"FAIL n={n}")

    # 3. NaN test
    print("\n--- NaN test ---")
    nan_hist = {f: data[f].iloc[-252:].copy() for f in engine.FIELDS}
    nan_hist["close"].loc[nan_hist["close"].index[-3:], "STOCK_03"] = np.nan
    w_nan = target_fn(nan_hist)
    if check_weights(w_nan, "NaN test"):
        print("OK NaN test")
    else:
        all_passed = False
        print("FAIL NaN test")

    # 4. Full run
    print("\n--- Full run ---")
    t0 = time.time()
    result = engine.backtest(target_fn, data, on_error="raise", verbose=False)
    elapsed = time.time() - t0
    errors = result["errors"]
    avg_exp = result["metrics"]["avg_exposure_%"]
    costs = result["metrics"]["costs_paid_%"]
    weight_changes = int((result["weights"].diff().abs().sum(axis=1) > 0).sum())

    print(f"Seconds taken: {elapsed:.2f}")
    print(f"result['errors']: {errors}")
    print(f"result['metrics']['avg_exposure_%']: {avg_exp:.2f}%")
    print(f"result['metrics']['costs_paid_%']: {costs:.2f}%")
    print(f"Days weights changed: {weight_changes}")

    # 5. Verdict
    passed = (
        all_passed
        and elapsed < 90
        and errors == 0
        and avg_exp >= 40
    )
    print("\n" + ("=" * 20))
    if passed:
        print("PASS")
    else:
        print("FAIL")
    print("=" * 20)

if __name__ == "__main__":
    main()
