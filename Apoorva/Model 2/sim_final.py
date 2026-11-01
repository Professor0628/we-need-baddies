"""Rolling train-on-window / trade-the-rest sim at the announced 0.3% cost. Compares final agent,
previous cost-robust agent (v2) and equal weight. Train sees ONLY its window; trading blocks start fresh."""
import numpy as np, pandas as pd, engine, importlib
data = engine.load("data/train.csv"); last = int(data["close"].index[-1])
WINDOWS = {"years 1-10": (0, 2519), "years 2-11": (252, 2771), "years 3-12": (504, 3023),
           "years 4-13": (756, 3275), "years 5-14": (1008, 3527), "years 6-15": (1260, last)}
engine.COST_PER_TRADE = 0.003
def run(decide, a, b):
    sl = lambda hi: {f: df.loc[:hi] for f, df in data.items()}
    fit = engine.backtest(decide, sl(b + 1), start_day=max(a, 50), on_error="hold", verbose=False)["log"]
    blocks = []
    if a > 51: blocks.append(engine.backtest(decide, sl(a), start_day=50, on_error="hold", verbose=False)["log"])
    if b < last - 1: blocks.append(engine.backtest(decide, data, start_day=b + 1, on_error="hold", verbose=False)["log"])
    test = pd.concat(blocks); m, mf = engine.metrics(test), engine.metrics(fit)
    return {"test_days": len(test), "net_return_%": m["total_return_%"], "sharpe": m["sharpe"], "calmar": m["calmar"],
            "max_drawdown_%": m["max_drawdown_%"], "fit_sharpe": mf["sharpe"], "generalization": m["sharpe"] - mf["sharpe"],
            "trading_accuracy_%": m["trading_accuracy_%"], "costs_paid_%": m["costs_paid_%"], "avg_exposure_%": m["avg_exposure_%"], "errors": int(test["error"].sum())}
rows = []
for name, modname in (("FINAL (0.3% optimised)", "strategy"),):
    mod = importlib.import_module(modname)
    for wname, (a, b) in WINDOWS.items():
        mod.TRAIN_WINDOW = (a, b); engine.train_agent(mod, data)
        rows.append({"agent": name, "train_window": wname, **run(mod.decide, a, b)})
        if name.startswith("FINAL"): rows.append({"agent": "equal weight", "train_window": wname, **run(engine.equal_weight, a, b)})
        print("done", name, wname, flush=True)
df = pd.DataFrame(rows)
avg = df.groupby("agent", sort=False).mean(numeric_only=True).reset_index(); avg["train_window"] = "AVERAGE"
df = pd.concat([df, avg], ignore_index=True).round(3); df.to_csv("sim_final_0p3.csv", index=False)
pd.set_option("display.width", 250)
print(df.drop(columns=["test_days", "errors"]).to_string(index=False))
