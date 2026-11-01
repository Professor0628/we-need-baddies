"""
visualization.py - cumulative portfolio growth of the submitted agent vs the equal-weight benchmark.
Shows the last 10 years of the data (days 1188-3707 = 2520 days), which contains the held-out
crash year (year 15). The shaded band is the part of this graph that the agent trained on.
Run:  python visualization.py   -> saves growth.png
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import engine
import strategy

HERE = Path(__file__).resolve().parent
engine.COST_PER_TRADE = 0.003          # announced trading cost: 0.3% of the amount traded
data = engine.load(HERE / "data" / "train.csv")
engine.train_agent(strategy, data)

mine = engine.backtest(strategy.decide, data, verbose=False)["log"]["return"]
bench = engine.backtest(engine.equal_weight, data, verbose=False)["log"]["return"]

last = mine.index[-1]
first = last - engine.MAX_TRAIN_DAYS + 1                      # at most 10 years on the graph
mine, bench = mine.loc[first:last], bench.loc[first:last]
curve = lambda r: engine.STARTING_CAPITAL * (1 + r).cumprod()

fig, ax = plt.subplots(figsize=(11, 5.5))
a, b = strategy.TRAIN_WINDOW
ax.axvspan(max(a, first), min(b, last), color="grey", alpha=0.12, label="agent's training window")
ax.plot(curve(bench), label="Equal-weight benchmark", color="#d9822b", lw=1.6)
ax.plot(curve(mine), label="Our agent (adaptive alpha)", color="#1f5fbf", lw=2)
ax.set_yscale("log")
ax.set_xlabel("Day")
ax.set_ylabel("Portfolio value (₹, log scale)")
ax.set_title(f"Portfolio growth from ₹1,00,000 - days {first} to {last}")
ax.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"₹{v:,.0f}"))
ax.grid(alpha=0.3)
ax.legend()
fig.tight_layout()
fig.savefig(HERE / "growth.png", dpi=200)
print("saved growth.png | final value: agent ₹{:,.0f} vs benchmark ₹{:,.0f}".format(curve(mine).iloc[-1], curve(bench).iloc[-1]))
