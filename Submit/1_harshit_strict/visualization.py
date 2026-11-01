import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker
import pandas as pd

import engine
import strategy

PLOT_STRATEGIES = ["decide_final"]

STRATEGY_LABELS = {
    "decide_final": "Harshit: inverse-vol + volatility brake",
    "decide_ap_cost_opt": "Apoorva: cost-opt momentum",
    "decide_ap_adaptive": "Apoorva: adaptive alpha",
}

COLORS = {
    "decide_final": "#1f77b4",
    "decide_ap_cost_opt": "#2ca02c",
    "decide_ap_adaptive": "#9467bd",
}


def to_portfolio_series(r):
    val = engine.STARTING_CAPITAL * (1 + r).cumprod()
    first_day = val.index[0]
    prepend = pd.Series([float(engine.STARTING_CAPITAL)], index=[first_day - 1])
    return pd.concat([prepend, val])


def main():
    # 2. Load data and train agent on real TRAIN_WINDOW
    data = engine.load("data/train.csv")
    engine.train_agent(strategy, data)

    # Benchmark run
    bench_ret = engine.backtest(engine.equal_weight, data, verbose=False)["log"]["return"]

    # --- Figure 1: Portfolio growth ---
    start, end = strategy.TRAIN_WINDOW
    fig, ax = plt.subplots(figsize=(11, 5.5))

    for name in PLOT_STRATEGIES:
        fn = getattr(strategy, name)
        ret = engine.backtest(fn, data, verbose=False)["log"]["return"]
        sliced = ret[(ret.index >= start) & (ret.index <= end)]
        val = to_portfolio_series(sliced)
        label = STRATEGY_LABELS.get(name, name)
        color = COLORS.get(name, None)
        ax.plot(val.index, val.values, label=label, color=color, linewidth=1.8)

    bench_win = bench_ret[(bench_ret.index >= start) & (bench_ret.index <= end)]
    bench_val = to_portfolio_series(bench_win)
    ax.plot(bench_val.index, bench_val.values, label="Equal-weight benchmark", color="grey", linestyle="--", linewidth=1.5)

    ax.set_xlabel("Day")
    ax.set_ylabel("Portfolio value (Rs)")
    ax.set_title(f"Portfolio growth, days {start}-{end} (start Rs 1,00,000)")
    ax.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda x, p: f"{x:,.0f}"))
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.savefig("portfolio_growth.png", dpi=150, bbox_inches="tight")
    print("Saved portfolio_growth.png")

    # --- Figure 2: Forward test ---
    engine.train_agent(strategy, data, window=(0, 2519))
    bench_fwd_ret = engine.backtest(engine.equal_weight, data, verbose=False)["log"]["return"]
    bench_fwd_slice = bench_fwd_ret[(bench_fwd_ret.index >= 2520) & (bench_fwd_ret.index <= 3707)]
    bench_fwd_val = to_portfolio_series(bench_fwd_slice)

    fig2, ax2 = plt.subplots(figsize=(11, 5.5))
    for name in PLOT_STRATEGIES:
        fn = getattr(strategy, name)
        ret = engine.backtest(fn, data, verbose=False)["log"]["return"]
        sliced = ret[(ret.index >= 2520) & (ret.index <= 3707)]
        val = to_portfolio_series(sliced)
        label = STRATEGY_LABELS.get(name, name)
        color = COLORS.get(name, None)
        ax2.plot(val.index, val.values, label=label, color=color, linewidth=1.8)

    ax2.plot(bench_fwd_val.index, bench_fwd_val.values, label="Equal-weight benchmark", color="grey", linestyle="--", linewidth=1.5)
    ax2.set_xlabel("Day")
    ax2.set_ylabel("Portfolio value (Rs)")
    ax2.set_title("Forward test: trained on days 0-2519, tested on days 2520-3707")
    ax2.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda x, p: f"{x:,.0f}"))
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    fig2.savefig("forward_test.png", dpi=150, bbox_inches="tight")
    print("Saved forward_test.png")

    # Restore real window
    engine.train_agent(strategy, data)
    print("Restored TRAIN_WINDOW:", strategy.TRAIN_WINDOW)

    plt.show()


if __name__ == "__main__":
    main()
