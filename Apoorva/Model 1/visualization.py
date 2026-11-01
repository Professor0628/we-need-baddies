"""
visualization.py - Cumulative portfolio growth graph for Alpha Arena submission.
"""
from pathlib import Path
import matplotlib.pyplot as plt

import engine
import strategy

HERE = Path(__file__).resolve().parent


def main():
    data = engine.load(HERE / "data" / "train.csv")
    window = engine.train_agent(strategy, data)
    
    # Backtest strategy and equal-weight benchmark
    res_strat = engine.backtest(strategy.decide, data, verbose=False)
    res_bench = engine.backtest(engine.equal_weight, data, verbose=False)
    
    # Slice the validation days outside the 10-year training window
    inside = (res_strat["log"].index >= window[0]) & (res_strat["log"].index <= window[1])
    val_idx = res_strat["log"].index[~inside]
    
    daily_strat = res_strat["log"].loc[val_idx, "return"]
    daily_bench = res_bench["log"].loc[val_idx, "return"]
    
    equity_strat = engine.STARTING_CAPITAL * (1.0 + daily_strat).cumprod()
    equity_bench = engine.STARTING_CAPITAL * (1.0 + daily_bench).cumprod()
    
    plt.figure(figsize=(10, 6))
    plt.plot(equity_strat.index, equity_strat.values, label=f"{strategy.TEAM_NAME} (₹{equity_strat.iloc[-1]:,.0f})", color="#008080", linewidth=2.2)
    plt.plot(equity_bench.index, equity_bench.values, label=f"Equal Weight Benchmark (₹{equity_bench.iloc[-1]:,.0f})", color="#2E86C1", linewidth=1.8, linestyle="--")
    
    plt.title(f"Cumulative Portfolio Growth — Validation Period (Trained on Days {window[0]}-{window[1]})", fontsize=12, fontweight="bold")
    plt.xlabel("Day", fontsize=11, fontweight="bold")
    plt.ylabel("Portfolio Value (₹)", fontsize=11, fontweight="bold")
    plt.gca().yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f"₹{int(x):,}"))
    plt.legend(loc="upper left", frameon=True)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    out_file = HERE / "portfolio_growth.png"
    plt.savefig(out_file, dpi=200)
    print(f"Saved {out_file.name}")


if __name__ == "__main__":
    main()
