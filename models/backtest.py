"""
Backtesting framework — full model comparison + benchmark normalization.

Strategies evaluated (all using 5-fold time-series OOS predictions):
  1. Baseline LR   — same-asset 3-feature logistic regression
  2. Extended LR   — full 12-feature cross-asset logistic regression
  3. XGBoost       — cross-asset XGBoost (threshold 0.55)
  4. Threshold grid — sweep threshold 0.50 → 0.70 on Extended LR signal
  5. TC sensitivity — Extended LR at TC = 0%, 0.1%, 0.5%

Benchmarks (via yfinance):
  - SPY buy-and-hold over the OOS trading period
  - BTC-USD buy-and-hold over the same period

All use binary PnL: +1 correct, −1 incorrect (before transaction costs).
$1 bet per trade on a $1,000 portfolio → 0.1% risk per contract.
Annualised Sharpe: 252 trading days × 26 fifteen-minute periods per day.

Loads:
    data/merged_contracts.csv
    data/lr_baseline_oos.csv        (from logistic_regression.py)
    data/lr_extended_oos.csv        (from logistic_regression.py)
    data/xgb_oos_predictions.csv    (from xgboost_model.py)
Outputs:
    figures/backtest_timeseries.png     — date-indexed cumulative returns, all strategies + benchmarks
    figures/backtest_roc_comparison.png — ROC curves for all 3 models on same axes
    figures/backtest_model_bars.png     — accuracy + AUC bar chart, all 3 models
    figures/backtest_sharpe_grid.png    — threshold sensitivity: Sharpe vs # trades
    figures/backtest_tc_sensitivity.png — transaction cost robustness
"""

import warnings
warnings.filterwarnings("ignore", category=UserWarning, message=".*non-interactive.*")

from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
FIGURES_DIR = ROOT / "figures"
FIGURES_DIR.mkdir(exist_ok=True)

PERIODS_PER_YEAR = 252 * 26   # 15-min contracts, ~26 per trading day
PORTFOLIO_VALUE  = 1_000.0    # starting portfolio ($)
BET_SIZE         = 1.0        # dollars risked per trade

plt.rcParams.update({
    "axes.spines.top":   False,
    "axes.spines.right": False,
    "axes.grid":         True,
    "grid.alpha":        0.3,
    "grid.linewidth":    0.5,
    "font.size":         11,
})


# ---------------------------------------------------------------------------
# Core backtest engine
# ---------------------------------------------------------------------------

def run_backtest(
    signal:    np.ndarray,
    target:    np.ndarray,
    threshold: float = 0.55,
    tc:        float = 0.0,
) -> dict:
    """
    Binary long/short strategy.

    signal   : predicted P(UP) from any model
    target   : true outcomes (1=UP, 0=DOWN)
    threshold: enter long if signal > threshold, short if signal < 1-threshold.
               Contracts in the dead-zone [1-threshold, threshold] are skipped.
    tc       : one-way transaction cost as fraction of BET_SIZE
    """
    pnl = []
    for sig, tgt in zip(signal, target):
        if sig > threshold:
            correct = int(tgt) == 1
        elif sig < 1 - threshold:
            correct = int(tgt) == 0
        else:
            continue
        pnl.append((BET_SIZE if correct else -BET_SIZE) - tc)

    if not pnl:
        return {"n_trades": 0, "hit_rate": np.nan, "total_pnl": np.nan,
                "sharpe": np.nan, "max_drawdown": np.nan,
                "pnl_series": np.array([]), "cum_pnl": np.array([])}

    pnl_arr  = np.array(pnl)
    cum      = np.cumsum(pnl_arr)
    peak     = np.maximum.accumulate(cum)
    mean_r   = pnl_arr.mean()
    std_r    = pnl_arr.std(ddof=1)
    sharpe   = mean_r / std_r * np.sqrt(PERIODS_PER_YEAR) if std_r > 0 else np.nan

    return {
        "n_trades":    len(pnl_arr),
        "hit_rate":    (pnl_arr > 0).mean(),
        "total_pnl":   pnl_arr.sum(),
        "sharpe":      sharpe,
        "max_drawdown": (cum - peak).min(),
        "pnl_series":  pnl_arr,
        "cum_pnl":     cum,
    }


# ---------------------------------------------------------------------------
# Load OOS predictions from all three models
# ---------------------------------------------------------------------------

def load_all_predictions() -> pd.DataFrame | None:
    """
    Merge OOS predictions from Baseline LR, Extended LR, and XGBoost on open_key.
    Only rows present in all three files are kept (inner join) so metrics are
    computed on the same set of contracts.
    Returns None if any required file is missing.
    """
    paths = {
        "baseline": DATA_DIR / "lr_baseline_oos.csv",
        "extended": DATA_DIR / "lr_extended_oos.csv",
        "xgb":      DATA_DIR / "xgb_oos_predictions.csv",
    }
    missing = [k for k, p in paths.items() if not p.exists()]
    if missing:
        print(f"  Missing OOS files: {missing}")
        print("  Run logistic_regression.py and xgboost_model.py first.")
        return None

    base = pd.read_csv(paths["baseline"], parse_dates=["open_key"])
    ext  = pd.read_csv(paths["extended"], parse_dates=["open_key"])
    xgb  = pd.read_csv(paths["xgb"],      parse_dates=["open_key"])

    # Rename before merge to avoid column conflicts
    base = base.rename(columns={"lr_proba": "baseline_proba", "target_btc": "target"})
    ext  = ext.rename( columns={"lr_proba": "extended_proba"})
    xgb  = xgb.rename( columns={"xgb_proba": "xgb_proba"})

    df = (
        base[["open_key", "target", "baseline_proba"]]
        .merge(ext[["open_key", "extended_proba"]],  on="open_key", how="inner")
        .merge(xgb[["open_key", "xgb_proba"]],       on="open_key", how="inner")
        .sort_values("open_key")
        .reset_index(drop=True)
    )
    print(f"  OOS contracts (all 3 models): {len(df):,}")
    print(f"  OOS period: {df['open_key'].min().date()} → {df['open_key'].max().date()}")
    return df


# ---------------------------------------------------------------------------
# Fetch benchmark data
# ---------------------------------------------------------------------------

def fetch_benchmarks(start: str, end: str) -> dict[str, pd.Series]:
    """
    Download SPY (S&P 500 ETF) and BTC-USD daily closing prices via yfinance.
    Returns daily close price series indexed by date.
    Falls back to empty dict on network error so the rest of the backtest runs.
    """
    try:
        import yfinance as yf
        spy = yf.download("SPY",     start=start, end=end, progress=False, auto_adjust=True)
        btc = yf.download("BTC-USD", start=start, end=end, progress=False, auto_adjust=True)

        def extract_close(raw):
            if raw.empty:
                return pd.Series(dtype=float)
            col = raw["Close"]
            # yfinance sometimes returns MultiIndex columns
            if isinstance(col, pd.DataFrame):
                col = col.iloc[:, 0]
            return col.dropna()

        return {"SPY": extract_close(spy), "BTC-USD": extract_close(btc)}
    except Exception as e:
        print(f"  yfinance error ({e}); benchmark comparison unavailable.")
        return {}


# ---------------------------------------------------------------------------
# Build date-indexed time series for each strategy
# ---------------------------------------------------------------------------

def build_strategy_timeseries(df: pd.DataFrame, threshold: float = 0.55) -> dict:
    """
    For each strategy: compute per-trade PnL, attach open_key timestamp,
    then compute cumulative portfolio value (starting at PORTFOLIO_VALUE).

    Returns a dict of {label: pd.Series(cumulative_value, index=open_key)}.
    """
    strategies = {
        "Baseline LR":  df["baseline_proba"].values,
        "Extended LR":  df["extended_proba"].values,
        "XGBoost":      df["xgb_proba"].values,
    }
    target = df["target"].values
    times  = df["open_key"].values

    series_dict = {}
    for label, signal in strategies.items():
        pnl_list  = []
        time_list = []
        for sig, tgt, t in zip(signal, target, times):
            if sig > threshold:
                correct = int(tgt) == 1
            elif sig < 1 - threshold:
                correct = int(tgt) == 0
            else:
                continue
            pnl_list.append(BET_SIZE if correct else -BET_SIZE)
            time_list.append(t)

        if not pnl_list:
            continue
        cum_pnl = np.cumsum(pnl_list)
        portfolio = PORTFOLIO_VALUE + cum_pnl   # dollar value of portfolio over time
        series_dict[label] = pd.Series(portfolio, index=pd.to_datetime(time_list))

    return series_dict


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------

def plot_timeseries(strategy_series: dict, benchmarks: dict,
                    oos_start: pd.Timestamp, oos_end: pd.Timestamp) -> None:
    """
    Cumulative portfolio value over calendar time, starting at $1,000.

    Trading strategies: $1 bet per trade, portfolio grows/shrinks accordingly.
    SPY / BTC-USD: $1,000 invested at first OOS date, held to end (buy-and-hold).
    This normalizes all curves to the same starting capital so they're directly
    comparable on the y-axis.
    """
    fig, ax = plt.subplots(figsize=(13, 6))

    colors = {
        "Baseline LR": "steelblue",
        "Extended LR": "darkorange",
        "XGBoost":     "tomato",
    }
    for label, series in strategy_series.items():
        ax.plot(series.index, series.values, color=colors.get(label, "gray"),
                linewidth=1.8, label=label, alpha=0.9)

    # Benchmark: normalize so buy-and-hold starts at PORTFOLIO_VALUE
    bench_styles = {"SPY": ("forestgreen", "--"), "BTC-USD": ("purple", "-.")}
    for ticker, series in benchmarks.items():
        if series.empty:
            continue
        # Forward-fill to cover weekend gaps when aligning with Kalshi 24/7 data
        idx = pd.date_range(oos_start.date(), oos_end.date(), freq="D")
        series_filled = series.reindex(idx).ffill().bfill()
        norm = PORTFOLIO_VALUE * series_filled / series_filled.iloc[0]
        color, ls = bench_styles.get(ticker, ("gray", ":"))
        ax.plot(norm.index, norm.values, color=color, linestyle=ls,
                linewidth=1.5, alpha=0.85, label=f"{ticker} buy-and-hold")

    ax.axhline(PORTFOLIO_VALUE, color="black", linewidth=0.8, alpha=0.5,
               linestyle=":", label=f"Breakeven (${PORTFOLIO_VALUE:,.0f})")
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))
    ax.xaxis.set_major_locator(mdates.WeekdayLocator(byweekday=0))
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=30, ha="right")
    ax.set_ylabel(f"Portfolio Value  (starting ${PORTFOLIO_VALUE:,.0f}, $1/trade)")
    ax.set_xlabel("Date")
    ax.set_title(
        "Cumulative Portfolio Value — BTC Kalshi Strategies vs Benchmarks\n"
        f"OOS period: {oos_start.date()} → {oos_end.date()}  |  threshold = 0.55"
    )
    ax.legend(fontsize=9)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "backtest_timeseries.png", dpi=150, bbox_inches="tight")
    plt.show()
    plt.close()
    print("Saved → figures/backtest_timeseries.png")


def plot_roc_comparison(df: pd.DataFrame) -> None:
    """
    ROC curves for all three models on the same OOS contract set.
    Puts model discriminative power side-by-side so the reader can see
    whether XGBoost's nonlinear capacity improves over LR.
    """
    from sklearn.metrics import roc_curve, roc_auc_score
    models = {
        "Baseline LR": "baseline_proba",
        "Extended LR": "extended_proba",
        "XGBoost":     "xgb_proba",
    }
    colors = {"Baseline LR": "steelblue", "Extended LR": "darkorange", "XGBoost": "tomato"}

    fig, ax = plt.subplots(figsize=(7, 6))
    ax.plot([0, 1], [0, 1], "k--", linewidth=1, alpha=0.4, label="Random (AUC = 0.50)")

    for label, col in models.items():
        fpr, tpr, _ = roc_curve(df["target"], df[col])
        auc_val = roc_auc_score(df["target"], df[col])
        ax.plot(fpr, tpr, color=colors[label], linewidth=2,
                label=f"{label}  (AUC = {auc_val:.3f})")

    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title(
        "ROC Curve Comparison — All Models  (BTC, OOS)\n"
        "Evaluated on the same set of held-out contracts"
    )
    ax.legend(loc="lower right", fontsize=9)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "backtest_roc_comparison.png", dpi=150, bbox_inches="tight")
    plt.show()
    plt.close()
    print("Saved → figures/backtest_roc_comparison.png")


def plot_model_bars(df: pd.DataFrame) -> None:
    """
    Side-by-side accuracy and AUC for all three models.
    Baseline LR = minimal same-asset benchmark; Extended LR and XGBoost show
    how much adding features and nonlinearity improves over the baseline.
    """
    from sklearn.metrics import accuracy_score, roc_auc_score
    models = {
        "Baseline LR": ("baseline_proba", "steelblue"),
        "Extended LR": ("extended_proba", "darkorange"),
        "XGBoost":     ("xgb_proba",      "tomato"),
    }
    target = df["target"].values

    metrics  = ["Accuracy", "AUC-ROC"]
    x        = np.arange(len(metrics))
    width    = 0.22
    offsets  = [-width, 0, width]

    fig, ax = plt.subplots(figsize=(8, 5))
    for (label, (col, color)), offset in zip(models.items(), offsets):
        pred  = (df[col].values >= 0.5).astype(int)
        acc   = accuracy_score(target, pred)
        a_roc = roc_auc_score(target, df[col].values)
        vals  = [acc, a_roc]
        bars  = ax.bar(x + offset, vals, width, label=label, color=color, alpha=0.85)
        for bar in bars:
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.003,
                    f"{bar.get_height():.3f}", ha="center", va="bottom", fontsize=8)

    ax.axhline(0.5, color="gray", linestyle="--", linewidth=1, alpha=0.6)
    ax.set_xticks(x)
    ax.set_xticklabels(metrics)
    ax.set_ylabel("Score")
    ax.set_ylim(0.45, 0.82)
    ax.set_title(
        "Model Comparison — Accuracy & AUC-ROC  (BTC, OOS)\n"
        "All evaluated on the same held-out contracts"
    )
    ax.legend()
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "backtest_model_bars.png", dpi=150, bbox_inches="tight")
    plt.show()
    plt.close()
    print("Saved → figures/backtest_model_bars.png")


def plot_sharpe_grid(thresholds: list, sharpe_vals: list, n_trades: list) -> None:
    """
    Threshold sensitivity on the Extended LR signal.
    Higher thresholds → fewer trades but higher hit rate (trades only on strong conviction).
    Diminishing returns set in as sample size shrinks.
    """
    fig, ax1 = plt.subplots(figsize=(9, 5))
    ax2 = ax1.twinx()

    color_s, color_n = "steelblue", "coral"
    ax1.plot(thresholds, sharpe_vals, "o-", color=color_s, linewidth=2,
             label="Annualised Sharpe")
    ax1.axhline(0, color="gray", linestyle="--", linewidth=0.8)
    ax2.bar(thresholds, n_trades, width=0.018, alpha=0.35, color=color_n,
            label="# Trades")

    ax1.set_xlabel("Entry Threshold")
    ax1.set_ylabel("Annualised Sharpe", color=color_s)
    ax2.set_ylabel("# Trades", color=color_n)
    ax1.set_title("Threshold Sensitivity: Sharpe vs # Trades  (Extended LR, TC=0)")
    ax1.set_xticks(thresholds)
    ax1.set_xticklabels([f"{t:.2f}" for t in thresholds])

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "backtest_sharpe_grid.png", dpi=150, bbox_inches="tight")
    plt.show()
    plt.close()
    print("Saved → figures/backtest_sharpe_grid.png")


def plot_tc_sensitivity(tcs: list, metrics: list) -> None:
    """
    How robust is the Extended LR strategy to transaction costs?
    Kalshi's spread/fee is low relative to $1 binary contract, so the strategy
    should be resilient. This plot quantifies the break-even TC level.
    """
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    for ax, key, ylabel in zip(
        axes,
        ["sharpe", "total_pnl"],
        ["Annualised Sharpe", "Total PnL ($)"],
    ):
        vals = [m[key] for m in metrics]
        ax.plot([t * 100 for t in tcs], vals, "o-", color="steelblue", linewidth=2)
        ax.axhline(0, color="gray", linestyle="--", linewidth=0.8)
        ax.set_xlabel("Transaction Cost (%)")
        ax.set_ylabel(ylabel)
        ax.set_xticks([t * 100 for t in tcs])
        ax.set_title(f"{ylabel} vs Transaction Cost  (Extended LR, threshold=0.55)")

    plt.suptitle("Transaction Cost Robustness  (BTC Extended LR Strategy)", y=1.02)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "backtest_tc_sensitivity.png", dpi=150, bbox_inches="tight")
    plt.show()
    plt.close()
    print("Saved → figures/backtest_tc_sensitivity.png")


# ---------------------------------------------------------------------------
# Print helpers
# ---------------------------------------------------------------------------

def print_metrics(label: str, r: dict) -> None:
    def fmt(val, fmt_str):
        return format(val, fmt_str) if not (isinstance(val, float) and np.isnan(val)) else "—"
    print(f"\n  {label}")
    print(f"    Trades     : {r['n_trades']:,}")
    print(f"    Hit rate   : {fmt(r['hit_rate'], '.3f')}")
    print(f"    Total PnL  : {fmt(r['total_pnl'], '.2f')}")
    print(f"    Sharpe     : {fmt(r['sharpe'], '.3f')}")
    print(f"    Max DD     : {fmt(r['max_drawdown'], '.2f')}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("Loading OOS predictions from all models...")
    df = load_all_predictions()
    if df is None:
        return

    signal_map = {
        "Baseline LR": "baseline_proba",
        "Extended LR": "extended_proba",
        "XGBoost":     "xgb_proba",
    }
    target = df["target"].values

    print("\n" + "=" * 58)
    print("Backtest Results — BTC Kalshi 15-min Contracts  (threshold=0.55)")
    print("=" * 58)

    # Per-model backtest summary
    for label, col in signal_map.items():
        r = run_backtest(df[col].values, target, threshold=0.55, tc=0.0)
        print_metrics(f"{label}  (TC = 0)", r)

    # Threshold grid on Extended LR (most informative signal)
    print("\n--- Threshold Grid  (Extended LR, TC=0) ---")
    thresholds = np.round(np.arange(0.50, 0.71, 0.05), 2).tolist()
    sharpes, n_trades_list = [], []
    print(f"\n  {'Threshold':>10}  {'# Trades':>9}  {'Hit Rate':>9}  "
          f"{'Total PnL':>10}  {'Sharpe':>8}")
    for thr in thresholds:
        r = run_backtest(df["extended_proba"].values, target, threshold=thr, tc=0.0)
        sharpes.append(r["sharpe"] if not np.isnan(r["sharpe"]) else 0)
        n_trades_list.append(r["n_trades"])
        print(f"  {thr:>10.2f}  {r['n_trades']:>9,}  {r['hit_rate']:>9.3f}  "
              f"{r['total_pnl']:>10.2f}  "
              + (f"{r['sharpe']:>8.3f}" if not np.isnan(r["sharpe"]) else "       —"))

    # TC sensitivity on Extended LR
    print("\n--- Transaction Cost Sensitivity  (Extended LR, threshold=0.55) ---")
    tcs = [0.0, 0.001, 0.005]
    tc_metrics = []
    print(f"\n  {'TC':>8}  {'# Trades':>9}  {'Hit Rate':>9}  {'Total PnL':>10}  {'Sharpe':>8}")
    for tc in tcs:
        r = run_backtest(df["extended_proba"].values, target, threshold=0.55, tc=tc)
        tc_metrics.append(r)
        print(f"  {tc*100:>7.1f}%  {r['n_trades']:>9,}  {r['hit_rate']:>9.3f}  "
              f"{r['total_pnl']:>10.2f}  "
              + (f"{r['sharpe']:>8.3f}" if not np.isnan(r["sharpe"]) else "       —"))

    # Benchmark data from yfinance
    oos_start = df["open_key"].min()
    oos_end   = df["open_key"].max()
    print(f"\nFetching benchmark data ({oos_start.date()} → {oos_end.date()})...")
    benchmarks = fetch_benchmarks(
        str(oos_start.date()),
        str((oos_end + pd.Timedelta(days=1)).date()),
    )
    for ticker, series in benchmarks.items():
        if not series.empty:
            ret = (series.iloc[-1] / series.iloc[0] - 1) * 100
            print(f"  {ticker}: {len(series)} daily bars, total return = {ret:+.2f}%")

    # Build date-indexed strategy curves
    strategy_series = build_strategy_timeseries(df, threshold=0.55)
    for label, series in strategy_series.items():
        total_gain = series.iloc[-1] - PORTFOLIO_VALUE
        print(f"  {label}: ${total_gain:+,.2f} gain over OOS period")

    # Figures
    print("\n--- Generating figures ---")
    plot_timeseries(strategy_series, benchmarks, oos_start, oos_end)
    plot_roc_comparison(df)
    plot_model_bars(df)
    plot_sharpe_grid(thresholds, sharpes, n_trades_list)
    plot_tc_sensitivity(tcs, tc_metrics)

    print("\nDone. All figures saved to figures/.")


if __name__ == "__main__":
    main()
