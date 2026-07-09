# =============================================================================
# reports/charts.py — FX Convergence System
#
# What this file does (plain English):
#   Generates all charts and saves them to reports/output/.
#   Three charts:
#     1. Equity Curve  — how account balance grew (or fell) over time
#     2. Trade P&L Distribution — histogram showing spread of wins vs losses
#     3. WFO Fold Summary — bar chart of Profit Factor across all 6 OOS folds
#
#   These charts go into the GitHub README as proof of results.
# =============================================================================

import logging
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
import seaborn as sns

import config

logger = logging.getLogger(__name__)

# Use a clean, professional style
plt.style.use("seaborn-v0_8-darkgrid")
PALETTE = {"green": "#2ecc71", "red": "#e74c3c", "blue": "#3498db", "grey": "#95a5a6"}


# ── Public API ────────────────────────────────────────────────────────────────

def plot_equity_curve(
    equity_curve: pd.Series,
    pair: str = "Portfolio",
    save: bool = True,
) -> None:
    """
    Plot account equity over time with drawdown overlay.

    Top panel: equity curve (USD)
    Bottom panel: rolling drawdown from peak (%)
    """
    Path(config.REPORTS_DIR).mkdir(parents=True, exist_ok=True)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 7), sharex=True,
                                    gridspec_kw={"height_ratios": [3, 1]})
    fig.suptitle(f"Equity Curve — {pair}", fontsize=14, fontweight="bold")

    # Equity curve
    ax1.plot(equity_curve.index, equity_curve.values,
             color=PALETTE["blue"], linewidth=1.5, label="Account Equity")
    ax1.axhline(y=equity_curve.iloc[0], color=PALETTE["grey"],
                linestyle="--", linewidth=0.8, label="Starting Capital")
    ax1.fill_between(equity_curve.index, equity_curve.iloc[0],
                     equity_curve.values,
                     where=(equity_curve.values >= equity_curve.iloc[0]),
                     alpha=0.15, color=PALETTE["green"])
    ax1.fill_between(equity_curve.index, equity_curve.iloc[0],
                     equity_curve.values,
                     where=(equity_curve.values < equity_curve.iloc[0]),
                     alpha=0.15, color=PALETTE["red"])
    ax1.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"${x:,.0f}"))
    ax1.set_ylabel("Account Balance (USD)")
    ax1.legend(loc="upper left", fontsize=9)

    # Drawdown panel
    rolling_max = equity_curve.cummax()
    drawdown    = (equity_curve - rolling_max) / rolling_max * 100
    ax2.fill_between(drawdown.index, drawdown.values, 0,
                     color=PALETTE["red"], alpha=0.6)
    ax2.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x:.0f}%"))
    ax2.set_ylabel("Drawdown %")
    ax2.set_xlabel("Date")

    plt.tight_layout()
    if save:
        path = Path(config.REPORTS_DIR) / f"equity_curve_{pair.replace('=','_')}.png"
        plt.savefig(path, dpi=150, bbox_inches="tight")
        logger.info(f"Saved: {path}")
    plt.show()
    plt.close()


def plot_trade_distribution(
    trades_df: pd.DataFrame,
    pair: str = "Portfolio",
    save: bool = True,
) -> None:
    """
    Plot distribution of trade P&L in USD.
    Wins in green, losses in red.
    Vertical dashed lines for avg win and avg loss.
    """
    if trades_df.empty:
        logger.warning("No trades to plot distribution for.")
        return

    Path(config.REPORTS_DIR).mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(10, 5))
    fig.suptitle(f"Trade P&L Distribution — {pair}", fontsize=14, fontweight="bold")

    pnl   = trades_df["pnl_usd"]
    wins  = pnl[pnl > 0]
    losses = pnl[pnl <= 0]

    bins = np.linspace(pnl.min() * 1.1, pnl.max() * 1.1, 40)
    ax.hist(losses, bins=bins, color=PALETTE["red"],   alpha=0.7, label=f"Losses (n={len(losses)})")
    ax.hist(wins,   bins=bins, color=PALETTE["green"], alpha=0.7, label=f"Wins   (n={len(wins)})")

    if len(wins) > 0:
        ax.axvline(wins.mean(),   color=PALETTE["green"], linestyle="--",
                   linewidth=1.5, label=f"Avg Win  ${wins.mean():.2f}")
    if len(losses) > 0:
        ax.axvline(losses.mean(), color=PALETTE["red"],   linestyle="--",
                   linewidth=1.5, label=f"Avg Loss ${losses.mean():.2f}")
    ax.axvline(0, color="black", linewidth=0.8)

    ax.set_xlabel("P&L per Trade (USD)")
    ax.set_ylabel("Number of Trades")
    ax.legend(fontsize=9)

    plt.tight_layout()
    if save:
        path = Path(config.REPORTS_DIR) / f"trade_distribution_{pair.replace('=','_')}.png"
        plt.savefig(path, dpi=150, bbox_inches="tight")
        logger.info(f"Saved: {path}")
    plt.show()
    plt.close()


def plot_wfo_summary(wfo_results: list[dict], save: bool = True) -> None:
    """
    Bar chart of Profit Factor across all WFO folds.
    Green bar = gate passed. Red bar = gate failed.
    Horizontal dashed line at PF = 2.0 (gate threshold).
    """
    if not wfo_results:
        logger.warning("No WFO results to plot.")
        return

    Path(config.REPORTS_DIR).mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle("Walk-Forward Optimization — OOS Results", fontsize=14, fontweight="bold")

    folds     = [r["fold"] for r in wfo_results]
    pf_vals   = [r["oos_metrics"]["profit_factor"]  for r in wfo_results]
    sort_vals = [r["oos_metrics"]["sortino_ratio"]   for r in wfo_results]
    dd_vals   = [r["oos_metrics"]["max_drawdown"] * 100 for r in wfo_results]
    colors    = [PALETTE["green"] if r["gate_passed"] else PALETTE["red"] for r in wfo_results]
    labels    = [f"F{r['fold']}\n{str(r['oos_start'])[2:7]}" for r in wfo_results]

    # PF chart
    _bar_chart(axes[0], folds, pf_vals,   labels, colors,
               "Profit Factor", "Profit Factor", config.GATE_PF, hline_label=f"Gate = {config.GATE_PF}")

    # Sortino chart
    _bar_chart(axes[1], folds, sort_vals, labels, colors,
               "Sortino Ratio", "Sortino",       config.GATE_SORTINO, hline_label=f"Gate = {config.GATE_SORTINO}")

    # MaxDD chart (lower is better — flip gate logic display)
    _bar_chart(axes[2], folds, dd_vals,   labels, colors,
               "Max Drawdown (%)", "MaxDD %",    config.GATE_MAX_DD * 100,
               hline_label=f"Gate = {config.GATE_MAX_DD:.0%}", invert_gate=True)

    n_pass = sum(r["gate_passed"] for r in wfo_results)
    fig.text(0.5, 0.01,
             f"Folds passed: {n_pass}/{len(wfo_results)}  (gate >= {config.GATE_MIN_FOLDS})  |  "
             f"Green = pass, Red = fail",
             ha="center", fontsize=10)

    plt.tight_layout(rect=[0, 0.04, 1, 1])
    if save:
        path = Path(config.REPORTS_DIR) / "wfo_summary.png"
        plt.savefig(path, dpi=150, bbox_inches="tight")
        logger.info(f"Saved: {path}")
    plt.show()
    plt.close()


# ── Internal helper ───────────────────────────────────────────────────────────

def _bar_chart(
    ax, folds, values, labels, colors,
    title, ylabel, gate_value,
    hline_label="", invert_gate=False,
) -> None:
    bars = ax.bar(range(len(folds)), values, color=colors, edgecolor="white", linewidth=0.5)
    ax.axhline(y=gate_value, color="black", linestyle="--",
               linewidth=1.2, label=hline_label)
    ax.set_xticks(range(len(folds)))
    ax.set_xticklabels(labels, fontsize=8)
    ax.set_title(title, fontsize=11, fontweight="bold")
    ax.set_ylabel(ylabel, fontsize=9)
    ax.legend(fontsize=8)

    # Value labels on bars
    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02,
                f"{val:.2f}", ha="center", va="bottom", fontsize=7)
