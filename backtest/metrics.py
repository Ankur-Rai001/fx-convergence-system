import logging

import numpy as np
import pandas as pd

import config

logger = logging.getLogger(__name__)


# ── Main metrics function ─────────────────────────────────────────────────────

def compute_metrics(
    trades_df: pd.DataFrame,
    equity_curve: pd.Series,
    starting_capital: float | None = None,
) -> dict:
    """
    Compute all performance metrics from completed trades and equity curve.

    Args:
        trades_df        : output from backtest.engine.run_backtest()
        equity_curve     : output from backtest.engine.run_backtest()
        starting_capital : starting account balance (defaults to config value)

    Returns:
        dict of all metrics — keys are metric names, values are floats/strings
    """
    start_cap = starting_capital or config.STARTING_CAPITAL

    if trades_df.empty:
        logger.warning("No trades to compute metrics on.")
        return _empty_metrics()

    wins  = trades_df[trades_df["pnl_usd"] > 0]["pnl_usd"]
    losses = trades_df[trades_df["pnl_usd"] <= 0]["pnl_usd"]

    total_profit = wins.sum()
    total_loss   = losses.abs().sum()
    end_capital  = equity_curve.iloc[-1]
    n_years      = (equity_curve.index[-1] - equity_curve.index[0]).days / 365.25

    metrics = {
        # ── Trade counts ──────────────────────────────────────────────────
        "n_trades"     : len(trades_df),
        "n_wins"       : len(wins),
        "n_losses"     : len(losses),
        "win_rate"     : len(wins) / len(trades_df),

        # ── Profit ────────────────────────────────────────────────────────
        "total_profit_usd" : round(total_profit, 2),
        "total_loss_usd"   : round(total_loss, 2),
        "net_pnl_usd"      : round(total_profit - total_loss, 2),
        "profit_factor"    : _profit_factor(total_profit, total_loss),

        # ── Per-trade ─────────────────────────────────────────────────────
        "avg_win_usd"       : round(wins.mean(), 2)         if len(wins)   > 0 else 0,
        "avg_loss_usd"      : round(losses.abs().mean(), 2) if len(losses) > 0 else 0,
        "expectancy_usd"    : _expectancy(wins, losses),

        # ── Returns ───────────────────────────────────────────────────────
        "cagr"             : _cagr(start_cap, end_capital, n_years),
        "total_return_pct" : round((end_capital - start_cap) / start_cap * 100, 2),

        # ── Risk ──────────────────────────────────────────────────────────
        "max_drawdown"     : _max_drawdown(equity_curve),
        "sortino_ratio"    : _sortino(equity_curve),

        # ── Integrity check ───────────────────────────────────────────────
        "tp_sl_ratio"      : _tp_sl_ratio(trades_df),
        "tp_sl_ok"         : _check_tp_sl_ratio(trades_df),

        # ── Exit breakdown ────────────────────────────────────────────────
        "tp_hit_count"     : (trades_df["exit_reason"] == "TP").sum(),
        "sl_hit_count"     : (trades_df["exit_reason"] == "SL").sum(),
        "tp_hit_rate"      : (trades_df["exit_reason"] == "TP").mean(),
    }

    _log_metrics(metrics)
    return metrics


def check_gate(metrics: dict) -> tuple[bool, list[str]]:
    """
    Check whether metrics pass all gate criteria defined in config.

    Returns:
        (passed: bool, failures: list of failure messages)
    """
    failures: list[str] = []

    if metrics["profit_factor"] < config.GATE_PF:
        failures.append(f"Profit Factor {metrics['profit_factor']:.2f} < {config.GATE_PF}")

    if metrics["sortino_ratio"] < config.GATE_SORTINO:
        failures.append(f"Sortino {metrics['sortino_ratio']:.2f} < {config.GATE_SORTINO}")

    if metrics["max_drawdown"] > config.GATE_MAX_DD:
        failures.append(f"MaxDD {metrics['max_drawdown']:.1%} > {config.GATE_MAX_DD:.0%}")

    if not metrics["tp_sl_ok"]:
        failures.append(f"TP/SL ratio {metrics['tp_sl_ratio']:.3f} outside 1.95–2.05 — BUG")

    return (len(failures) == 0, failures)


def print_metrics(metrics: dict, label: str = "") -> None:
    """Print a formatted metrics summary to stdout."""
    sep = "=" * 52
    print(f"\n{sep}")
    if label:
        print(f"  {label}")
    print(sep)
    print(f"  Trades         : {metrics['n_trades']}")
    print(f"  Win Rate       : {metrics['win_rate']:.1%}")
    print(f"  Profit Factor  : {metrics['profit_factor']:.2f}")
    print(f"  Sortino Ratio  : {metrics['sortino_ratio']:.2f}")
    print(f"  Max Drawdown   : {metrics['max_drawdown']:.1%}")
    print(f"  CAGR           : {metrics['cagr']:.1%}")
    print(f"  Net P&L        : ${metrics['net_pnl_usd']:,.2f}")
    print(f"  Expectancy     : ${metrics['expectancy_usd']:.2f} / trade")
    print(f"  TP Hit Rate    : {metrics['tp_hit_rate']:.1%}")
    print(f"  TP/SL Ratio    : {metrics['tp_sl_ratio']:.3f}  {'OK' if metrics['tp_sl_ok'] else 'BUG!'}")
    print(sep)


# ── Individual metric calculations ────────────────────────────────────────────

def _profit_factor(total_profit: float, total_loss: float) -> float:
    """
    Profit Factor = Total Gross Profit / Total Gross Loss.
    PF = 1.0 → breakeven. PF = 2.0 → target.
    Returns 0.0 if no losses (edge case).
    """
    if total_loss == 0:
        return 0.0 if total_profit == 0 else 999.0
    return round(total_profit / total_loss, 3)


def _expectancy(wins: pd.Series, losses: pd.Series) -> float:
    """
    Average USD profit per trade across all trades.
    Expectancy = (Win Rate × Avg Win) - (Loss Rate × Avg Loss)
    Positive expectancy = system makes money on average per trade.
    """
    n = len(wins) + len(losses)
    if n == 0:
        return 0.0
    total_wins   = wins.sum()   if len(wins)   > 0 else 0
    total_losses = losses.sum() if len(losses) > 0 else 0
    return round((total_wins + total_losses) / n, 2)


def _cagr(start: float, end: float, years: float) -> float:
    """
    Compound Annual Growth Rate.
    CAGR = (end / start)^(1/years) - 1
    Answers: "what consistent annual % would take us from start to end?"
    """
    if years <= 0 or start <= 0:
        return 0.0
    return round((end / start) ** (1 / years) - 1, 4)


def _max_drawdown(equity_curve: pd.Series) -> float:
    """
    Maximum peak-to-trough decline in equity as a fraction.
    MaxDD = 0.15 means at worst we were 15% below our previous peak.
    Lower is better. Gate: <= 20%.
    """
    rolling_max = equity_curve.cummax()
    drawdown    = (equity_curve - rolling_max) / rolling_max
    return round(abs(drawdown.min()), 4)


def _sortino(equity_curve: pd.Series, risk_free_rate: float = 0.0) -> float:
    """
    Sortino Ratio = (Annualised Return - Risk Free Rate) / Downside Deviation.

    Unlike Sharpe, Sortino only penalises NEGATIVE returns (losses).
    Upside volatility (big wins) is not penalised.

    We compute daily returns from the equity curve, then annualise.
    Trading days per year = 252.
    """
    
    daily_returns = equity_curve.pct_change().dropna()
    if len(daily_returns) < 20:
        return 0.0

    ann_return   = daily_returns.mean() * 252
    downside     = daily_returns[daily_returns < 0]

    if len(downside) == 0:
        return 999.0

    downside_std = downside.std() * np.sqrt(252)
    if downside_std == 0:
        return 0.0

    sortino = (ann_return - risk_free_rate) / downside_std
    return round(min(sortino, 999.0), 3)  


def _tp_sl_ratio(trades_df: pd.DataFrame) -> float:
    """
    Integrity check: TP distance / SL distance should always equal TP_MULT / SL_MULT.
    For config defaults (TP=2.0, SL=1.0) this must be ~2.0.

    If outside 1.95–2.05, there is a reference price bug in the engine.
    This check must pass before any performance analysis is trusted.
    """
    if "entry_price" not in trades_df.columns or "tp_price" not in trades_df.columns:
        return 0.0

    tp_dist = (trades_df["tp_price"] - trades_df["entry_price"]).abs()
    sl_dist = (trades_df["sl_price"] - trades_df["entry_price"]).abs()

    valid = sl_dist[sl_dist > 0]
    if valid.empty:
        return 0.0

    ratio = (tp_dist[sl_dist > 0] / valid).mean()
    return round(ratio, 3)


def _check_tp_sl_ratio(trades_df: pd.DataFrame) -> bool:
    """Return True if TP/SL ratio is within acceptable range (1.95–2.05)."""
    ratio = _tp_sl_ratio(trades_df)
    return 1.95 <= ratio <= 2.05


def _empty_metrics() -> dict:
    """Return zero-valued metrics dict when no trades exist."""
    return {k: 0 for k in [
        "n_trades", "n_wins", "n_losses", "win_rate", "total_profit_usd",
        "total_loss_usd", "net_pnl_usd", "profit_factor", "avg_win_usd",
        "avg_loss_usd", "expectancy_usd", "cagr", "total_return_pct",
        "max_drawdown", "sortino_ratio", "tp_sl_ratio", "tp_sl_ok",
        "tp_hit_count", "sl_hit_count", "tp_hit_rate",
    ]}


def _log_metrics(metrics: dict) -> None:
    logger.info(
        f"Metrics: PF={metrics['profit_factor']:.2f} | "
        f"Sortino={metrics['sortino_ratio']:.2f} | "
        f"MaxDD={metrics['max_drawdown']:.1%} | "
        f"CAGR={metrics['cagr']:.1%} | "
        f"Trades={metrics['n_trades']} | "
        f"TP/SL={metrics['tp_sl_ratio']:.3f}({'OK' if metrics['tp_sl_ok'] else 'BUG'})"
    )
