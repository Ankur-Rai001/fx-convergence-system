import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import argparse
import logging
import sys
from pathlib import Path
from datetime import datetime
import config
from data.fetcher import load_all_pairs
from strategy.signal_generator import generate_signals
from backtest.engine import run_backtest
from backtest.metrics import compute_metrics, check_gate, print_metrics
from walk_forward.wfo import run_wfo
# from reports.charts import plot_equity_curve, plot_trade_distribution, plot_wfo_summary

# ── Logging setup ─────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt="%H:%M:%S",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(Path(config.REPORTS_DIR) / "run.log", mode="a"),
    ],
)
logger = logging.getLogger("main")


# ── Main entry point ──────────────────────────────────────────────────────────

def main() -> None:
    Path(config.REPORTS_DIR).mkdir(parents=True, exist_ok=True)

    parser = argparse.ArgumentParser(description="FX Convergence System")
    parser.add_argument(
        "--mode", choices=["backtest", "wfo", "live"],
        default="backtest",
        help="backtest: single pair backtest | wfo: walk-forward | live: check signals now",
    )
    parser.add_argument(
        "--pair", default=None,
        help="Pair to backtest (e.g. EURUSD=X). Defaults to all pairs in config.",
    )
    parser.add_argument(
        "--refresh", action="store_true",
        help="Force re-download data (ignore cache)",
    )
    args = parser.parse_args()

    # ── Load data ─────────────────────────────────────────────────────────────
    logger.info(f"Mode: {args.mode.upper()} | Pairs: {args.pair or 'ALL'}")
    data = load_all_pairs(refresh=args.refresh)

    if not data:
        logger.error("No data loaded. Check your internet connection or pair tickers.")
        sys.exit(1)

    # Filter to single pair if requested
    if args.pair:
        if args.pair not in data:
            logger.error(f"Pair '{args.pair}' not found. Available: {list(data.keys())}")
            sys.exit(1)
        data = {args.pair: data[args.pair]}

    # ── Route to mode ─────────────────────────────────────────────────────────
    if args.mode == "backtest":
        _run_backtest(data)
    elif args.mode == "wfo":
        _run_wfo(data)
    elif args.mode == "live":
        _run_live(data)


# ── Backtest mode ─────────────────────────────────────────────────────────────
def _run_backtest(data: dict) -> None:
    import pandas as pd
    all_trades        = []
    all_equity_curves = []
    timestamp         = datetime.now().strftime("%Y%m%d_%H%M%S")

    for pair, df in data.items():
        logger.info(f"--- Backtesting {pair} ---")
        df.attrs["pair"] = pair

        df_signals              = generate_signals(df)
        trades_df, equity_curve = run_backtest(df_signals)

        if not trades_df.empty:
            metrics = compute_metrics(trades_df, equity_curve)
            print_metrics(metrics, label=f"{pair} Backtest Results")

            gate_passed, failures = check_gate(metrics)
            if not gate_passed:
                logger.warning(f"{pair}: Gate failures: {failures}")

            all_trades.append(trades_df)
            all_equity_curves.append(equity_curve)

    out = Path(config.REPORTS_DIR)

    if all_trades:
        combined = pd.concat(all_trades, ignore_index=True)

        # ── File 1: Trade log ──────────────────────────────────────
        trade_log_path = out / f"trade_log_{timestamp}.csv"
        combined.to_csv(trade_log_path, index=False)
        logger.info(f"Trade log saved: {trade_log_path}")

        # ── File 2: Metrics summary ────────────────────────────────
        # Build proper daily combined equity curve (fixes Sortino bug)
        # Sum daily P&L changes across all pairs on each calendar day.
        start_cap       = config.STARTING_CAPITAL
        daily_pnl_parts = [eq.diff().fillna(0) for eq in all_equity_curves]
        combined_daily  = pd.concat(daily_pnl_parts, axis=1).fillna(0).sum(axis=1)
        equity_proxy    = start_cap + combined_daily.cumsum()

        portfolio_metrics = compute_metrics(combined, equity_proxy, start_cap)
        print_metrics(portfolio_metrics, label="PORTFOLIO — All Pairs Combined")

        summary_path = out / f"summary_{timestamp}.txt"
        with open(summary_path, "w") as f:
            f.write(f"FX Convergence System — Run Summary\n")
            f.write(f"Timestamp : {timestamp}\n")
            f.write(f"Pairs     : {list(data.keys())}\n")
            f.write(f"Period    : {config.START_DATE} to {config.END_DATE}\n")
            f.write("=" * 52 + "\n")
            for k, v in portfolio_metrics.items():
                f.write(f"  {k:<22}: {v}\n")
            f.write("=" * 52 + "\n")
            gate_passed, failures = check_gate(portfolio_metrics)
            f.write(f"  Gate Result: {'PASS' if gate_passed else 'FAIL'}\n")
            if failures:
                for fail in failures:
                    f.write(f"    - {fail}\n")
        logger.info(f"Summary saved: {summary_path}")


# ── WFO mode ──────────────────────────────────────────────────────────────────

def _run_wfo(data: dict) -> None:
    """Run 6-fold Walk-Forward Optimization and print/chart results."""
    wfo_results = run_wfo(data)
    plot_wfo_summary(wfo_results)

    # Save WFO results to CSV for reference
    import pandas as pd
    rows = []
    for r in wfo_results:
        row = {
            "fold"       : r["fold"],
            "oos_start"  : r["oos_start"],
            "oos_end"    : r["oos_end"],
            "gate_passed": r["gate_passed"],
            "best_sl"    : r["best_params"].get("SL_MULT"),
            "best_tp"    : r["best_params"].get("TP_MULT"),
            **{f"oos_{k}": v for k, v in r["oos_metrics"].items()
               if isinstance(v, (int, float))},
        }
        rows.append(row)
    results_df = pd.DataFrame(rows)
    out_path   = Path(config.REPORTS_DIR) / "wfo_results.csv"
    results_df.to_csv(out_path, index=False)
    logger.info(f"WFO results saved to {out_path}")


# ── Live signal mode ──────────────────────────────────────────────────────────

def _run_live(data: dict) -> None:
    """
    Check for live signals on the most recent bar of each pair.

    This is the "run anytime" mode — checks whether right now (latest bar)
    both the SR zone and MACD divergence signals are firing.
    If yes: prints entry details (direction, entry price, SL, TP).
    """
    import config as cfg
    from strategy.indicators import compute_atr

    print(f"\n{'='*55}")
    print(f"  LIVE SIGNAL CHECK — {len(data)} pairs")
    print(f"{'='*55}")

    signals_found = 0

    for pair, df in data.items():
        df.attrs["pair"] = pair
        df_signals = generate_signals(df)

        latest     = df_signals.iloc[-1]
        latest_date = df_signals.index[-1]
        signal     = int(latest["signal"])
        atr        = latest["atr"]
        close      = latest["Close"]

        direction_str = {1: "LONG", -1: "SHORT", 0: "NONE"}[signal]

        if signal == 0:
            print(f"  {pair:<12} | {latest_date.date()} | No signal")
            continue

        sl_price = close - signal * cfg.SL_MULT * atr
        tp_price = close + signal * cfg.TP_MULT * atr
        lot_size = _calc_lot(cfg.STARTING_CAPITAL, atr, cfg.SL_MULT)

        print(f"\n  {'*'*50}")
        print(f"  SIGNAL: {pair} — {direction_str}")
        print(f"  Date         : {latest_date.date()}")
        print(f"  Entry (now)  : {close:.5f}")
        print(f"  Stop Loss    : {sl_price:.5f}  ({cfg.SL_MULT}x ATR)")
        print(f"  Take Profit  : {tp_price:.5f}  ({cfg.TP_MULT}x ATR)")
        print(f"  ATR          : {atr:.5f}")
        print(f"  Lot Size     : {lot_size}")
        print(f"  {'*'*50}\n")
        signals_found += 1

    if signals_found == 0:
        print(f"\n  No signals firing right now across {len(data)} pairs.")
    print(f"{'='*55}\n")


def _calc_lot(capital: float, atr: float, sl_mult: float) -> float:
    risk    = capital * config.RISK_PCT
    sl_pips = (sl_mult * atr) / config.PIP_SIZE
    raw     = risk / (sl_pips * config.PIP_VALUE)
    return round((raw // config.LOT_STEP) * config.LOT_STEP, 2)


# ── Entry ─────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    main()