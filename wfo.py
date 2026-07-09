# =============================================================================
# walk_forward/wfo.py — FX Convergence System
#
# What this file does (plain English):
#   Walk-Forward Optimization (WFO) — the honest way to test a strategy.
#
#   The problem with normal backtesting:
#       If you optimise parameters on ALL historical data, the strategy learns
#       the specific quirks of that dataset. It looks great in backtest but
#       fails in live trading. This is called curve-fitting or overfitting.
#       Think of it as memorising exam answers vs actually learning.
#
#   How WFO solves it:
#       We divide data into 6 folds. Each fold has:
#         - In-Sample (IS):  data the optimiser SEES     → find best params
#         - Out-of-Sample (OOS): data the optimiser NEVER sees → honest test
#       Best IS params → frozen → tested on OOS.
#       If strategy works across all 6 OOS windows → genuine edge, not luck.
#
#   Fold structure (anchored expanding — IS window grows each fold):
#       Fold 1: IS = Jan2015-Dec2016 | OOS = Jan-Jun 2017
#       Fold 2: IS = Jan2015-Jun2017 | OOS = Jul-Dec 2017
#       ...     (IS grows forward, OOS always fresh 6 months)
#       Fold 6: IS = Jan2015-Jun2019 | OOS = Jul-Dec 2019
# =============================================================================

import itertools
import logging
from dateutil.relativedelta import relativedelta

import pandas as pd

import config
from strategy.signal_generator import generate_signals
from backtest.engine import run_backtest
from backtest.metrics import compute_metrics, check_gate, print_metrics

logger = logging.getLogger(__name__)


# ── Public API ────────────────────────────────────────────────────────────────

def run_wfo(
    data: dict[str, pd.DataFrame],
    param_grid: dict | None = None,
    n_folds: int | None = None,
    oos_months: int | None = None,
) -> list[dict]:
    """
    Run Walk-Forward Optimization across all pairs.

    Args:
        data       : dict of {pair: OHLCV DataFrame} from data.fetcher
        param_grid : parameter combinations to test (defaults to config.PARAM_GRID)
        n_folds    : number of WFO folds (default config.WFO_N_FOLDS)
        oos_months : OOS window size in months (default config.WFO_OOS_MONTHS)

    Returns:
        List of fold result dicts, one per fold, containing:
            fold, oos_start, oos_end, best_params, oos_metrics, gate_passed
    """
    grid       = param_grid  or config.PARAM_GRID
    n_folds    = n_folds     or config.WFO_N_FOLDS
    oos_months = oos_months  or config.WFO_OOS_MONTHS
    combos     = _expand_grid(grid)

    logger.info(f"WFO: {n_folds} folds | {len(combos)} param combos | OOS={oos_months}mo")

    # Combine all pairs into one DataFrame for portfolio-level WFO
    # (optimise on aggregate across all pairs simultaneously)
    folds   = _build_folds(data, n_folds, oos_months)
    results = []

    for fold_num, fold in enumerate(folds, start=1):
        logger.info(
            f"\n{'='*60}\n"
            f"FOLD {fold_num}/{n_folds}  "
            f"IS: {fold['is_start'].date()} -> {fold['is_end'].date()}  |  "
            f"OOS: {fold['oos_start'].date()} -> {fold['oos_end'].date()}"
            f"\n{'='*60}"
        )

        # ── IS: find best param combo ──────────────────────────────────────
        best_params, best_pf = _optimize_is(data, fold, combos)
        logger.info(f"Fold {fold_num}: best IS params: {best_params} (PF={best_pf:.2f})")

        # ── OOS: evaluate best params on unseen data ───────────────────────
        oos_metrics = _evaluate_oos(data, fold, best_params)

        gate_passed, failures = check_gate(oos_metrics)
        status = "PASS" if gate_passed else f"FAIL ({', '.join(failures)})"
        logger.info(f"Fold {fold_num}: OOS gate → {status}")

        results.append({
            "fold"        : fold_num,
            "is_start"    : fold["is_start"].date(),
            "is_end"      : fold["is_end"].date(),
            "oos_start"   : fold["oos_start"].date(),
            "oos_end"     : fold["oos_end"].date(),
            "best_params" : best_params,
            "best_is_pf"  : best_pf,
            "oos_metrics" : oos_metrics,
            "gate_passed" : gate_passed,
            "gate_failures": failures,
        })

        print_metrics(oos_metrics, label=f"FOLD {fold_num} OOS  ({fold['oos_start'].date()} - {fold['oos_end'].date()})")

    _print_wfo_summary(results)
    return results


# ── Fold builder ──────────────────────────────────────────────────────────────

def _build_folds(
    data: dict[str, pd.DataFrame],
    n_folds: int,
    oos_months: int,
) -> list[dict]:
    """
    Build fold date windows.

    Anchored expanding IS window:
        IS always starts at the very beginning of the data.
        Each fold, IS end moves forward by oos_months.
        OOS always immediately follows IS end.

    Example with oos_months=6, n_folds=6:
        Fold 1: IS end = start+24mo, OOS = [IS_end, IS_end+6mo]
        Fold 2: IS end = IS_end+6mo, OOS = [new IS_end, new IS_end+6mo]
        ...
    """
    # Use the first pair's index as the reference calendar
    first_pair = next(iter(data.values()))
    global_start = first_pair.index.min()

    # IS window for fold 1: we need at least (n_folds × oos_months) months
    # remaining for OOS, so IS starts at global_start and ends at
    # global_start + (n_folds × oos_months) after skipping a 24-month warmup.
    warmup_months = 24   # minimum IS window for fold 1
    is_end_fold1  = global_start + relativedelta(months=warmup_months)

    folds = []
    for i in range(n_folds):
        is_end   = is_end_fold1 + relativedelta(months=i * oos_months)
        oos_start = is_end
        oos_end   = oos_start + relativedelta(months=oos_months)

        folds.append({
            "is_start" : global_start,
            "is_end"   : is_end,
            "oos_start": oos_start,
            "oos_end"  : oos_end,
        })

    return folds


# ── IS optimisation ───────────────────────────────────────────────────────────

def _optimize_is(
    data: dict[str, pd.DataFrame],
    fold: dict,
    combos: list[dict],
) -> tuple[dict, float]:
    """
    Test every parameter combination on IS data.
    Select the combo with the highest Profit Factor.

    Returns:
        (best_params dict, best_profit_factor float)
    """
    best_pf     = -1.0
    best_params = combos[0]

    for params in combos:
        pf = _run_combo_all_pairs(data, fold["is_start"], fold["is_end"], params)
        if pf > best_pf:
            best_pf     = pf
            best_params = params

    return best_params, best_pf


def _evaluate_oos(
    data: dict[str, pd.DataFrame],
    fold: dict,
    params: dict,
) -> dict:
    """
    Run the frozen best IS params on the OOS window.
    Aggregate metrics across all pairs.

    CRITICAL: OOS data must never have been seen by the optimizer.
    We slice [oos_start : oos_end] AFTER running signals on full history
    (so warmup indicators like MA and MACD have sufficient history),
    then filter trades to the OOS window only.
    """
    all_trades: list = []
    all_equity: list = []

    for pair, df in data.items():
        # Slice data up to OOS end (full history for indicator warmup)
        df_full = df[df.index <= fold["oos_end"]].copy()
        df_full.attrs["pair"] = pair

        # Generate signals on full history (no lookahead — indicators use past bars only)
        df_signals = generate_signals(df_full, params)

        # Run backtest on full slice
        trades_df, equity_curve = run_backtest(df_signals, params)

        if trades_df.empty:
            continue

        # Filter trades to OOS window only
        oos_trades = trades_df[
            (trades_df["entry_date"] >= fold["oos_start"]) &
            (trades_df["entry_date"] <  fold["oos_end"])
        ].copy()

        all_trades.append(oos_trades)

    if not all_trades:
        from backtest.metrics import _empty_metrics
        return _empty_metrics()

    combined_trades = pd.concat(all_trades, ignore_index=True)

    # Build a combined equity curve for Sortino/MaxDD
    # (use net P&L cumsum from starting capital as proxy)
    start_cap  = config.STARTING_CAPITAL
    equity_sim = pd.Series(
        start_cap + combined_trades["pnl_usd"].cumsum().values,
        index=combined_trades["entry_date"].values,
    )

    return compute_metrics(combined_trades, equity_sim, start_cap)


# ── Grid helpers ──────────────────────────────────────────────────────────────

def _expand_grid(param_grid: dict) -> list[dict]:
    """
    Convert PARAM_GRID dict into a flat list of all parameter combinations.

    Example:
        {"A": [1,2], "B": [3,4]}  →  [{"A":1,"B":3}, {"A":1,"B":4},
                                        {"A":2,"B":3}, {"A":2,"B":4}]
    """
    keys   = list(param_grid.keys())
    values = list(param_grid.values())
    combos = [dict(zip(keys, v)) for v in itertools.product(*values)]
    logger.info(f"Grid expanded: {len(combos)} parameter combinations")
    return combos


def _run_combo_all_pairs(
    data: dict[str, pd.DataFrame],
    start: pd.Timestamp,
    end: pd.Timestamp,
    params: dict,
) -> float:
    """
    Run a single parameter combo on all pairs within [start, end].
    Returns aggregate Profit Factor across all pairs.
    Returns 0.0 on any error.
    """
    all_wins   = 0.0
    all_losses = 0.0

    for pair, df in data.items():
        df_slice = df[(df.index >= start) & (df.index <= end)].copy()
        df_slice.attrs["pair"] = pair

        if len(df_slice) < 60:   # not enough bars for reliable indicators
            continue

        try:
            df_signals          = generate_signals(df_slice, params)
            trades_df, _        = run_backtest(df_signals, params)
            if trades_df.empty:
                continue
            all_wins   += trades_df[trades_df["pnl_usd"] > 0]["pnl_usd"].sum()
            all_losses += trades_df[trades_df["pnl_usd"] < 0]["pnl_usd"].abs().sum()
        except Exception as exc:
            logger.debug(f"Combo failed on {pair}: {exc}")

    if all_losses == 0:
        return 0.0
    return round(all_wins / all_losses, 3)


# ── Summary printer ───────────────────────────────────────────────────────────

def _print_wfo_summary(results: list[dict]) -> None:
    """Print a concise WFO summary table."""
    print(f"\n{'='*70}")
    print(f"  WALK-FORWARD OPTIMIZATION SUMMARY")
    print(f"{'='*70}")
    print(f"  {'Fold':<6} {'OOS Period':<25} {'PF':>6} {'Sortino':>8} {'MaxDD':>7} {'Gate'}")
    print(f"  {'-'*62}")
    n_pass = 0
    for r in results:
        m    = r["oos_metrics"]
        gate = "PASS" if r["gate_passed"] else "FAIL"
        if r["gate_passed"]:
            n_pass += 1
        print(
            f"  {r['fold']:<6} "
            f"{str(r['oos_start'])+' - '+str(r['oos_end']):<25} "
            f"{m['profit_factor']:>6.2f} "
            f"{m['sortino_ratio']:>8.2f} "
            f"{m['max_drawdown']:>7.1%} "
            f"{gate}"
        )
    print(f"  {'-'*62}")
    print(f"  Folds passed: {n_pass}/{len(results)}  (gate: >= {config.GATE_MIN_FOLDS})")
    overall = "SYSTEM APPROVED" if n_pass >= config.GATE_MIN_FOLDS else "SYSTEM REJECTED"
    print(f"  Final verdict: {overall}")
    print(f"{'='*70}\n")
