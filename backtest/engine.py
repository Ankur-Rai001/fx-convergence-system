import logging

import numpy as np
import pandas as pd

import config

logger = logging.getLogger(__name__)


# ── Main backtest function ────────────────────────────────────────────────────

def run_backtest(
    df: pd.DataFrame,
    params: dict | None = None,
    starting_capital: float | None = None,
) -> tuple[pd.DataFrame, pd.Series]:
    """
    Run a full bar-by-bar backtest on a single pair.

    Args:
        df               : DataFrame from signal_generator.generate_signals()
                           Must contain: Open, High, Low, Close, atr, signal columns
        params           : parameter overrides (SL_MULT, TP_MULT, etc.)
        starting_capital : starting USD account balance

    Returns:
        trades_df    : DataFrame of all completed trades with entry/exit details
        equity_curve : pd.Series of account equity at each bar (DatetimeIndex)
    """
    p        = _resolve_params(params)
    capital  = starting_capital or config.STARTING_CAPITAL
    sl_mult  = p["SL_MULT"]
    tp_mult  = p["TP_MULT"]

    trades: list[dict] = []
    short_cooldown: int = 0
    equity: list[float] = []
    position: dict | None = None    # holds the current open trade

    for i in range(1, len(df)):
        if short_cooldown > 0:
            short_cooldown -= 1
        bar      = df.iloc[i]
        prev_bar = df.iloc[i - 1]
        date     = df.index[i]

        # ── Step 1: Check if existing trade hits SL or TP ─────────────────
        if position is not None:
            result = _check_exit(bar, position)
            if result is not None:
                pnl     = _calculate_pnl(position, result["exit_price"], capital)
                capital += pnl["dollar_pnl"]
                position_direction = position["direction"]
                trades.append({
                    **position,
                    "exit_date"  : date,
                    "exit_price" : result["exit_price"],
                    "exit_reason": result["reason"],
                    "pnl_usd"    : pnl["dollar_pnl"],
                    "pnl_pips"   : pnl["pip_pnl"],
                    "capital_after": capital,
                })
                position = None
                if result["reason"] == "SL" and position_direction == -1:
                    short_cooldown = config.SHORT_COOLDOWN_BARS

        # ── Step 2: Check if previous bar had a signal → enter today ──────
        if position is None and prev_bar["signal"] != 0 and not (int(prev_bar["signal"]) == -1 and short_cooldown > 0):
            direction = int(prev_bar["signal"])   # +1 LONG, -1 SHORT
            fill_price = bar["Open"]              # fill at today's open
            atr        = prev_bar["atr"]

            if pd.isna(atr) or atr == 0:
                equity.append(capital)
                continue

            sl_price, tp_price = _calculate_levels(fill_price, atr, direction, sl_mult, tp_mult)
            lot_size = _calculate_lot_size(capital, atr, sl_mult)

            if lot_size < config.MIN_LOT:
                equity.append(capital)
                continue

            position = {
                "pair"       : df.attrs.get("pair", "UNKNOWN"),
                "direction"  : direction,
                "entry_date" : date,
                "entry_price": fill_price,
                "sl_price"   : sl_price,
                "tp_price"   : tp_price,
                "atr_at_entry": atr,
                "lot_size"   : lot_size,
            }

        equity.append(capital)

    # Force-close any open trade at end of data
    if position is not None:
        last_bar   = df.iloc[-1]
        last_date  = df.index[-1]
        exit_price = last_bar["Close"]
        pnl        = _calculate_pnl(position, exit_price, capital)
        capital   += pnl["dollar_pnl"]
        trades.append({
            **position,
            "exit_date"    : last_date,
            "exit_price"   : exit_price,
            "exit_reason"  : "END_OF_DATA",
            "pnl_usd"      : pnl["dollar_pnl"],
            "pnl_pips"     : pnl["pip_pnl"],
            "capital_after": capital,
        })
        if equity:
            equity[-1] = capital

    trades_df    = pd.DataFrame(trades)
    equity_curve = pd.Series(equity, index=df.index[1:len(equity) + 1], name="equity")

    _log_summary(trades_df, starting_capital or config.STARTING_CAPITAL, capital)
    return trades_df, equity_curve


# ── Exit check ────────────────────────────────────────────────────────────────

def _check_exit(bar: pd.Series, position: dict) -> dict | None:
    """
    Check if the current bar closes the trade via SL or TP.

    Convention (conservative):
        Check SL first. If both SL and TP were reached in the same bar,
        we assume SL was hit first (worst case — more realistic).

    LONG trade:
        SL hit if bar Low  <= sl_price  → exit at sl_price
        TP hit if bar High >= tp_price  → exit at tp_price

    SHORT trade:
        SL hit if bar High >= sl_price  → exit at sl_price
        TP hit if bar Low  <= tp_price  → exit at tp_price

    Returns:
        dict with exit_price and reason, or None if trade still open.
    """
    direction = position["direction"]
    sl        = position["sl_price"]
    tp        = position["tp_price"]

    if direction == 1:   # LONG
        if bar["Low"] <= sl:
            return {"exit_price": sl, "reason": "SL"}
        if bar["High"] >= tp:
            return {"exit_price": tp, "reason": "TP"}

    elif direction == -1:   # SHORT
        if bar["High"] >= sl:
            return {"exit_price": sl, "reason": "SL"}
        if bar["Low"] <= tp:
            return {"exit_price": tp, "reason": "TP"}

    return None   # trade still open


# ── Level calculation ─────────────────────────────────────────────────────────

def _calculate_levels(
    fill_price: float,
    atr: float,
    direction: int,
    sl_mult: float,
    tp_mult: float,
) -> tuple[float, float]:
    """
    Calculate SL and TP prices from fill price.

    LONG:   SL = fill - sl_mult×ATR    TP = fill + tp_mult×ATR
    SHORT:  SL = fill + sl_mult×ATR    TP = fill - tp_mult×ATR

    All levels anchored to fill_price. Never to signal_price or bar close.
    This was a critical bug in early versions — wrong anchor destroys R:R.
    """
    sl_distance = sl_mult * atr
    tp_distance = tp_mult * atr

    if direction == 1:   # LONG
        sl_price = fill_price - sl_distance
        tp_price = fill_price + tp_distance
    else:                # SHORT
        sl_price = fill_price + sl_distance
        tp_price = fill_price - tp_distance

    return sl_price, tp_price


# ── Position sizing ───────────────────────────────────────────────────────────

def _calculate_lot_size(
    capital: float,
    atr: float,
    sl_mult: float,
) -> float:
    """
    Calculate lot size so that if SL is hit, we lose exactly RISK_PCT of capital.

        risk_amount  = capital × RISK_PCT
        sl_distance  = sl_mult × ATR   (in price units)
        sl_in_pips   = sl_distance / PIP_SIZE
        lot_size     = risk_amount / (sl_in_pips × PIP_VALUE_per_standard_lot)

    Then round DOWN to the nearest LOT_STEP (conservative).
    """
    risk_amount = capital * config.RISK_PCT
    sl_distance = sl_mult * atr
    sl_in_pips  = sl_distance / config.PIP_SIZE
    raw_lots    = risk_amount / (sl_in_pips * config.PIP_VALUE)

    # Round down to nearest LOT_STEP
    lot_size = (raw_lots // config.LOT_STEP) * config.LOT_STEP
    return round(lot_size, 2)


# ── PnL calculation ───────────────────────────────────────────────────────────

def _calculate_pnl(position: dict, exit_price: float, capital: float) -> dict:
    """
    Calculate P&L in USD and pips for a closed trade.

    LONG:   pip_pnl = (exit_price - entry_price) / PIP_SIZE
    SHORT:  pip_pnl = (entry_price - exit_price) / PIP_SIZE
    dollar_pnl = pip_pnl × PIP_VALUE × lot_size
    """
    direction    = position["direction"]
    entry_price  = position["entry_price"]
    lot_size     = position["lot_size"]

    if direction == 1:   # LONG
        pip_pnl = (exit_price - entry_price) / config.PIP_SIZE
    else:                # SHORT
        pip_pnl = (entry_price - exit_price) / config.PIP_SIZE

    dollar_pnl = pip_pnl * config.PIP_VALUE * lot_size

    return {
        "pip_pnl"   : round(pip_pnl, 1),
        "dollar_pnl": round(dollar_pnl, 2),
    }


# ── Logging summary ───────────────────────────────────────────────────────────

def _log_summary(trades_df: pd.DataFrame, start_capital: float, end_capital: float) -> None:
    if trades_df.empty:
        logger.info("Backtest complete: 0 trades generated")
        return
    n_trades = len(trades_df)
    n_wins   = (trades_df["pnl_usd"] > 0).sum()
    logger.info(
        f"Backtest complete: {n_trades} trades | "
        f"Win rate: {n_wins/n_trades:.1%} | "
        f"Capital: ${start_capital:,.2f} -> ${end_capital:,.2f}"
    )


# ── Parameter resolver ────────────────────────────────────────────────────────

def _resolve_params(params: dict | None) -> dict:
    defaults = {
        "SL_MULT": config.SL_MULT,
        "TP_MULT": config.TP_MULT,
    }
    if params:
        defaults.update(params)
    return defaults