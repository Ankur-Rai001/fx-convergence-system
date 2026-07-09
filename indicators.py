# =============================================================================
# strategy/indicators.py — FX Convergence System
#
# What this file does (plain English):
#   Three building blocks that every other module depends on:
#
#   1. ATR  — measures how much price moves on average each day
#              Used to set SL/TP distances that adapt to market volatility
#
#   2. MACD — measures the gap between a fast and slow moving average
#              When that gap diverges from price direction = our signal
#
#   3. Swing High / Swing Low detection
#              Finds natural turning points in price history
#              These become our Support/Resistance levels
# =============================================================================

import numpy as np
import pandas as pd


# ── ATR ───────────────────────────────────────────────────────────────────────

def compute_atr(df: pd.DataFrame, period: int) -> pd.Series:
    """
    Average True Range (ATR).

    True Range = largest of:
        (a) High - Low               (normal bar range)
        (b) |High - Previous Close|  (gap up then volatile)
        (c) |Low  - Previous Close|  (gap down then volatile)

    ATR = exponential moving average of True Range over `period` bars.

    Plain English:
        ATR tells you "on average, how many pips does this pair move per day?"
        A higher ATR = wider SL and TP (more room for the trade to breathe).
        A lower ATR  = tighter SL and TP.

    Returns:
        pd.Series of ATR values, NaN for the first `period` bars.
    """
    high  = df["High"]
    low   = df["Low"]
    close = df["Close"]
    prev_close = close.shift(1)

    tr = pd.concat(
        [
            high - low,
            (high - prev_close).abs(),
            (low  - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)

    atr = tr.ewm(span=period, adjust=False).mean()
    return atr


# ── MACD ──────────────────────────────────────────────────────────────────────

def compute_macd(
    df: pd.DataFrame,
    fast: int,
    slow: int,
    signal_period: int,
) -> pd.DataFrame:
    """
    MACD (Moving Average Convergence Divergence).

        Fast EMA    = EMA(close, fast)
        Slow EMA    = EMA(close, slow)
        MACD Line   = Fast EMA - Slow EMA
        Signal Line = EMA(MACD Line, signal_period)
        Histogram   = MACD Line - Signal Line

    Plain English:
        Fast EMA reacts quickly to recent price moves.
        Slow EMA reacts slowly — it represents the longer trend.
        MACD Line = difference between these two = momentum of the trend.
        When MACD makes higher lows while price makes lower lows = bullish divergence.

    Returns:
        DataFrame with columns: macd_line, signal_line, histogram
    """
    close  = df["Close"]
    ema_fast   = close.ewm(span=fast,   adjust=False).mean()
    ema_slow   = close.ewm(span=slow,   adjust=False).mean()
    macd_line  = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal_period, adjust=False).mean()
    histogram   = macd_line - signal_line

    return pd.DataFrame(
        {
            "macd_line"  : macd_line,
            "signal_line": signal_line,
            "histogram"  : histogram,
        },
        index=df.index,
    )


# ── Swing Highs and Lows ──────────────────────────────────────────────────────

def find_swing_highs(df: pd.DataFrame, window: int) -> pd.Series:
    """
    Detect swing highs — bars whose High is the highest in the surrounding window.

    A swing high is a bar where:
        High[i] == max(High[i-window : i+window+1])

    Plain English:
        A swing high is a local peak — a bar that is higher than all neighbours
        within `window` bars on each side. These are natural resistance levels:
        places where price previously turned down.

    Args:
        df     : OHLCV DataFrame
        window : number of bars on each side to compare

    Returns:
        pd.Series with the High price at swing highs, NaN elsewhere.
    """
    high   = df["High"]
    result = pd.Series(np.nan, index=df.index)

    for i in range(window, len(high) - window):
        neighbourhood = high.iloc[i - window : i + window + 1]
        if high.iloc[i] == neighbourhood.max():
            result.iloc[i] = high.iloc[i]

    return result


def find_swing_lows(df: pd.DataFrame, window: int) -> pd.Series:
    """
    Detect swing lows — bars whose Low is the lowest in the surrounding window.

    A swing low is a bar where:
        Low[i] == min(Low[i-window : i+window+1])

    Plain English:
        A swing low is a local trough — a bar that is lower than all neighbours
        within `window` bars on each side. These are natural support levels:
        places where price previously turned up.

    Args:
        df     : OHLCV DataFrame
        window : number of bars on each side to compare

    Returns:
        pd.Series with the Low price at swing lows, NaN elsewhere.
    """
    low    = df["Low"]
    result = pd.Series(np.nan, index=df.index)

    for i in range(window, len(low) - window):
        neighbourhood = low.iloc[i - window : i + window + 1]
        if low.iloc[i] == neighbourhood.min():
            result.iloc[i] = low.iloc[i]

    return result


# ── Convenience: add all indicators to a DataFrame ────────────────────────────

def add_indicators(df: pd.DataFrame, params: dict) -> pd.DataFrame:
    """
    Add ATR, MACD, and swing columns to a OHLCV DataFrame.

    Args:
        df     : raw OHLCV DataFrame
        params : dict with keys matching config (ATR_PERIOD, MACD_FAST, etc.)

    Returns:
        df with new columns: atr, macd_line, signal_line, histogram,
                             swing_high, swing_low
    """
    import config

    df = df.copy()

    # ATR
    df["atr"] = compute_atr(df, period=params.get("ATR_PERIOD", config.ATR_PERIOD))

    # MACD
    macd_df = compute_macd(
        df,
        fast=params.get("MACD_FAST",   config.MACD_FAST),
        slow=params.get("MACD_SLOW",   config.MACD_SLOW),
        signal_period=params.get("MACD_SIGNAL", config.MACD_SIGNAL),
    )
    df["macd_line"]   = macd_df["macd_line"]
    df["signal_line"] = macd_df["signal_line"]
    df["histogram"]   = macd_df["histogram"]

    # Swing levels
    swing_window = params.get("SWING_WINDOW", config.SWING_WINDOW)
    df["swing_high"] = find_swing_highs(df, window=swing_window)
    df["swing_low"]  = find_swing_lows(df,  window=swing_window)

    return df
