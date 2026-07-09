import numpy as np
import pandas as pd

import config
from strategy.indicators import add_indicators


# ── Public API ────────────────────────────────────────────────────────────────

def generate_signals(df: pd.DataFrame, params: dict | None = None) -> pd.DataFrame:
    """
    Generate LONG/SHORT signals for every bar in df.

    Args:
        df     : OHLCV DataFrame (already cleaned)
        params : optional parameter overrides (used during WFO grid search)
                 falls back to config defaults if not provided

    Returns:
        df with added columns:
            sr_long      (bool) — price near support level
            sr_short     (bool) — price near resistance level
            macd_long    (bool) — bullish MACD divergence
            macd_short   (bool) — bearish MACD divergence
            signal       (int)  — +1 = LONG, -1 = SHORT, 0 = no trade
    """
    p = _resolve_params(params)
    df = add_indicators(df, p)

    n = len(df)
    sr_long   = np.zeros(n, dtype=bool)
    sr_short  = np.zeros(n, dtype=bool)
    macd_long  = np.zeros(n, dtype=bool)
    macd_short = np.zeros(n, dtype=bool)

    sr_lb   = p["SR_LOOKBACK"]
    sr_tol  = p["SR_TOLERANCE"]
    mac_lb  = p["MACD_LOOKBACK"]
    sw_win  = p["SWING_WINDOW"]

    for i in range(max(sr_lb, mac_lb, sw_win * 2), n):
        close = df["Close"].iloc[i]
        atr   = df["atr"].iloc[i]

        if pd.isna(atr) or atr == 0:
            continue

        # ── Signal 1: SR Zone ─────────────────────────────────────────────
        window_slice = df.iloc[max(0, i - sr_lb): i]
        swing_highs  = window_slice["swing_high"].dropna().values
        swing_lows   = window_slice["swing_low"].dropna().values

        near_support    = _near_any_level(close, swing_lows,  atr, sr_tol)
        near_resistance = _near_any_level(close, swing_highs, atr, sr_tol)

        sr_long[i]  = near_support
        sr_short[i] = near_resistance

        # ── Signal 2: MACD Divergence ─────────────────────────────────────
        price_window = df["Close"].iloc[max(0, i - mac_lb): i + 1]
        macd_window  = df["macd_line"].iloc[max(0, i - mac_lb): i + 1]

        macd_long[i]  = _bullish_divergence(price_window, macd_window, sw_win)
        macd_short[i] = _bearish_divergence(price_window, macd_window, sw_win)

    df["sr_long"]    = sr_long
    df["sr_short"]   = sr_short
    df["macd_long"]  = macd_long
    df["macd_short"] = macd_short

    # ── Convergence: both signals must fire ───────────────────────────────
    df["signal"] = 0
    df.loc[df["sr_long"]  & df["macd_long"],  "signal"] = 1   # LONG
    df.loc[df["sr_short"] & df["macd_short"], "signal"] = -1  # SHORT

        # MA200 hard filter — no LONG below MA200, no SHORT above MA200
    df["ma200"] = df["Close"].rolling(200).mean()
    df.loc[(df["signal"] ==  1) & (df["Close"] < df["ma200"]), "signal"] = 0
    df.loc[(df["signal"] == -1) & (df["Close"] > df["ma200"]), "signal"] = 0
    return df


# ── SR Zone helper ────────────────────────────────────────────────────────────

def _near_any_level(
    price: float,
    levels: np.ndarray,
    atr: float,
    tolerance: float,
) -> bool:
    """
    Return True if `price` is within tolerance×ATR of any level in `levels`.

    Plain English:
        We don't require an exact touch of the swing level (prices rarely
        repeat exactly). We allow a zone of ±tolerance×ATR around each level.
        If current price falls inside any of these zones → SR signal fires.
    """
    if len(levels) == 0:
        return False
    distances = np.abs(price - levels)
    return bool(distances.min() <= tolerance * atr)


# ── MACD Divergence helpers ───────────────────────────────────────────────────

def _bullish_divergence(
    price: pd.Series,
    macd: pd.Series,
    swing_window: int,
) -> bool:
    """
    Bullish divergence: price lower low + MACD higher low.

    Plain English:
        Find the two most recent swing lows in the price window.
        If the second (more recent) swing low is LOWER than the first
        BUT the MACD at the second swing low is HIGHER than at the first
        → sellers are losing strength. Buy signal.

    Returns True if bullish divergence detected, False otherwise.
    """
    lows_price, lows_macd = _find_swing_low_pairs(price, macd, swing_window)
    if lows_price is None:
        return False

    price_lower_low  = lows_price[1] < lows_price[0]   # price made new low
    macd_higher_low  = lows_macd[1]  > lows_macd[0]    # MACD did not confirm

    return bool(price_lower_low and macd_higher_low)


def _bearish_divergence(
    price: pd.Series,
    macd: pd.Series,
    swing_window: int,
) -> bool:
    """
    Bearish divergence: price higher high + MACD lower high.

    Plain English:
        Find the two most recent swing highs in the price window.
        If the second (more recent) swing high is HIGHER than the first
        BUT the MACD at the second swing high is LOWER than at the first
        → buyers are losing strength. Sell signal.

    Returns True if bearish divergence detected, False otherwise.
    """
    highs_price, highs_macd = _find_swing_high_pairs(price, macd, swing_window)
    if highs_price is None:
        return False

    price_higher_high = highs_price[1] > highs_price[0]  # price made new high
    macd_lower_high   = highs_macd[1]  < highs_macd[0]   # MACD did not confirm

    return bool(price_higher_high and macd_lower_high)


def _find_swing_low_pairs(
    price: pd.Series,
    macd: pd.Series,
    window: int,
) -> tuple[list | None, list | None]:
    """
    Find the two most recent swing lows in `price` within the current window.
    Returns their price values and corresponding MACD values.
    Returns (None, None) if fewer than 2 swing lows found.
    """
    lows_p, lows_m = [], []
    vals = price.values
    macd_vals = macd.values

    for i in range(window, len(vals) - window):
        neighbourhood = vals[i - window: i + window + 1]
        if vals[i] == neighbourhood.min():
            lows_p.append(vals[i])
            lows_m.append(macd_vals[i])

    if len(lows_p) < 2:
        return None, None
    return lows_p[-2:], lows_m[-2:]  # last two swing lows


def _find_swing_high_pairs(
    price: pd.Series,
    macd: pd.Series,
    window: int,
) -> tuple[list | None, list | None]:
    """
    Find the two most recent swing highs in `price` within the current window.
    Returns their price values and corresponding MACD values.
    Returns (None, None) if fewer than 2 swing highs found.
    """
    highs_p, highs_m = [], []
    vals = price.values
    macd_vals = macd.values

    for i in range(window, len(vals) - window):
        neighbourhood = vals[i - window: i + window + 1]
        if vals[i] == neighbourhood.max():
            highs_p.append(vals[i])
            highs_m.append(macd_vals[i])

    if len(highs_p) < 2:
        return None, None
    return highs_p[-2:], highs_m[-2:]  # last two swing highs


# ── Parameter resolver ────────────────────────────────────────────────────────

def _resolve_params(params: dict | None) -> dict:
    """Merge provided params with config defaults."""
    defaults = {
        "ATR_PERIOD"   : config.ATR_PERIOD,
        "MACD_FAST"    : config.MACD_FAST,
        "MACD_SLOW"    : config.MACD_SLOW,
        "MACD_SIGNAL"  : config.MACD_SIGNAL,
        "MACD_LOOKBACK": config.MACD_LOOKBACK,
        "SR_LOOKBACK"  : config.SR_LOOKBACK,
        "SR_TOLERANCE" : config.SR_TOLERANCE,
        "SWING_WINDOW" : config.SWING_WINDOW,
        "SL_MULT"      : config.SL_MULT,
        "TP_MULT"      : config.TP_MULT,
    }
    if params:
        defaults.update(params)
    return defaults
