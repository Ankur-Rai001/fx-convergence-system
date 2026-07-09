# =============================================================================
# data/fetcher.py — FX Convergence System
#
# What this file does (plain English):
#   1. Downloads daily OHLCV data from Yahoo Finance for each FX pair
#   2. Applies sanity checks (bad OHLC, missing bars, large price jumps)
#   3. Saves to a local CSV cache so we don't re-download every run
#   4. Returns a clean pandas DataFrame per pair
# =============================================================================

import logging
from pathlib import Path

import pandas as pd
import yfinance as yf

import config

logger = logging.getLogger(__name__)


# ── Public API ────────────────────────────────────────────────────────────────

def load_all_pairs(refresh: bool = False) -> dict[str, pd.DataFrame]:
    """
    Load OHLCV data for all pairs defined in config.PAIRS.

    Args:
        refresh: If True, delete cache and re-download from Yahoo Finance.

    Returns:
        dict mapping pair ticker -> cleaned DataFrame (OHLCV, DatetimeIndex)
    """
    data: dict[str, pd.DataFrame] = {}
    for pair in config.PAIRS:
        try:
            data[pair] = _load_pair(pair, refresh=refresh)
            logger.info(f"{pair}: loaded {len(data[pair])} bars "
                        f"({data[pair].index[0].date()} to {data[pair].index[-1].date()})")
        except Exception as exc:
            logger.error(f"{pair}: failed to load — {exc}")
    return data


# ── Internal helpers ──────────────────────────────────────────────────────────

def _load_pair(pair: str, refresh: bool = False) -> pd.DataFrame:
    """Download or load from cache a single FX pair."""
    cache_path = _cache_path(pair)

    if cache_path.exists() and not refresh:
        logger.info(f"{pair}: reading from cache {cache_path}")
        df = pd.read_csv(cache_path, index_col=0, parse_dates=True)
        df.index = pd.to_datetime(df.index)
        return df

    logger.info(f"{pair}: downloading from Yahoo Finance ...")
    raw = yf.download(
        pair,
        start=config.START_DATE,
        end=config.END_DATE,
        interval=config.INTERVAL,
        auto_adjust=True,
        progress=False,
    )

    if raw.empty:
        raise ValueError(f"Yahoo Finance returned no data for {pair}")

    # yfinance sometimes returns MultiIndex columns — flatten them
    if isinstance(raw.columns, pd.MultiIndex):
        raw.columns = raw.columns.get_level_values(0)

    df = raw[["Open", "High", "Low", "Close", "Volume"]].copy()
    df = df.dropna(how="all")
    df.index = pd.to_datetime(df.index)
    df.index.name = "Date"

    df = _sanity_checks(df, pair)
    df = _fill_missing(df, pair)

    Path(config.DATA_DIR).mkdir(parents=True, exist_ok=True)
    df.to_csv(cache_path)
    logger.info(f"{pair}: saved to cache — {len(df)} clean bars")
    return df


def _sanity_checks(df: pd.DataFrame, pair: str) -> pd.DataFrame:
    """
    Remove / flag bars that violate basic OHLC rules.

    Rules applied:
        1. High must be >= max(Open, Close)      — otherwise data is corrupt
        2. Low  must be <= min(Open, Close)       — otherwise data is corrupt
        3. Any price <= 0                         — impossible, remove
        4. Daily move > 10% from previous close   — flag as anomaly (keep but log)
    """
    n_start = len(df)

    bad_high  = df["High"] < df[["Open", "Close"]].max(axis=1)
    bad_low   = df["Low"]  > df[["Open", "Close"]].min(axis=1)
    bad_price = (df[["Open", "High", "Low", "Close"]] <= 0).any(axis=1)

    # Daily move flag (log only — FX can have large moves on NFP etc.)
    prev_close  = df["Close"].shift(1)
    daily_move  = (df["Close"] - prev_close).abs() / prev_close
    large_moves = daily_move > 0.10
    if large_moves.any():
        logger.warning(
            f"{pair}: {large_moves.sum()} bars with >10% daily move "
            f"— inspect dates: {df.index[large_moves].tolist()}"
        )

    bad_mask = bad_high | bad_low | bad_price
    if bad_mask.any():
        logger.warning(
            f"{pair}: removing {bad_mask.sum()} bars with OHLC violations"
        )
    df = df[~bad_mask].copy()

    logger.info(f"{pair}: sanity checks — {n_start} in, {len(df)} out")
    return df


def _fill_missing(df: pd.DataFrame, pair: str) -> pd.DataFrame:
    """
    Forward-fill missing trading days up to a maximum of 3 consecutive days.
    Beyond 3 = likely a data error or exchange closure — those bars are dropped.

    FX is nearly 24/5 so gaps > 3 days are suspicious.
    """
    # Reindex to business day calendar to surface gaps
    full_idx = pd.date_range(df.index.min(), df.index.max(), freq="B")
    df = df.reindex(full_idx)
    df.index.name = "Date"

    # Count consecutive NaN streaks
    is_nan   = df["Close"].isna()
    streak   = is_nan.groupby((~is_nan).cumsum()).cumsum()
    too_long = streak > 3                         # gap longer than 3 days

    # Forward-fill gaps of 1–3 days, drop the rest
    df = df.ffill(limit=3)
    df = df[~too_long].copy()
    df = df.dropna(how="any")

    return df


def _cache_path(pair: str) -> Path:
    """Return deterministic cache file path for a pair."""
    safe_name = pair.replace("=", "_")
    return Path(config.DATA_DIR) / f"{safe_name}_{config.START_DATE}_{config.END_DATE}.csv"
