PAIRS      = ["EURUSD=X","AUDUSD=X","NZDUSD=X"]
START_DATE = "2015-01-01"
END_DATE   = "2024-12-31"
INTERVAL   = "1d"                      # daily bars
DATA_DIR   = "data/cache"              # local CSV cache folder

# ── Indicators ────────────────────────────────────────────────────────────────
ATR_PERIOD   = 14                      # ATR lookback (bars)
MACD_FAST    = 12                      # MACD fast EMA period
MACD_SLOW    = 26                      # MACD slow EMA period
MACD_SIGNAL  = 9                       # MACD signal line EMA period
MACD_LOOKBACK = 40                     # bars to scan for divergence

SR_LOOKBACK  = 200                     # bars to scan for swing levels
SR_TOLERANCE = 0.65                   # SR zone = within TOLERANCE x ATR of swing level
SWING_WINDOW = 5                       # bars each side for swing high/low detection

# ── Exit ──────────────────────────────────────────────────────────────────────
SL_MULT = 1.0                          # Stop Loss  = SL_MULT  x ATR from fill price
TP_MULT = 2.0                          # Take Profit = TP_MULT x ATR from fill price
# TP/SL ratio integrity check: TP_MULT / SL_MULT must equal 2.0 always
SHORT_COOLDOWN_BARS = 15    # bars to block new SHORT after a SHORT SL hit


# ── Risk & Position Sizing ────────────────────────────────────────────────────
STARTING_CAPITAL = 10000.0            # USD
RISK_PCT         = 0.015               # 1.5% of capital risked per trade
MIN_LOT          = 0.01                # minimum micro lot
LOT_STEP         = 0.01                # lot size rounding step
PIP_SIZE         = 0.0001              # 1 pip in price units (EURUSD, GBPUSD, AUDUSD)
PIP_VALUE        = 10.0                # USD per pip per standard lot (1.0 lot)
MAX_POSITIONS    = 1                   # max open trades per pair at any time

# ── Walk-Forward Optimization ─────────────────────────────────────────────────
WFO_N_FOLDS    = 6                     # number of WFO folds
WFO_OOS_MONTHS = 6                     # OOS window size (months)

# ── WFO Parameter Grid ────────────────────────────────────────────────────────
# Every combination of these values is tested during IS optimization.
# Total = 3 x 3 x 2 x 2 x 2 = 72 combinations per fold.
PARAM_GRID = {
    "SL_MULT"      : [0.8, 1.0, 1.2],
    "TP_MULT"      : [1.5, 2.0, 2.5],
    "SR_LOOKBACK"  : [100, 200],
    "MACD_LOOKBACK": [20, 30],
    "SR_TOLERANCE" : [0.3, 0.5],
}

# ── Gate Criteria ─────────────────────────────────────────────────────────────
GATE_PF         = 2.0                  # minimum Profit Factor
GATE_SORTINO    = 1.5                  # minimum Sortino ratio
GATE_MAX_DD     = 0.20                 # maximum drawdown (20%)
GATE_MIN_FOLDS  = 4                    # minimum profitable folds out of WFO_N_FOLDS

# ── Reports ───────────────────────────────────────────────────────────────────
REPORTS_DIR = "reports/output"
