from __future__ import annotations
import math
from collections import deque
from typing import Deque, Dict, List, Optional, Sequence, Tuple

from core.types import BarData


def zero_lag_min_bars(length: int) -> int:
    lag = max(1, int((length - 1) / 2))
    return max(length * 3 + 1, lag + 2)


# --- Wilder RMA ---
def _rma(values: Sequence[float], period: int) -> List[float]:
    if period <= 0 or not values:
        return []
    rma = [values[0]]
    alpha = 1.0 / period
    for v in values[1:]:
        rma.append(rma[-1] + alpha * (v - rma[-1]))
    return rma


# --- ATR (TradingView equivalent) ---
def _atr(highs: Sequence[float], lows: Sequence[float], closes: Sequence[float], period: int) -> List[float]:
    trs = []
    prev_close = closes[0]
    for i in range(len(closes)):
        if i == 0:
            tr = highs[i] - lows[i]
        else:
            tr = max(
                highs[i] - lows[i],
                abs(highs[i] - prev_close),
                abs(lows[i] - prev_close),
            )
        trs.append(tr)
        prev_close = closes[i]
    return _rma(trs, period)


# --- Rolling max (highest) ---
def _rolling_max(values: Sequence[float], window: int) -> List[float]:
    if window <= 0:
        return [math.nan] * len(values)
    result = []
    dq: Deque[Tuple[int, float]] = deque()

    for i, v in enumerate(values):
        while dq and dq[-1][1] <= v:
            dq.pop()
        dq.append((i, v))

        while dq and dq[0][0] <= i - window:
            dq.popleft()

        if i + 1 < window:
            result.append(math.nan)
        else:
            result.append(dq[0][1])
    return result


# --- EMA ---
def _ema_series(values: Sequence[float], span: int) -> List[float]:
    if not values:
        return []
    k = 2 / (span + 1)
    ema = [values[0]]
    for v in values[1:]:
        ema.append(v * k + ema[-1] * (1 - k))
    return ema


def zero_lag_from_bars(
    bars: Sequence[BarData],
    length: int,
    mult: float,
    prev_trend: int,
) -> Tuple[Optional[int], Dict[str, float], int]:

    max_window = length * 3 + 1
    window_bars = bars[-max_window:] if len(bars) > max_window else bars

    closes = [b.close for b in window_bars]
    highs = [b.high for b in window_bars]
    lows = [b.low for b in window_bars]

    if len(closes) == 0:
        return None, {}, prev_trend

    lag = max(1, int((length - 1) / 2))
    min_bars = max_window

    if len(closes) < min_bars:
        return None, {}, prev_trend

    # --- Zero-lag adjusted price ---
    adjusted = []
    for i, c in enumerate(closes):
        if i < lag:
            adjusted.append(c)
        else:
            adjusted.append(c + (c - closes[i - lag]))

    zlema = _ema_series(adjusted, length)

    # ATR & highest ATR
    atr = _atr(highs, lows, closes, length)
    highest_atr = _rolling_max(atr, length * 3)

    last = len(closes) - 1
    prev = last - 1

    zlema_curr = zlema[last]
    atr_prev = highest_atr[prev]

    if not (math.isfinite(zlema_curr) and math.isfinite(atr_prev)):
        return None, {}, prev_trend

    volatility = atr_prev * mult
    upper = zlema_curr + volatility
    lower = zlema_curr - volatility

    # --- TradingView equivalent trend logic ---
    trend = prev_trend
    c = closes[last]
    p = closes[prev]

    if p <= upper and c > upper:   # crossover
        trend = 1
    elif p >= lower and c < lower: # crossunder
        trend = -1

    breakout = (trend != prev_trend)

    metrics = {
        "zlema": float(zlema_curr),
        "upper": float(upper),
        "lower": float(lower),
        "trend": trend,
        "breakout": breakout,
    }

    decision = 1 if (breakout and trend == 1) else None
    return decision, metrics, trend
