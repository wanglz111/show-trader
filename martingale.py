from __future__ import annotations

import logging
import math
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Deque, Dict, Iterable, List, Optional, Tuple

from core.types import BarData, OrderResult, OrderSide, SignalAction, TradeSignal
from strategies.base import BaseStrategy
from strategies.zerolag import zero_lag_from_bars, zero_lag_min_bars

logger = logging.getLogger(__name__)


@dataclass
class PositionState:
    size: float = 0.0
    avg_price: float = 0.0
    levels: int = 0
    entry_timestamp: Optional[datetime] = None
    fills: List[Dict[str, Any]] = field(default_factory=list)

    def reset(self) -> None:
        self.size = 0.0
        self.avg_price = 0.0
        self.levels = 0
        self.entry_timestamp = None
        self.fills.clear()


class MartingaleStrategy(BaseStrategy):
    DEFAULTS: Dict[str, float] = {
        "entry_logic": "ZERO_LAG",
        "take_profit_percent": 5.0,
        "take_profit_min_percent": 2.0,
        "take_profit_decay_hours": 240.0,
        "martingale_trigger": 10.0,
        "martingale_mult": 2.4,
        "base_position_pct": 0.05,
        "fixed_position": False,
        "start_position_size": 10.0,
        "symbol": "ETHUSDT",
        "zero_lag_length": 70,
        "zero_lag_mult": 1.2,
        "lot_size": 0.0,
    }

    def __init__(self, **params: float) -> None:
        merged = {**self.DEFAULTS, **params}
        max_levels_override = merged.pop("max_levels", None)
        if "max_levels" in params:
            logger.warning("Strategy param 'max_levels' is deprecated; configure via risk.max_levels instead.")
        super().__init__(symbol=merged["symbol"], params=merged)
        zero_lag_length = int(merged.get("zero_lag_length", 70))
        history_window = max(600, zero_lag_length * 3 + 10)
        self.history: Deque[BarData] = deque(maxlen=history_window)
        self.position = PositionState()
        self.last_signal: Optional[TradeSignal] = None
        self.trend: int = 0  # -1=空头, 1=多头, 0=无趋势
        self._last_zero_lag_metrics: Dict[str, float] = {}
        self.last_trend_change_ts: Optional[datetime] = None
        lot_size = float(merged.get("lot_size", 0.0) or 0.0)
        flat_tol = merged.get("flat_tolerance")
        if flat_tol is None:
            flat_tol = lot_size
        self.flat_tolerance = max(float(flat_tol or 0.0), 0.0)
        self.max_levels_limit: Optional[int] = None
        self.set_max_levels_limit(max_levels_override)

    # State management ----------------------------------------------------

    def apply_snapshot(
        self,
        size: float,
        avg_price: float,
        levels: int,
        entry_timestamp: Optional[str],
        fills: Optional[List[Dict[str, Any]]] = None,
    ) -> None:
        self.position.size = size
        self.position.avg_price = avg_price
        self.position.levels = levels
        if entry_timestamp:
            ts = entry_timestamp.replace("Z", "+00:00") if entry_timestamp.endswith("Z") else entry_timestamp
            self.position.entry_timestamp = datetime.fromisoformat(ts).astimezone(timezone.utc)
        else:
            self.position.entry_timestamp = None
        self.position.fills = [dict(fill) for fill in (fills or [])]

    def reset(self) -> None:
        super().reset()
        self.history.clear()
        self.position.reset()
        self.last_signal = None
        self.trend = 0
        self.last_trend_change_ts = None

    def seed_history(self, bars: Iterable[BarData]) -> None:
        for bar in bars:
            self.history.append(bar)

    def initialize_trend_from_history(self) -> None:
        """Replay ZeroLag calculations over seeded history to infer the latest trend."""
        if not self.history:
            return
        bars: List[BarData] = list(self.history)
        length = int(self.params.get("zero_lag_length", 70))
        mult = float(self.params.get("zero_lag_mult", 1.2))
        min_bars = zero_lag_min_bars(length)
        incremental: List[BarData] = []
        prev_trend = 0
        last_metrics: Dict[str, float] = {}
        trend_change_ts: Optional[datetime] = None
        for bar in bars:
            incremental.append(bar)
            if len(incremental) < min_bars:
                continue
            prior_trend = prev_trend
            _, metrics, prev_trend = zero_lag_from_bars(incremental, length, mult, prev_trend)
            if metrics:
                last_metrics = metrics
                if prev_trend != prior_trend:
                    trend_change_ts = bar.timestamp
        if last_metrics:
            self._last_zero_lag_metrics = dict(last_metrics)
        self.trend = prev_trend
        self.last_trend_change_ts = trend_change_ts
        if self.trend == 0 and last_metrics:
            latest_close = bars[-1].close
            upper = last_metrics.get("upper")
            lower = last_metrics.get("lower")
            if upper is not None and latest_close >= upper:
                self.trend = 1
                self.last_trend_change_ts = bars[-1].timestamp
            elif lower is not None and latest_close <= lower:
                self.trend = -1
                self.last_trend_change_ts = bars[-1].timestamp

    def set_max_levels_limit(self, limit: Optional[float]) -> None:
        if limit is None:
            self.max_levels_limit = None
            return
        try:
            parsed = int(limit)
        except (TypeError, ValueError):
            logger.warning("Invalid max_levels limit %s; ignoring", limit)
            self.max_levels_limit = None
            return
        self.max_levels_limit = max(0, parsed)

    def zero_lag_snapshot(self) -> Optional[Dict[str, float]]:
        decision, metrics = self._zero_lag_signal()
        if not metrics:
            return None
        snapshot = {**metrics, "signal": decision}
        return snapshot

    def last_zero_lag_metrics(self) -> Optional[Dict[str, float]]:
        if not self._last_zero_lag_metrics:
            return None
        return dict(self._last_zero_lag_metrics)

    def _zero_lag_signal(self) -> Tuple[Optional[int], Dict[str, float]]:
        length = int(self.params.get("zero_lag_length", 70))
        mult = float(self.params.get("zero_lag_mult", 1.2))
        decision, metrics, new_trend = zero_lag_from_bars(self.history, length, mult, self.trend)
        if metrics:
            self._last_zero_lag_metrics = dict(metrics)
            if new_trend != self.trend and self.history:
                self.last_trend_change_ts = self.history[-1].timestamp
            self.trend = new_trend
        else:
            self._last_zero_lag_metrics = {}
        return decision, metrics

    # Trade decision helpers ----------------------------------------------

    def _is_flat_size(self, size: float) -> bool:
        if self.flat_tolerance <= 0:
            return size <= 0
        return abs(size) <= self.flat_tolerance

    def is_flat(self) -> bool:
        """Expose flat check to external callers without duplicating tolerance logic."""
        return self._is_flat()

    def _is_flat(self) -> bool:
        return self._is_flat_size(self.position.size)

    def _should_add_position(self, price: float) -> bool:
        trigger = float(self.params["martingale_trigger"])
        if self._is_flat() or self.position.avg_price <= 0:
            return False
        drop_pct = (price - self.position.avg_price) / self.position.avg_price * 100
        if drop_pct > -trigger:
            return False
        if self.max_levels_limit and self.position.levels >= self.max_levels_limit:
            return False
        return True

    def _current_take_profit(self, ts: datetime) -> float:
        take_profit = float(self.params["take_profit_percent"])
        entry_time = self.position.entry_timestamp
        if not entry_time or ts <= entry_time:
            return take_profit
        tau = float(self.params.get("take_profit_decay_hours", 0.0))
        tp_min = float(self.params.get("take_profit_min_percent", take_profit))
        if tau <= 0:
            return max(tp_min, take_profit)
        elapsed_hours = (ts - entry_time).total_seconds() / 3600
        decayed = take_profit * math.exp(-elapsed_hours / tau)
        return max(tp_min, decayed)

    def take_profit_target(self, ts: Optional[datetime] = None) -> Optional[float]:
        """Expose current decayed take-profit target for monitoring/logging."""
        if ts is None:
            ts = datetime.now(timezone.utc)
        if self._is_flat():
            return None
        return self._current_take_profit(ts)

    def _should_take_profit(self, price: float, ts: datetime) -> bool:
        if self._is_flat() or self.position.avg_price <= 0:
            return False
        profit_pct = (price - self.position.avg_price) / self.position.avg_price * 100
        return profit_pct >= self._current_take_profit(ts)

    def target_entry_quantity(self, cash_balance: float, price: float) -> float:
        if self.params.get("fixed_position"):
            return float(self.params["start_position_size"])
        pct = float(self.params["base_position_pct"])
        notional = cash_balance * pct
        return round(notional / price, 6)

    def next_add_quantity(self) -> float:
        return round(self.position.size * float(self.params["martingale_mult"]), 6)

    # Public API ----------------------------------------------------------

    def on_bar(self, bar: BarData) -> TradeSignal:
        self.history.append(bar)
        trend_signal: Optional[int] = None
        metrics: Dict[str, float] = {}
        logic = self.params["entry_logic"].upper()
        if logic in {"ZERO_LAG", "MACD"}:
            trend_signal, metrics = self._zero_lag_signal()

        price = bar.close
        signal = self._build_trade_signal(price, bar.timestamp, trend_signal, logic)
        self.last_signal = signal
        logger.debug(
            "Signal %s action=%s position=%s price=%.2f metrics=%s",
            self.symbol,
            signal.action,
            self.position,
            price,
            metrics,
        )
        return signal

    def mark_price_breakout(self, price: float, ts: Optional[datetime] = None) -> TradeSignal:
        if ts is None:
            ts = datetime.now(timezone.utc)
        if self.trend != 1:
            return TradeSignal(action=SignalAction.HOLD, info={"reason": "MARK_PRICE_BREAKOUT_IGNORED"})
        signal = self._build_trade_signal(price, ts, trend_signal=1, reason="MARK_PRICE_BREAKOUT")
        self.last_signal = signal
        logger.debug(
            "MarkPrice breakout signal %s action=%s position=%s price=%.2f",
            self.symbol,
            signal.action,
            self.position,
            price,
        )
        return signal

    # 更新：_build_trade_signal 以支持 trend 从 -1 到 1 的转换
    def _build_trade_signal(self, price: float, ts: datetime, trend_signal: Optional[int], reason: str) -> TradeSignal:
        if self._is_flat():
            if trend_signal == 1:
                return TradeSignal(action=SignalAction.ENTER, info={"reason": reason})
            else:
                return TradeSignal(action=SignalAction.HOLD, info={"reason": reason})
        if self._should_take_profit(price, ts):
            return TradeSignal(action=SignalAction.EXIT, info={"reason": "take_profit"})
        if self._should_add_position(price):
            return TradeSignal(action=SignalAction.ADD, info={"level": self.position.levels + 1})
        return TradeSignal(action=SignalAction.HOLD)

    def on_order_fill(self, order: OrderResult) -> None:
        if order.side == OrderSide.BUY:
            previous = self.position.size
            new_level = self.position.levels + 1
            total_cost = self.position.avg_price * previous + (order.avg_price or 0.0) * order.filled_qty
            new_size = previous + order.filled_qty
            self.position.size = new_size
            self.position.avg_price = total_cost / new_size if new_size else 0.0
            self.position.levels = new_level
            if order.timestamp:
                if not self.position.entry_timestamp or self._is_flat_size(previous):
                    self.position.entry_timestamp = order.timestamp
                else:
                    weight_new = order.filled_qty / new_size
                    prev_entry = self.position.entry_timestamp
                    assert prev_entry is not None
                    delta = order.timestamp - prev_entry
                    self.position.entry_timestamp = prev_entry + weight_new * delta
            ts = order.timestamp or datetime.now(timezone.utc)
            if ts.tzinfo is None:
                ts = ts.replace(tzinfo=timezone.utc)
            else:
                ts = ts.astimezone(timezone.utc)
            self.position.fills.append(
                {
                    "timestamp": ts.isoformat(),
                    "quantity": float(order.filled_qty),
                    "price": float(order.avg_price or 0.0),
                    "level": new_level,
                }
            )
        else:
            self.position.reset()
