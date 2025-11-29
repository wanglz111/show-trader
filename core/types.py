from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, Optional


class OrderSide(Enum):
    BUY = "BUY"
    SELL = "SELL"


class SignalAction(Enum):
    ENTER = "ENTER"
    EXIT = "EXIT"
    ADD = "ADD"
    HOLD = "HOLD"


@dataclass
class BarData:
    timestamp: datetime
    open: float
    high: float
    low: float
    close: float
    volume: float = 0.0


@dataclass
class TradeSignal:
    action: SignalAction
    info: Dict[str, Any] = field(default_factory=dict)


@dataclass
class OrderResult:
    side: OrderSide
    filled_qty: float
    avg_price: float
    timestamp: Optional[datetime] = None
