from __future__ import annotations

from typing import Any, Dict


class BaseStrategy:
    """Minimal stub of the real BaseStrategy used by the martingale logic."""

    def __init__(self, symbol: str, params: Dict[str, Any]) -> None:
        self.symbol = symbol
        self.params = params

    def reset(self) -> None:
        """Hook for clearing state; the martingale strategy overrides as needed."""
        return
