from __future__ import annotations

import argparse
import csv
import mmap
import struct
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List

import backtrader as bt

from core.types import BarData, OrderResult, OrderSide, SignalAction
from martingale import MartingaleStrategy
from strategies.zerolag import zero_lag_from_bars, zero_lag_stream


def _to_utc_timestamp(dt: datetime) -> int:
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return int(dt.timestamp())


def _load_bars(csv_path: Path, dt_format: str) -> List[BarData]:
    bars: List[BarData] = []
    with csv_path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            ts = datetime.strptime(row["datetime"], dt_format)
            bars.append(
                BarData(
                    timestamp=ts,
                    open=float(row["open"]),
                    high=float(row["high"]),
                    low=float(row["low"]),
                    close=float(row["close"]),
                    volume=float(row.get("volume", 0.0) or 0.0),
                )
            )
    return bars


def _precompute_zero_lag_series(
    bars: List[BarData], length: int, mult: float
) -> List[Dict[str, object] | None]:
    precomputed: List[Dict[str, object] | None] = []
    for decision, metrics, trend in zero_lag_stream(bars, length, mult, prev_trend=0):
        if metrics:
            precomputed.append({"decision": decision, "metrics": metrics, "trend": trend})
        else:
            precomputed.append(None)
    return precomputed


def _write_zero_lag_cache_file(
    bars: List[BarData], length: int, mult: float, cache_path: Path
) -> int:
    """
    Write zero-lag metrics to a fixed-width binary file to avoid RAM blowup.
    Record layout: valid(1 byte), decision(int8, 127=none), trend(int8), upper(float64), lower(float64), zlema(float64)
    """
    struct_fmt = struct.Struct("<Bbbddd")
    with cache_path.open("wb") as f:
        count = 0
        for decision, metrics, trend in zero_lag_stream(bars, length, mult, prev_trend=0):
            if metrics:
                dec_byte = 127 if decision is None else int(decision)
                data = struct_fmt.pack(
                    1,
                    dec_byte,
                    int(trend),
                    float(metrics.get("upper", 0.0)),
                    float(metrics.get("lower", 0.0)),
                    float(metrics.get("zlema", 0.0)),
                )
            else:
                data = struct_fmt.pack(0, 127, 0, 0.0, 0.0, 0.0)
            f.write(data)
            count += 1
    return count


def _zero_lag_record_size() -> int:
    return struct.Struct("<Bbbddd").size


def ensure_zero_lag_cache(
    csv_path: Path, datetime_format: str, length: int, mult: float, cache_path: Path
) -> int:
    if cache_path.exists():
        size = cache_path.stat().st_size
        rec_size = _zero_lag_record_size()
        if size % rec_size == 0:
            return size // rec_size
    bars = _load_bars(csv_path, datetime_format)
    return _write_zero_lag_cache_file(bars, length, mult, cache_path)

class MartingaleBacktrader(bt.Strategy):
    """
    Backtrader adapter that drives the martingale strategy and records data for charting.
    """

    params = (
        ("martingale_params", None),
    )

    def __init__(self) -> None:
        kwargs = self.params.martingale_params or {}
        self.martingale = MartingaleStrategy(**kwargs)
        self.candles: List[Dict[str, float]] = []
        self.buy_markers: List[Dict[str, float]] = []
        self.sell_markers: List[Dict[str, float]] = []
        # Track each round of buy/add/exit for profit summary
        self._cycle_qty: float = 0.0
        self._cycle_cost: float = 0.0
        self._cycle_open_ts: int | None = None
        self.cycles: List[Dict[str, float]] = []
        # Equity/position tracking
        self.equity_points: List[float] = []
        self.max_equity_peak: float = 0.0
        self.max_drawdown_pct: float = 0.0
        self.max_position_size: float = 0.0
        self.max_position_value: float = 0.0

    def next(self) -> None:
        dt = self.data.datetime.datetime(0)
        bar = BarData(
            timestamp=dt,
            open=float(self.data.open[0]),
            high=float(self.data.high[0]),
            low=float(self.data.low[0]),
            close=float(self.data.close[0]),
            volume=float(self.data.volume[0]),
        )
        signal = self.martingale.on_bar(bar)
        self.candles.append(
            {
                "time": _to_utc_timestamp(dt),
                "open": bar.open,
                "high": bar.high,
                "low": bar.low,
                "close": bar.close,
            }
        )

        price = bar.close
        # Track position/equity stats on every bar
        self.equity_points.append(float(self.broker.getvalue()))
        self.max_equity_peak = max(self.max_equity_peak, self.equity_points[-1])
        if self.max_equity_peak:
            drawdown_pct = (self.max_equity_peak - self.equity_points[-1]) / self.max_equity_peak * 100.0
            self.max_drawdown_pct = max(self.max_drawdown_pct, drawdown_pct)
        pos_size = abs(float(self.position.size))
        self.max_position_size = max(self.max_position_size, pos_size)
        self.max_position_value = max(self.max_position_value, pos_size * price)

        cash = self.broker.cash
        if signal.action == SignalAction.ENTER:
            qty = self.martingale.target_entry_quantity(cash, price)
            if qty > 0:
                self.buy(size=qty)
        elif signal.action == SignalAction.ADD:
            qty = self.martingale.next_add_quantity()
            if qty > 0:
                self.buy(size=qty)
        elif signal.action == SignalAction.EXIT and self.position.size > 0:
            self.sell(size=self.position.size)

    def notify_order(self, order) -> None:  # type: ignore[override]
        if order.status != order.Completed:
            return
        ts = bt.num2date(order.executed.dt)
        side = OrderSide.BUY if order.isbuy() else OrderSide.SELL
        result = OrderResult(
            side=side,
            filled_qty=abs(order.executed.size),
            avg_price=order.executed.price,
            timestamp=ts,
        )
        if order.isbuy():
            self.martingale.on_order_fill(result)
            if self._cycle_qty <= 0:
                self._cycle_open_ts = _to_utc_timestamp(ts)
                self._cycle_cost = 0.0
                self._cycle_qty = 0.0
            qty = abs(order.executed.size)
            cost = order.executed.price * qty
            self._cycle_qty += qty
            self._cycle_cost += cost
            self.buy_markers.append(
                {
                    "time": _to_utc_timestamp(ts),
                    "price": round(order.executed.price, 4),
                    "quantity": round(abs(order.executed.size), 4),
                    "level": self.martingale.position.levels,
                }
            )
        else:
            fills_snapshot = [dict(f) for f in self.martingale.position.fills]
            qty = abs(order.executed.size)
            proceeds = order.executed.price * qty
            profit = proceeds - self._cycle_cost
            self.cycles.append(
                {
                    "open_time": self._cycle_open_ts or _to_utc_timestamp(ts),
                    "close_time": _to_utc_timestamp(ts),
                    "quantity": round(self._cycle_qty, 4),
                    "cost": round(self._cycle_cost, 4),
                    "proceeds": round(proceeds, 4),
                    "profit": round(profit, 4),
                    "fills": fills_snapshot,
                }
            )
            self.martingale.on_order_fill(result)
            self._cycle_cost = 0.0
            self._cycle_qty = 0.0
            self._cycle_open_ts = None
            self.sell_markers.append(
                {
                    "time": _to_utc_timestamp(ts),
                    "price": round(order.executed.price, 4),
                    "quantity": round(abs(order.executed.size), 4),
                }
            )


def _build_datafeed(csv_path: Path, dt_format: str) -> bt.feeds.GenericCSVData:
    return bt.feeds.GenericCSVData(
        dataname=str(csv_path),
        dtformat=dt_format,
        timeframe=bt.TimeFrame.Minutes,
        datetime=0,
        open=1,
        high=2,
        low=3,
        close=4,
        volume=5,
        openinterest=-1,
        headers=True,
    )


def _write_chart(
    output: Path,
    candles: List[Dict[str, float]],
    buys: List[Dict[str, float]],
    sells: List[Dict[str, float]],
    cycles: List[Dict[str, float]],
) -> None:
    payload = {"candles": candles, "buys": buys, "sells": sells, "cycles": cycles}
    html = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <title>Martingale Backtest</title>
  <style>
    body {{ margin: 0; padding: 0; background: #0f172a; color: #e2e8f0; font-family: 'Segoe UI', sans-serif; }}
    #chart {{ width: 100vw; height: 80vh; }}
    .legend {{ padding: 12px 16px; }}
    .legend span {{ margin-right: 12px; }}
  </style>
  <script src="https://unpkg.com/lightweight-charts@4.1.1/dist/lightweight-charts.standalone.production.js"></script>
</head>
<body>
  <div id="chart"></div>
  <div class="legend">
    <span>Buy markers show qty @ price</span>
    <span>Total candles: {len(candles)}</span>
    <span>Buy count: {len(buys)}</span>
    <span>Sell count: {len(sells)}</span>
    <span>Cycle count: {len(cycles)}</span>
  </div>
  <div id="summary"></div>
  <script>
    const DATA = {json.dumps(payload)};
    const container = document.getElementById('chart');
    const chart = LightweightCharts.createChart(container, {{
      layout: {{ background: {{ type: 'solid', color: '#0f172a' }}, textColor: '#cbd5e1' }},
      grid: {{ vertLines: {{ color: 'rgba(148,163,184,0.15)' }}, horzLines: {{ color: 'rgba(148,163,184,0.15)' }} }},
      timeScale: {{ timeVisible: true, secondsVisible: false }},
      rightPriceScale: {{ borderVisible: false }},
      crosshair: {{ mode: LightweightCharts.CrosshairMode.Magnet }},
    }});

    const candleSeries = chart.addCandlestickSeries({{
      upColor: '#22c55e',
      downColor: '#ef4444',
      borderVisible: false,
      wickUpColor: '#22c55e',
      wickDownColor: '#ef4444',
    }});
    candleSeries.setData(DATA.candles);

    const buyMarkers = DATA.buys.map(buy => ({{
      time: buy.time,
      position: 'belowBar',
      color: '#22c55e',
      shape: 'arrowUp',
      text: `L${{buy.level}} qty ${{buy.quantity}} @ $${{buy.price}}`,
    }}));
    const sellMarkers = DATA.sells.map(sell => ({{
      time: sell.time,
      position: 'aboveBar',
      color: '#f97316',
      shape: 'arrowDown',
      text: `Sell qty ${{sell.quantity}} @ $${{sell.price}}`,
    }}));
    candleSeries.setMarkers([...buyMarkers, ...sellMarkers]);

    // Draw horizontal lines at each buy price for quick reference
    DATA.buys.forEach(buy => {{
      candleSeries.createPriceLine({{
        price: buy.price,
        color: 'rgba(34,197,94,0.4)',
        lineWidth: 1,
        lineStyle: LightweightCharts.LineStyle.Dashed,
        axisLabelVisible: true,
        title: `Buy qty ${{buy.quantity}} @ $${{buy.price}}`,
      }});
    }});
    DATA.sells.forEach(sell => {{
      candleSeries.createPriceLine({{
        price: sell.price,
        color: 'rgba(249,115,22,0.4)',
        lineWidth: 1,
        lineStyle: LightweightCharts.LineStyle.Dotted,
        axisLabelVisible: true,
        title: `Sell qty ${{sell.quantity}} @ $${{sell.price}}`,
      }});
    }});

    // Profit table
    const formatTs = (t) => new Date(t * 1000).toLocaleString();
    const formatFillTs = (t) => t ? new Date(t).toLocaleString() : '-';
    const totalProfit = DATA.cycles.reduce((acc, c) => acc + (c.profit || 0), 0);
    const rows = DATA.cycles.map((c, idx) => {{
      const fillDetails = (c.fills || []).map((f, fidx) => {{
        const ts = formatFillTs(f.timestamp);
        const qty = f.quantity ?? '';
        const price = f.price ?? '';
        const level = f.level ?? (fidx + 1);
        const cost = (qty && price) ? (qty * price).toFixed(4) : '';
        return 'L' + level + ': ' + ts + ' qty ' + qty + ' cost ' + cost + ' @ $' + price;
      }}).join('<br>');
      return `
      <tr>
        <td>${{idx + 1}}</td>
        <td>${{formatTs(c.open_time)}}</td>
        <td>${{formatTs(c.close_time)}}</td>
        <td>${{c.quantity}}</td>
        <td>${{fillDetails}}</td>
        <td>${{c.cost}}</td>
        <td>${{c.proceeds}}</td>
        <td style="color: ${{c.profit >= 0 ? '#22c55e' : '#ef4444'}}">${{c.profit}}</td>
      </tr>
    `;
    }}).join('');
    document.getElementById('summary').innerHTML = `
      <style>
        #summary {{ padding: 12px 16px; font-size: 14px; }}
        #summary table {{ border-collapse: collapse; width: 100%; color: #e2e8f0; }}
        #summary th, #summary td {{ border: 1px solid rgba(148,163,184,0.25); padding: 6px 8px; text-align: left; }}
        #summary th {{ background: rgba(148,163,184,0.1); }}
      </style>
      <div><strong>总利润: ${{totalProfit.toFixed(4)}}</strong></div>
      <table>
        <thead>
          <tr>
            <th>#</th>
            <th>开仓时间</th>
            <th>平仓时间</th>
            <th>数量</th>
            <th>开仓/加仓明细</th>
            <th>成本</th>
            <th>收入</th>
            <th>利润</th>
          </tr>
        </thead>
        <tbody>${{rows}}</tbody>
      </table>
    `;

    const resizeObserver = new ResizeObserver(entries => {{
      for (const entry of entries) {{
        chart.applyOptions({{ width: entry.contentRect.width, height: entry.contentRect.height }});
      }}
    }});
    resizeObserver.observe(container);
  </script>
</body>
</html>
"""
    output.write_text(html, encoding="utf-8")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run martingale strategy with Backtrader and export a lightweight-charts HTML.")
    parser.add_argument("--data", required=True, help="CSV file with columns: datetime,open,high,low,close,volume (with header).")
    parser.add_argument("--datetime-format", default="%Y-%m-%d %H:%M:%S", help="Python datetime.strptime format for the CSV datetime column.")
    parser.add_argument("--cash", type=float, default=10000.0, help="Starting cash.")
    parser.add_argument("--commission", type=float, default=0.0001, help="Commission (fractional).")
    parser.add_argument("--chart", default="chart.html", help="Output HTML file.")
    parser.add_argument("--symbol", default="ETHUSDT", help="Symbol name used by the strategy.")
    parser.add_argument("--zero-lag-length", type=int, default=60, help="Zero-lag length parameter.")
    parser.add_argument("--zero-lag-mult", type=float, default=1.2, help="Zero-lag multiplier parameter.")
    parser.add_argument("--martingale-trigger", type=float, default=8.0, help="Percent drop to trigger add position.")
    parser.add_argument("--martingale-mult", type=float, default=1.6, help="Position size multiplier when averaging down.")
    parser.add_argument(
        "--base-position-pct",
        type=float,
        default=None,
        help="Base position percent of cash for first entry (defaults to cost.py formula, rounded down to 4 decimals).",
    )
    parser.add_argument("--start-position-size", type=float, default=10.0, help="Fixed start size when fixed_position is true.")
    parser.add_argument("--fixed-position", action="store_true", help="Use fixed start position size instead of pct-of-cash.")
    parser.add_argument("--take-profit-percent", type=float, default=MartingaleStrategy.DEFAULTS["take_profit_percent"])
    parser.add_argument("--take-profit-min-percent", type=float, default=MartingaleStrategy.DEFAULTS["take_profit_min_percent"])
    parser.add_argument("--take-profit-decay-hours", type=float, default=MartingaleStrategy.DEFAULTS["take_profit_decay_hours"])
    return parser.parse_args()


def run_backtest(
    csv_path: Path,
    chart_path: Path,
    martingale_params: Dict[str, float],
    cash: float,
    commission: float,
    datetime_format: str,
) -> Dict[str, float]:
    zero_lag_length = int(martingale_params.get("zero_lag_length", MartingaleStrategy.DEFAULTS["zero_lag_length"]))
    zero_lag_mult = float(martingale_params.get("zero_lag_mult", MartingaleStrategy.DEFAULTS["zero_lag_mult"]))
    martingale_mult = float(martingale_params.get("martingale_mult", MartingaleStrategy.DEFAULTS["martingale_mult"]))
    martingale_trigger = float(martingale_params.get("martingale_trigger", MartingaleStrategy.DEFAULTS["martingale_trigger"]))
    cache_path = martingale_params.get("zero_lag_cache_path")
    cache_count = martingale_params.get("zero_lag_cache_count")
    if cache_path:
        cache_path = Path(cache_path)
        cache_count = ensure_zero_lag_cache(csv_path, datetime_format, zero_lag_length, zero_lag_mult, cache_path)
    else:
        cache_path = chart_path.with_suffix(".zlcache.bin")
        cache_count = ensure_zero_lag_cache(csv_path, datetime_format, zero_lag_length, zero_lag_mult, cache_path)
    base_position_pct = martingale_params.get("base_position_pct")
    if base_position_pct is None:
        base_position_pct = MartingaleStrategy.compute_default_base_position_pct(martingale_mult, martingale_trigger)
    martingale_params = {
        **martingale_params,
        "zero_lag_cache_path": str(cache_path),
        "zero_lag_cache_count": cache_count,
        "precomputed_zero_lag": None,  # avoid duplicating in RAM
        "base_position_pct": base_position_pct,
    }

    cerebro = bt.Cerebro()
    datafeed = _build_datafeed(csv_path, datetime_format)
    cerebro.adddata(datafeed)
    cerebro.broker.setcash(cash)
    cerebro.broker.setcommission(commission=commission)

    cerebro.addstrategy(MartingaleBacktrader, martingale_params=martingale_params)

    [result] = cerebro.run()
    _write_chart(chart_path, result.candles, result.buy_markers, result.sell_markers, result.cycles)

    total_profit = sum(c.get("profit", 0.0) for c in result.cycles)
    final_value = cerebro.broker.getvalue()
    cycle_durations = [(c["close_time"] - c["open_time"]) for c in result.cycles if c.get("close_time") and c.get("open_time")]
    max_hold_hours = max(cycle_durations) / 3600 if cycle_durations else 0.0
    avg_hold_hours = (sum(cycle_durations) / len(cycle_durations) / 3600) if cycle_durations else 0.0
    wins = [p for p in (c.get("profit", 0.0) for c in result.cycles) if p > 0]
    losses = [-p for p in (c.get("profit", 0.0) for c in result.cycles) if p < 0]
    profit_factor = (sum(wins) / sum(losses)) if losses else float("inf") if wins else 0.0
    win_rate = (len(wins) / len(result.cycles) * 100.0) if result.cycles else 0.0
    return {
        "chart": str(chart_path),
        "total_profit": total_profit,
        "final_value": final_value,
        "return_pct": (total_profit / cash * 100.0) if cash else 0.0,
        "max_drawdown_pct": result.max_drawdown_pct,
        "max_position_size": result.max_position_size,
        "max_position_value": result.max_position_value,
        "trade_count": len(result.cycles),
        "win_rate": win_rate,
        "profit_factor": profit_factor,
        "max_hold_hours": max_hold_hours,
        "avg_hold_hours": avg_hold_hours,
    }


def main() -> None:
    args = _parse_args()
    csv_path = Path(args.data).expanduser().resolve()
    chart_path = Path(args.chart).expanduser().resolve()

    martingale_params = dict(
        symbol=args.symbol,
        zero_lag_length=args.zero_lag_length,
        zero_lag_mult=args.zero_lag_mult,
        martingale_trigger=args.martingale_trigger,
        martingale_mult=args.martingale_mult,
        base_position_pct=(
            args.base_position_pct
            if args.base_position_pct is not None
            else MartingaleStrategy.compute_default_base_position_pct(args.martingale_mult, args.martingale_trigger)
        ),
        start_position_size=args.start_position_size,
        fixed_position=args.fixed_position,
        take_profit_percent=args.take_profit_percent if hasattr(args, "take_profit_percent") else MartingaleStrategy.DEFAULTS["take_profit_percent"],
        take_profit_min_percent=args.take_profit_min_percent if hasattr(args, "take_profit_min_percent") else MartingaleStrategy.DEFAULTS["take_profit_min_percent"],
        take_profit_decay_hours=args.take_profit_decay_hours if hasattr(args, "take_profit_decay_hours") else MartingaleStrategy.DEFAULTS["take_profit_decay_hours"],
    )

    summary = run_backtest(
        csv_path=csv_path,
        chart_path=chart_path,
        martingale_params=martingale_params,
        cash=args.cash,
        commission=args.commission,
        datetime_format=args.datetime_format,
    )
    print(f"Chart written to {chart_path}")
    print(f"Total profit: {summary['total_profit']:.4f} | Return: {summary['return_pct']:.2f}% | Final value: {summary['final_value']:.4f}")


if __name__ == "__main__":
    main()
