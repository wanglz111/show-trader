from __future__ import annotations

import argparse
import csv
import json
from concurrent.futures import ProcessPoolExecutor, as_completed
import random
from itertools import product
from pathlib import Path
from typing import Iterator, List, Dict

import pandas as pd

from backtest_lightweight_chart import run_backtest, ensure_zero_lag_cache
from martingale import MartingaleStrategy


def frange(start: float, stop: float, step: float) -> Iterator[float]:
    val = start
    # Avoid floating drift by rounding to 4 decimals
    while val <= stop + 1e-9:
        yield round(val, 4)
        val = round(val + step, 10)


def combo_filename(
    combo: Dict[str, float], args: Dict[str, float | str | int | bool], window_label: str | None = None
) -> str:
    base = (
        f"{args['symbol']}_tpp{combo['take_profit_percent']}_"
        f"tpmin{combo['take_profit_min_percent']}_"
        f"tpdh{combo['take_profit_decay_hours']}_"
        f"mt{combo['martingale_trigger']}_"
        f"mm{combo['martingale_mult']}.html"
    )
    return f"{window_label}_{base}" if window_label else base


def build_combos() -> List[dict]:
    """参数网格：TP 4-7%，TP 下限 2.5-4%，衰减 240h，触发 7-11%，倍数 1.5-2.2。"""
    take_profit_percent_range = frange(4.0, 7.0, 0.5)
    take_profit_min_percent_range = frange(2.5, 4.0, 0.5)
    take_profit_decay_hours_range = [240.0]
    martingale_trigger_range = frange(7.0, 11.0, 1.0)
    martingale_mult_range = frange(1.5, 2.2, 0.1)

    combos: List[dict] = []
    for take_profit_percent, take_profit_min_percent, take_profit_decay_hours, martingale_trigger, martingale_mult in product(
        take_profit_percent_range,
        take_profit_min_percent_range,
        take_profit_decay_hours_range,
        martingale_trigger_range,
        martingale_mult_range,
    ):
        if take_profit_min_percent > take_profit_percent:
            continue
        combos.append(
            dict(
                take_profit_percent=take_profit_percent,
                take_profit_min_percent=take_profit_min_percent,
                take_profit_decay_hours=take_profit_decay_hours,
                martingale_trigger=martingale_trigger,
                martingale_mult=martingale_mult,
            )
        )
    return combos


SUMMARY_HEADERS = [
    "window_months",
    "chart",
    "take_profit_percent",
    "take_profit_min_percent",
    "take_profit_decay_hours",
    "martingale_trigger",
    "martingale_mult",
    "total_profit",
    "return_pct",
    "final_value",
    "max_drawdown_pct",
    "max_position_size",
    "max_position_value",
    "trade_count",
    "win_rate",
    "profit_factor",
    "max_hold_hours",
    "avg_hold_hours",
]

AGGREGATE_HEADERS = [
    "take_profit_percent",
    "take_profit_min_percent",
    "take_profit_decay_hours",
    "martingale_trigger",
    "martingale_mult",
    "runs",
    "avg_return_pct",
    "avg_avg_hold_hours",
]


REQUIRED_FIELDS = [
    "take_profit_percent",
    "take_profit_min_percent",
    "take_profit_decay_hours",
    "martingale_trigger",
    "martingale_mult",
]


def load_combos_from_file(path: Path) -> List[dict]:
    path = path.expanduser().resolve()
    if not path.exists():
        raise FileNotFoundError(f"Combos file not found: {path}")

    def _cast_row(row: Dict[str, object]) -> dict:
        return {
            "take_profit_percent": float(row["take_profit_percent"]),
            "take_profit_min_percent": float(row["take_profit_min_percent"]),
            "take_profit_decay_hours": float(row["take_profit_decay_hours"]),
            "martingale_trigger": float(row["martingale_trigger"]),
            "martingale_mult": float(row["martingale_mult"]),
        }

    combos: List[dict] = []
    if path.suffix.lower() == ".json":
        raw = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(raw, dict):
            raw = raw.get("combos") or raw.get("params") or []
        for row in raw:
            if not all(k in row for k in REQUIRED_FIELDS):
                continue
            combos.append(_cast_row(row))
    else:
        with path.open(newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                if not row:
                    continue
                if not all(k in row and row[k] not in (None, "") for k in REQUIRED_FIELDS):
                    continue
                combos.append(_cast_row(row))
    if not combos:
        raise ValueError(f"No valid combos found in {path}. Expected fields: {', '.join(REQUIRED_FIELDS)}")
    return combos


def write_summaries(summaries: List[Dict[str, float]], summary_csv: Path, summary_html: Path) -> None:
    if not summaries:
        return
    summary_csv.parent.mkdir(parents=True, exist_ok=True)
    with summary_csv.open("w", encoding="utf-8") as f:
        f.write(",".join(SUMMARY_HEADERS) + "\n")
        for row in summaries:
            f.write(",".join(str(row.get(h, "")) for h in SUMMARY_HEADERS) + "\n")

    summary_html.parent.mkdir(parents=True, exist_ok=True)
    rows_html = "\n".join(
        f"<tr><td>{row['window_months']}</td>"
        f"<td>{row['chart']}</td>"
        f"<td>{row['take_profit_percent']}</td>"
        f"<td>{row['take_profit_min_percent']}</td>"
        f"<td>{row['take_profit_decay_hours']}</td>"
        f"<td>{row['martingale_trigger']}</td>"
        f"<td>{row['martingale_mult']}</td>"
        f"<td>{row['total_profit']:.4f}</td>"
        f"<td>{row['return_pct']:.2f}%</td>"
        f"<td>{row['final_value']:.4f}</td>"
        f"<td>{row['max_drawdown_pct']:.2f}%</td>"
        f"<td>{row['max_position_value']:.4f}</td>"
        f"<td>{row['trade_count']}</td>"
        f"<td>{row['win_rate']:.2f}%</td>"
        f"<td>{row['profit_factor']:.2f}</td>"
        f"<td>{row['max_hold_hours']:.2f}</td>"
        f"<td>{row['avg_hold_hours']:.2f}</td></tr>"
        for row in summaries
    )
    summary_html.write_text(
        f"""<!doctype html>
<html><head><meta charset="utf-8"><title>Batch Summary</title>
<style>
body {{ background:#0f172a; color:#e2e8f0; font-family:'Segoe UI',sans-serif; padding:16px; }}
table {{ width:100%; border-collapse: collapse; }}
th, td {{ padding:8px; border:1px solid rgba(148,163,184,0.3); }}
th {{ background: rgba(148,163,184,0.15); }}
tr:nth-child(even) {{ background: rgba(148,163,184,0.05); }}
</style>
</head>
<body>
<h2>Batch Backtest Summary ({len(summaries)} charts)</h2>
<table>
  <thead>
    <tr>
      <th>Window (m)</th>
      <th>Chart</th>
      <th>TP%</th>
      <th>TP Min%</th>
      <th>TP Decay H</th>
      <th>Trigger%</th>
      <th>Mult</th>
      <th>Total Profit</th>
      <th>Return %</th>
      <th>Final Value</th>
      <th>Max DD %</th>
      <th>Max Pos Value</th>
      <th>Trades</th>
      <th>Win %</th>
      <th>Profit Factor</th>
      <th>Max Hold (h)</th>
      <th>Avg Hold (h)</th>
    </tr>
  </thead>
  <tbody>
    {rows_html}
  </tbody>
</table>
</body></html>
""",
        encoding="utf-8",
    )


def ensure_window_csv(source_csv: Path, months: int, datetime_format: str, outdir: Path) -> Path:
    """
    Slice the source CSV to the last N months and cache it under outdir/windowed_data/.
    """
    source_csv = source_csv.expanduser().resolve()
    outdir = outdir.expanduser().resolve()
    window_dir = outdir / "windowed_data"
    window_dir.mkdir(parents=True, exist_ok=True)
    target = window_dir / f"{source_csv.stem}_last{months}m.csv"

    if target.exists() and target.stat().st_mtime >= source_csv.stat().st_mtime:
        return target

    df = pd.read_csv(source_csv, parse_dates=["datetime"])
    if df.empty:
        raise ValueError(f"Source CSV {source_csv} is empty.")
    cutoff = df["datetime"].max() - pd.DateOffset(months=months)
    sliced = df[df["datetime"] >= cutoff]
    if sliced.empty:
        raise ValueError(f"No data found within last {months} months in {source_csv}.")
    sliced.to_csv(target, index=False)
    return target


def aggregate_combo_stats(rows: List[Dict[str, float]]) -> List[Dict[str, float]]:
    """Average return_pct and avg_hold_hours grouped by parameter combo."""
    grouped: Dict[tuple, dict] = {}
    for row in rows:
        key = (
            row["take_profit_percent"],
            row["take_profit_min_percent"],
            row["take_profit_decay_hours"],
            row["martingale_trigger"],
            row["martingale_mult"],
        )
        stats = grouped.setdefault(key, {"runs": 0, "return_sum": 0.0, "avg_hold_sum": 0.0})
        stats["runs"] += 1
        stats["return_sum"] += float(row.get("return_pct", 0.0))
        stats["avg_hold_sum"] += float(row.get("avg_hold_hours", 0.0))
    aggregates: List[Dict[str, float]] = []
    for key, stats in grouped.items():
        runs = stats["runs"] or 1
        aggregates.append(
            {
                "take_profit_percent": key[0],
                "take_profit_min_percent": key[1],
                "take_profit_decay_hours": key[2],
                "martingale_trigger": key[3],
                "martingale_mult": key[4],
                "runs": runs,
                "avg_return_pct": stats["return_sum"] / runs,
                "avg_avg_hold_hours": stats["avg_hold_sum"] / runs,
            }
        )
    return sorted(aggregates, key=lambda r: r["avg_return_pct"], reverse=True)


def write_aggregate_csv(rows: List[Dict[str, float]], path: Path) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        f.write(",".join(AGGREGATE_HEADERS) + "\n")
        for row in rows:
            f.write(",".join(str(row.get(h, "")) for h in AGGREGATE_HEADERS) + "\n")


def parse_random_slices(raw: str | None) -> List[tuple[int, int]]:
    """
    Parse --random-slices input like "6:2,12:2,24:2" into [(6,2), (12,2), (24,2)].
    """
    if not raw:
        return []
    pairs: List[tuple[int, int]] = []
    for chunk in raw.split(","):
        chunk = chunk.strip()
        if not chunk:
            continue
        if ":" not in chunk:
            raise ValueError(f"Invalid random slice spec '{chunk}', expected months:count")
        months_s, count_s = chunk.split(":", 1)
        months, count = int(months_s), int(count_s)
        if months <= 0 or count <= 0:
            raise ValueError("Random slice months and count must be positive.")
        pairs.append((months, count))
    return pairs


def parse_coverage_window(raw: str | None) -> tuple[int, int] | None:
    """
    Parse --coverage-window input like "12:6" into (12, 6).
    """
    if not raw:
        return None
    if ":" not in raw:
        raise ValueError("Invalid coverage window, expected months:count, e.g. 12:6")
    months_s, count_s = raw.split(":", 1)
    months, count = int(months_s), int(count_s)
    if months <= 0 or count <= 0:
        raise ValueError("Coverage window months and count must be positive.")
    return months, count


def build_random_windows(
    source_csv: Path,
    datetime_format: str,
    base_months: int,
    slices: List[tuple[int, int]],
    outdir: Path,
    seed: int | None = None,
) -> List[dict]:
    """
    Randomly slice the last `base_months` of data into the requested slice plan.
    Returns metadata dicts with keys: label, months, csv_path.
    """
    if not slices:
        return []
    df = pd.read_csv(source_csv, parse_dates=["datetime"])
    if df.empty:
        raise ValueError(f"Source CSV {source_csv} is empty.")
    df = df.sort_values("datetime")
    if base_months > 0:
        cutoff = df["datetime"].max() - pd.DateOffset(months=base_months)
        df = df[df["datetime"] >= cutoff]
    if df.empty:
        raise ValueError(f"No data found within last {base_months} months in {source_csv}.")
    rng = random.Random(seed)
    window_dir = outdir.expanduser().resolve() / "windowed_random"
    window_dir.mkdir(parents=True, exist_ok=True)
    windows: List[dict] = []
    for months, count in slices:
        max_start = df["datetime"].max() - pd.DateOffset(months=months)
        eligible = df[df["datetime"] <= max_start]
        if eligible.empty:
            raise ValueError(f"Not enough data to build a {months}-month slice.")
        for idx in range(1, count + 1):
            start_dt = eligible.iloc[rng.randrange(len(eligible))]["datetime"]
            end_dt = start_dt + pd.DateOffset(months=months)
            sliced = df[(df["datetime"] >= start_dt) & (df["datetime"] <= end_dt)]
            if sliced.empty:
                raise ValueError(f"Random slice {months}m#{idx} produced no rows.")
            label = f"{months}m-r{idx}"
            target = window_dir / f"{source_csv.stem}_{label}.csv"
            sliced.to_csv(target, index=False)
            windows.append({"label": label, "months": months, "csv_path": target})
    return windows


def build_spanning_windows(
    source_csv: Path,
    datetime_format: str,
    months: int,
    count: int,
    outdir: Path,
) -> List[dict]:
    """
    Evenly space fixed-length windows across the full dataset timeframe.
    Ensures we cover the earliest and latest data points.
    """
    if count <= 0 or months <= 0:
        return []
    df = pd.read_csv(source_csv, parse_dates=["datetime"])
    if df.empty:
        raise ValueError(f"Source CSV {source_csv} is empty.")
    df = df.sort_values("datetime")
    start_dt = df["datetime"].min()
    end_dt = df["datetime"].max()
    latest_start = end_dt - pd.DateOffset(months=months)
    # If the dataset is shorter than the window, clamp to start.
    if latest_start < start_dt:
        latest_start = start_dt
    if count == 1 or latest_start == start_dt:
        start_points = [latest_start]
    else:
        total_seconds = (latest_start - start_dt).total_seconds()
        step_seconds = total_seconds / (count - 1)
        start_points = [start_dt + pd.Timedelta(seconds=step_seconds * idx) for idx in range(count)]

    window_dir = outdir.expanduser().resolve() / "windowed_spanning"
    window_dir.mkdir(parents=True, exist_ok=True)
    windows: List[dict] = []
    for idx, start in enumerate(start_points, start=1):
        end = start + pd.DateOffset(months=months)
        sliced = df[(df["datetime"] >= start) & (df["datetime"] <= end)]
        if sliced.empty:
            continue
        label = f"{months}m-span{idx}"
        target = window_dir / f"{source_csv.stem}_{label}.csv"
        sliced.to_csv(target, index=False)
        windows.append({"label": label, "months": months, "csv_path": target})
    if not windows:
        raise ValueError(f"No spanning windows generated from {source_csv} with {months}m x {count}.")
    return windows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Batch-generate martingale backtest charts for parameter grids.")
    parser.add_argument("--data", default="ETHUSDT_1h.csv", help="Path to CSV (datetime,open,high,low,close,volume with header).")
    parser.add_argument("--outdir", default="charts", help="Directory to write HTML charts.")
    parser.add_argument("--cash", type=float, default=10000.0)
    parser.add_argument("--commission", type=float, default=0.0005)
    parser.add_argument("--symbol", default="ETHUSDT")
    parser.add_argument("--zero-lag-length", type=int, default=MartingaleStrategy.DEFAULTS["zero_lag_length"])
    parser.add_argument("--zero-lag-mult", type=float, default=MartingaleStrategy.DEFAULTS["zero_lag_mult"])
    parser.add_argument("--max-charts", type=int, default=50, help="Safety cap to avoid generating thousands of charts in one go.")
    parser.add_argument("--datetime-format", default="%Y-%m-%d %H:%M:%S")
    parser.add_argument("--summary-csv", default="charts/summary.csv", help="Where to write the summary CSV.")
    parser.add_argument("--summary-html", default="charts/summary.html", help="Where to write the summary HTML.")
    parser.add_argument("--aggregate-csv", default="charts/aggregate_summary.csv", help="Where to write averaged metrics per combo.")
    parser.add_argument(
        "--combos-file",
        help="CSV/JSON with explicit combos. Columns/keys: take_profit_percent,take_profit_min_percent,"
        "take_profit_decay_hours,martingale_trigger,martingale_mult",
    )
    parser.add_argument(
        "--windows",
        nargs="+",
        type=int,
        default=[36],
        help="Window sizes in months to backtest (default: 36).",
    )
    parser.add_argument(
        "--checkpoint-every",
        type=int,
        default=1,
        help="Rewrite summary/aggregate every N completed backtests to avoid losing progress on interruption.",
    )
    parser.add_argument("--workers", type=int, default=None, help="Number of parallel processes (default: cpu_count())")
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Skip combos whose chart already exists (also load existing summary.csv if present).",
    )
    parser.add_argument(
        "--random-slices",
        help="Random slice plan like '6:2,12:2,24:2' (months:count). When set, use random slices instead of fixed windows.",
    )
    parser.add_argument("--random-base-months", type=int, default=36, help="Use only the most recent N months as the pool for random slices.")
    parser.add_argument("--random-seed", type=int, default=42, help="Seed for random slice reproducibility.")
    parser.add_argument(
        "--coverage-window",
        help="Evenly spaced fixed windows across full dataset, e.g. '12:6' for six 12-month spans (overrides --windows / --random-slices).",
    )
    return parser.parse_args()


def run_combo(
    idx: int, combo: Dict[str, float], args: Dict[str, float | str | int | bool], window_label: str
) -> Dict[str, float | str]:
    csv_path = Path(args["csv_path"]).expanduser().resolve()
    outdir = Path(args["outdir"]).expanduser().resolve()
    filename = combo_filename(combo, args, window_label=window_label)
    chart_path = outdir / filename
    martingale_params = dict(
        symbol=args["symbol"],
        zero_lag_length=args["zero_lag_length"],
        zero_lag_mult=args["zero_lag_mult"],
        zero_lag_cache_path=args["zero_lag_cache_path"],
        zero_lag_cache_count=args["zero_lag_cache_count"],
        martingale_trigger=combo["martingale_trigger"],
        martingale_mult=combo["martingale_mult"],
        base_position_pct=MartingaleStrategy.DEFAULTS["base_position_pct"],
        start_position_size=MartingaleStrategy.DEFAULTS["start_position_size"],
        fixed_position=MartingaleStrategy.DEFAULTS["fixed_position"],
        take_profit_percent=combo["take_profit_percent"],
        take_profit_min_percent=combo["take_profit_min_percent"],
        take_profit_decay_hours=combo["take_profit_decay_hours"],
    )
    summary = run_backtest(
        csv_path=csv_path,
        chart_path=chart_path,
        martingale_params=martingale_params,
        cash=float(args["cash"]),
        commission=float(args["commission"]),
        datetime_format=str(args["datetime_format"]),
    )
    return {
        "idx": idx,
        "window_months": args["window_months"],
        "chart": chart_path.name,
        "take_profit_percent": combo["take_profit_percent"],
        "take_profit_min_percent": combo["take_profit_min_percent"],
        "take_profit_decay_hours": combo["take_profit_decay_hours"],
        "martingale_trigger": combo["martingale_trigger"],
        "martingale_mult": combo["martingale_mult"],
        "total_profit": summary["total_profit"],
        "return_pct": summary["return_pct"],
        "final_value": summary["final_value"],
        "max_drawdown_pct": summary["max_drawdown_pct"],
        "max_position_size": summary["max_position_size"],
        "max_position_value": summary["max_position_value"],
        "trade_count": summary["trade_count"],
        "win_rate": summary["win_rate"],
        "profit_factor": summary["profit_factor"],
        "max_hold_hours": summary["max_hold_hours"],
        "avg_hold_hours": summary["avg_hold_hours"],
    }


def main() -> None:
    args = parse_args()
    csv_path = Path(args.data).expanduser().resolve()
    outdir = Path(args.outdir).expanduser().resolve()
    outdir.mkdir(parents=True, exist_ok=True)
    summary_csv = Path(args.summary_csv).expanduser().resolve()
    summary_html = Path(args.summary_html).expanduser().resolve()
    agg_path = Path(args.aggregate_csv).expanduser().resolve()

    combos = load_combos_from_file(Path(args.combos_file)) if args.combos_file else build_combos()
    summaries: List[Dict[str, float]] = []
    args_dict: Dict[str, float | str | int | bool] = vars(args).copy()
    existing_charts: dict[str, set[str]] = {}

    def flush_outputs() -> None:
        if not summaries:
            return
        write_summaries(summaries, summary_csv, summary_html)
        aggregates = aggregate_combo_stats(summaries)
        if aggregates:
            write_aggregate_csv(aggregates, agg_path)

    if args.resume and summary_csv.exists():
        with summary_csv.open(newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            has_window = reader.fieldnames and "window_months" in reader.fieldnames
            for row in reader:
                try:
                    window_months = int(row["window_months"]) if has_window else args.windows[0]
                    parsed = {
                        "window_months": window_months,
                        "chart": row["chart"],
                        "take_profit_percent": float(row["take_profit_percent"]),
                        "take_profit_min_percent": float(row["take_profit_min_percent"]),
                        "take_profit_decay_hours": float(row["take_profit_decay_hours"]),
                        "martingale_trigger": float(row["martingale_trigger"]),
                        "martingale_mult": float(row["martingale_mult"]),
                        "total_profit": float(row["total_profit"]),
                        "return_pct": float(row["return_pct"]),
                        "final_value": float(row["final_value"]),
                        "max_drawdown_pct": float(row.get("max_drawdown_pct", 0.0)),
                        "max_position_size": float(row.get("max_position_size", 0.0)),
                        "max_position_value": float(row.get("max_position_value", 0.0)),
                        "trade_count": float(row.get("trade_count", 0.0)),
                        "win_rate": float(row.get("win_rate", 0.0)),
                        "profit_factor": float(row.get("profit_factor", 0.0)),
                        "max_hold_hours": float(row.get("max_hold_hours", 0.0)),
                        "avg_hold_hours": float(row.get("avg_hold_hours", 0.0)),
                    }
                    summaries.append(parsed)
                    label = f"{window_months}m"
                    existing_charts.setdefault(label, set()).add(row["chart"])
                except (KeyError, ValueError):
                    continue
        print(f"Loaded {len(summaries)} rows from existing summary for resume.")

    if not combos:
        print("No combos to run (empty grid or combos file).")
        return

    coverage_plan = parse_coverage_window(args.coverage_window)
    random_slice_plan = parse_random_slices(args.random_slices)
    if coverage_plan:
        cov_months, cov_count = coverage_plan
        windows = build_spanning_windows(csv_path, args.datetime_format, cov_months, cov_count, outdir)
    elif random_slice_plan:
        windows = build_random_windows(csv_path, args.datetime_format, args.random_base_months, random_slice_plan, outdir, seed=args.random_seed)
    else:
        windows = [
            {"label": f"{months}m", "months": months, "csv_path": ensure_window_csv(csv_path, months, args.datetime_format, outdir)}
            for months in args.windows
        ]
    if not windows:
        print("No windows to process (check --windows or --random-slices).")
        return

    workers = args.workers or None
    for window in windows:
        months = int(window["months"])
        window_label = str(window["label"])
        window_csv = Path(window["csv_path"]).expanduser().resolve()
        window_outdir = outdir / window_label
        window_outdir.mkdir(parents=True, exist_ok=True)
        cache_path = window_outdir / f"zerolag_len{args.zero_lag_length}_mult{args.zero_lag_mult}_{window_label}.bin"
        cache_count = ensure_zero_lag_cache(window_csv, args.datetime_format, args.zero_lag_length, args.zero_lag_mult, cache_path)

        args_dict_window = args_dict | {
            "csv_path": str(window_csv),
            "outdir": str(window_outdir),
            "window_months": months,
            "zero_lag_cache_path": str(cache_path),
            "zero_lag_cache_count": cache_count,
        }

        combos_to_run = combos
        skipped = 0
        if args.resume:
            filtered_combos: List[dict] = []
            for combo in combos:
                filename = combo_filename(combo, args_dict_window, window_label=window_label)
                if filename in existing_charts.get(window_label, set()) or (window_outdir / filename).exists():
                    skipped += 1
                    existing_charts.setdefault(window_label, set()).add(filename)
                    continue
                filtered_combos.append(combo)
            combos_to_run = filtered_combos

        if args.max_charts and len(combos_to_run) > args.max_charts:
            combos_to_run = combos_to_run[: args.max_charts]
            print(f"[{window_label}] Limiting to first {args.max_charts} combos to avoid huge batch.")

        if not combos_to_run:
            print(f"[{window_label}] No combos to run (skipped {skipped} via resume).")
            continue

        print(f"[{window_label}] Running {len(combos_to_run)} backtests with {workers or 'cpu_count()'} workers...")
        pool: ProcessPoolExecutor | None = None
        try:
            pool = ProcessPoolExecutor(max_workers=workers)
            future_to_idx = {
                pool.submit(run_combo, idx, combo, args_dict_window, window_label): idx
                for idx, combo in enumerate(combos_to_run, start=1)
            }
            for future in as_completed(future_to_idx):
                idx = future_to_idx[future]
                try:
                    row = future.result()
                    summaries.append(row)
                    print(
                        f"[{window_label} {idx}/{len(combos_to_run)}] {row['chart']} done. "
                        f"Profit {row['total_profit']:.2f}, Return {row['return_pct']:.2f}%"
                    )
                    if len(summaries) % max(args.checkpoint_every, 1) == 0:
                        flush_outputs()
                except Exception as exc:
                    print(f"[{window_label} {idx}/{len(combos_to_run)}] job failed: {exc}")
        except KeyboardInterrupt:
            if pool:
                pool.shutdown(wait=False, cancel_futures=True)
            print("Interrupted by user; cancelling pending backtests and shutting down workers.")
            return
        finally:
            if pool:
                pool.shutdown(wait=True, cancel_futures=True)

    if summaries:
        flush_outputs()
        print(f"Summary CSV/HTML written to {summary_csv} / {summary_html}")
        print(f"Aggregate averages written to {agg_path}")


if __name__ == "__main__":
    main()
