# Martingale Backtest + Lightweight-Charts

快速用 Backtrader 跑 martingale 策略，输出轻量级 K 线网页（含买/卖标记、均价线、每轮利润表），支持批量参数扫描并行跑，生成总览表。

## 环境
- Python 3.9+（本地 `python3`）
- 依赖：`backtrader`

安装依赖：
```bash
pip3 install backtrader
```

## 数据要求
CSV 带表头，列顺序：
```
datetime,open,high,low,close,volume
```
默认时间格式 `%Y-%m-%d %H:%M:%S`，可用 `--datetime-format` 调整。

## 单次回测 & 生成图
```bash
python3 backtest_lightweight_chart.py \
  --data ETHUSDT_1h.csv \
  --chart chart.html \
  --cash 20000 \
  --commission 0.0005 \
  --symbol ETHUSDT \
  --take-profit-percent 7.0 \
  --take-profit-min-percent 4.75 \
  --take-profit-decay-hours 144 \
  --martingale-trigger 11.0 \
  --martingale-mult 2.0
```
打开 `chart.html` 查看：K 线 + 买入/卖出标记 + 价格线 + 每轮 open/add/exit 的利润表（含总利润）。

## 批量参数扫描（并行跑满 CPU）
`batch_backtest.py` 生成多组图和汇总表：
```bash
python3 batch_backtest.py \
  --data ETHUSDT_1h.csv \
  --outdir charts \
  --max-charts 0 \            # 0 表示不限制；默认 50 防炸机器
  --workers 14 \              # 并行进程数，默认 cpu_count()
  --summary-csv charts/summary.csv \
  --summary-html charts/summary.html
```
参数网格（可在 `build_combos()` 里改）：
- take_profit_percent: 5–7.5 步长 0.5
- take_profit_min_percent: 3–5 步长 0.5
- take_profit_decay_hours: 120–240 步长 30
- martingale_trigger: 8–12 步长 1
- martingale_mult: 1.6–2.4 步长 0.2

产物：
- `charts/*.html`：每组参数的轻量图（含交易标记、均价线、每轮利润表、总利润）。
- `charts/summary.csv`：参数+总利润+收益率+最终权益。
- `charts/summary.html`：同上表格的网页版。

### 随机窗口切片 + 均值指标
从最近 36 个月里随机切出多段窗口（仍然复用 zerolag 缓存），跑完后会额外生成参数组合的平均收益率、平均持仓时间汇总。
```bash
python3 batch_backtest.py \
  --data BNBUSDT_1m_last37m_20221115-20251129.csv \
  --symbol BNBUSDT \
  --outdir charts_bnb \
  --max-charts 0 \
  --workers 14 \
  --random-slices 6:2,12:2,24:2 \   # 各切出 2 段，共 6 段
  --random-base-months 36 \         # 只从最近 36 个月里抽样
  --aggregate-csv charts_bnb/aggregate_summary.csv
```
结果：
- `charts_bnb/<窗口标签>/*.html`：每段随机窗口的图，zerolag 为该窗口复用缓存。
- `charts_bnb/summary.csv` / `summary.html`：每段窗口的单次结果。
- `charts_bnb/aggregate_summary.csv`：按参数组合聚合后的平均收益率、平均持仓时间。

## 关键文件
- `martingale.py`：策略逻辑。
- `core/types.py` / `strategies/*.py`：轻量依赖。
- `backtest_lightweight_chart.py`：单次回测 + 图表导出。
- `batch_backtest.py`：参数扫描并行跑，输出汇总。

## 常见问题
- **太慢**：用 `--workers` 开多进程，或减小 `--max-charts`、缩短数据窗口。
- **时间格式不匹配**：调整 `--datetime-format`。
- **Python 命令不存在**：使用 `python3`。***
