"""
自动下载 Binance 真实 K 线数据
默认拉取 ETH/USDT 最近 N 个月 1m 数据，文件名中会带月份范围
"""

import ccxt
import pandas as pd
from datetime import datetime, timedelta
import time

# 创建交易所实例
exchange = ccxt.binance()

# === 参数设置 ===
symbol = 'BTC/USDT'
timeframe = '1m'
limit = 1000  # Binance 每次最多返回1000条
months = 62    # 拉取最近 months 个月数据
symbol_code = symbol.replace("/", "")

# 计算起始时间戳
end_time = exchange.milliseconds()
end_dt = datetime.utcnow()
start_dt = end_dt - timedelta(days=30 * months)
start_time = int(start_dt.timestamp() * 1000)
timeframe_ms = int(exchange.parse_timeframe(timeframe) * 1000)

print(f" 正在从 {exchange.id} 下载 {symbol} {timeframe} 数据（{start_dt:%Y-%m-%d} -> {end_dt:%Y-%m-%d}）...")

all_data = []
since = start_time
while True:
    try:
        bars = exchange.fetch_ohlcv(symbol, timeframe=timeframe, since=since, limit=limit)
        if not bars:
            break
        all_data += bars
        print(f"  下载进度: {len(all_data)} 条", end='\r')

        # 下一轮从最后一条时间后开始
        since = bars[-1][0] + timeframe_ms
        time.sleep(0.8)  # 避免API限速
        if since >= end_time:
            break
    except Exception as e:
        print(" 出错，重试中...", e)
        time.sleep(3)
        continue

# === 转换为 DataFrame ===
df = pd.DataFrame(all_data, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
df['datetime'] = pd.to_datetime(df['timestamp'], unit='ms')
df = df[['datetime', 'open', 'high', 'low', 'close', 'volume']]

# 保存CSV，文件名包含时间范围
filename = f"{symbol_code}_{timeframe}_last{months}m_{start_dt:%Y%m%d}-{end_dt:%Y%m%d}.csv"
df.to_csv(filename, index=False, float_format='%.2f')
bars_per_day = max(int((24 * 60) / max(timeframe_ms // (60 * 1000), 1)), 1)
print(f"\n 已保存 {filename}，共 {len(df)} 条记录（约 {len(df)//bars_per_day} 天数据）")
