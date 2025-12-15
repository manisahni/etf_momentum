"""
ES Futures Multi-Timeframe Test

Test Kalman, Hull MA, and Donchian on:
- 15-minute bars (native)
- 30-minute bars (resampled)
- 60-minute bars (resampled)

The Gaussian Filter that WORKED was on 15-min, so let's validate
that 15-min is actually the right timeframe.
"""

import pandas as pd
import numpy as np
import sqlite3
from dataclasses import dataclass
from typing import List, Optional
import warnings
warnings.filterwarnings('ignore')

DB_PATH = '/Users/nish_macbook/development/trading/emini-futures/data/emini_futures_15min.db'


@dataclass
class Trade:
    entry_time: object
    entry_price: float
    direction: str
    exit_time: Optional[object] = None
    exit_price: Optional[float] = None
    pnl_pct: Optional[float] = None


def load_es_data() -> pd.DataFrame:
    """Load ES futures 15-min data."""
    conn = sqlite3.connect(DB_PATH)
    query = "SELECT timestamp, open, high, low, close, volume FROM es_15min_bars ORDER BY timestamp"
    df = pd.read_sql_query(query, conn)
    conn.close()
    # Parse timestamps with UTC to avoid mixed timezone issues
    df['timestamp'] = pd.to_datetime(df['timestamp'], utc=True)
    df = df.set_index('timestamp')
    # Ensure we have a proper DatetimeIndex
    df.index = pd.DatetimeIndex(df.index)
    return df


def resample_ohlcv(df: pd.DataFrame, rule: str) -> pd.DataFrame:
    """Resample OHLCV data to higher timeframe."""
    resampled = df.resample(rule).agg({
        'open': 'first',
        'high': 'max',
        'low': 'min',
        'close': 'last',
        'volume': 'sum'
    }).dropna()
    return resampled


def hull_ma(prices: pd.Series, period: int = 20) -> pd.Series:
    """Hull Moving Average."""
    half_period = max(1, int(period / 2))
    sqrt_period = max(1, int(np.sqrt(period)))

    def wma(series, n):
        weights = np.arange(1, n + 1)
        return series.rolling(n).apply(lambda x: np.dot(x, weights) / weights.sum(), raw=True)

    wma_half = wma(prices, half_period)
    wma_full = wma(prices, period)
    raw_hma = 2 * wma_half - wma_full
    hma = wma(raw_hma, sqrt_period)
    return hma


def kalman_filter(prices: pd.Series) -> pd.DataFrame:
    """Simple Kalman filter."""
    n = len(prices)
    x_est = np.zeros(n)
    velocity = np.zeros(n)
    x_est[0] = prices.iloc[0]
    p_est = 1.0

    for i in range(1, n):
        x_pred = x_est[i-1]
        p_pred = p_est + 0.01
        K = p_pred / (p_pred + 1.0)
        x_est[i] = x_pred + K * (prices.iloc[i] - x_pred)
        p_est = (1 - K) * p_pred
        velocity[i] = x_est[i] - x_est[i-1]

    return pd.DataFrame({'estimate': x_est, 'velocity': velocity}, index=prices.index)


def calculate_atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """Calculate ATR."""
    high_low = df['high'] - df['low']
    high_close = abs(df['high'] - df['close'].shift())
    low_close = abs(df['low'] - df['close'].shift())
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    return tr.rolling(period).mean()


def backtest_kalman(df: pd.DataFrame, vel_thresh: float = 0.5,
                    atr_stop: float = 1.5, atr_target: float = 4.0) -> List[Trade]:
    """Backtest Kalman filter strategy."""
    kalman = kalman_filter(df['close'])
    df = df.copy()
    df['velocity'] = kalman['velocity']
    df['velocity_prev'] = df['velocity'].shift(1)
    df['atr'] = calculate_atr(df)
    df['volume_ma'] = df['volume'].rolling(20).mean()
    df['vol_ok'] = df['volume'] > df['volume_ma']

    df['long_signal'] = (df['velocity'] > vel_thresh) & (df['velocity_prev'] <= vel_thresh)
    df['short_signal'] = (df['velocity'] < -vel_thresh) & (df['velocity_prev'] >= -vel_thresh)

    trades = []
    position = None

    for i in range(20, len(df)):
        bar = df.iloc[i]
        atr = bar['atr']
        if pd.isna(atr) or atr == 0:
            continue

        if position is not None:
            t = trades[-1]
            if position == 'long':
                stop = t.entry_price - atr_stop * atr
                target = t.entry_price + atr_target * atr
                if bar['low'] <= stop:
                    t.exit_time, t.exit_price = bar.name, stop
                    t.pnl_pct = (stop - t.entry_price) / t.entry_price * 100
                    position = None
                elif bar['high'] >= target:
                    t.exit_time, t.exit_price = bar.name, target
                    t.pnl_pct = (target - t.entry_price) / t.entry_price * 100
                    position = None
                elif bar['short_signal']:
                    t.exit_time, t.exit_price = bar.name, bar['close']
                    t.pnl_pct = (bar['close'] - t.entry_price) / t.entry_price * 100
                    trades.append(Trade(bar.name, bar['close'], 'short'))
                    position = 'short'
            else:
                stop = t.entry_price + atr_stop * atr
                target = t.entry_price - atr_target * atr
                if bar['high'] >= stop:
                    t.exit_time, t.exit_price = bar.name, stop
                    t.pnl_pct = (t.entry_price - stop) / t.entry_price * 100
                    position = None
                elif bar['low'] <= target:
                    t.exit_time, t.exit_price = bar.name, target
                    t.pnl_pct = (t.entry_price - target) / t.entry_price * 100
                    position = None
                elif bar['long_signal']:
                    t.exit_time, t.exit_price = bar.name, bar['close']
                    t.pnl_pct = (t.entry_price - bar['close']) / t.entry_price * 100
                    trades.append(Trade(bar.name, bar['close'], 'long'))
                    position = 'long'
        else:
            if bar['long_signal']:
                trades.append(Trade(bar.name, bar['close'], 'long'))
                position = 'long'
            elif bar['short_signal']:
                trades.append(Trade(bar.name, bar['close'], 'short'))
                position = 'short'

    if position and trades:
        t = trades[-1]
        if t.exit_time is None:
            t.exit_time = df.index[-1]
            t.exit_price = df.iloc[-1]['close']
            t.pnl_pct = ((t.exit_price - t.entry_price) / t.entry_price * 100
                        if t.direction == 'long' else
                        (t.entry_price - t.exit_price) / t.entry_price * 100)

    return trades


def backtest_hull(df: pd.DataFrame, period: int = 50,
                  atr_stop: float = 1.5, atr_target: float = 4.0) -> List[Trade]:
    """Backtest Hull MA strategy."""
    df = df.copy()
    df['hma'] = hull_ma(df['close'], period)
    df['hma_slope'] = df['hma'].diff()
    df['hma_prev_slope'] = df['hma_slope'].shift(1)
    df['atr'] = calculate_atr(df)
    df['volume_ma'] = df['volume'].rolling(20).mean()
    df['vol_ok'] = df['volume'] > df['volume_ma']

    df['long_signal'] = (df['hma_slope'] > 0) & (df['hma_prev_slope'] <= 0) & df['vol_ok']
    df['short_signal'] = (df['hma_slope'] < 0) & (df['hma_prev_slope'] >= 0) & df['vol_ok']

    trades = []
    position = None

    for i in range(max(period, 20), len(df)):
        bar = df.iloc[i]
        atr = bar['atr']
        if pd.isna(atr) or atr == 0:
            continue

        if position is not None:
            t = trades[-1]
            if position == 'long':
                stop = t.entry_price - atr_stop * atr
                target = t.entry_price + atr_target * atr
                if bar['low'] <= stop:
                    t.exit_time, t.exit_price = bar.name, stop
                    t.pnl_pct = (stop - t.entry_price) / t.entry_price * 100
                    position = None
                elif bar['high'] >= target:
                    t.exit_time, t.exit_price = bar.name, target
                    t.pnl_pct = (target - t.entry_price) / t.entry_price * 100
                    position = None
                elif bar['short_signal']:
                    t.exit_time, t.exit_price = bar.name, bar['close']
                    t.pnl_pct = (bar['close'] - t.entry_price) / t.entry_price * 100
                    trades.append(Trade(bar.name, bar['close'], 'short'))
                    position = 'short'
            else:
                stop = t.entry_price + atr_stop * atr
                target = t.entry_price - atr_target * atr
                if bar['high'] >= stop:
                    t.exit_time, t.exit_price = bar.name, stop
                    t.pnl_pct = (t.entry_price - stop) / t.entry_price * 100
                    position = None
                elif bar['low'] <= target:
                    t.exit_time, t.exit_price = bar.name, target
                    t.pnl_pct = (t.entry_price - target) / t.entry_price * 100
                    position = None
                elif bar['long_signal']:
                    t.exit_time, t.exit_price = bar.name, bar['close']
                    t.pnl_pct = (t.entry_price - bar['close']) / t.entry_price * 100
                    trades.append(Trade(bar.name, bar['close'], 'long'))
                    position = 'long'
        else:
            if bar['long_signal']:
                trades.append(Trade(bar.name, bar['close'], 'long'))
                position = 'long'
            elif bar['short_signal']:
                trades.append(Trade(bar.name, bar['close'], 'short'))
                position = 'short'

    if position and trades:
        t = trades[-1]
        if t.exit_time is None:
            t.exit_time = df.index[-1]
            t.exit_price = df.iloc[-1]['close']
            t.pnl_pct = ((t.exit_price - t.entry_price) / t.entry_price * 100
                        if t.direction == 'long' else
                        (t.entry_price - t.exit_price) / t.entry_price * 100)

    return trades


def backtest_donchian(df: pd.DataFrame, period: int = 50,
                      atr_stop: float = 2.0) -> List[Trade]:
    """Backtest Donchian Channel strategy."""
    df = df.copy()
    df['dc_upper'] = df['high'].rolling(period).max()
    df['dc_lower'] = df['low'].rolling(period).min()
    df['dc_mid'] = (df['dc_upper'] + df['dc_lower']) / 2
    df['atr'] = calculate_atr(df)
    df['volume_ma'] = df['volume'].rolling(20).mean()
    df['vol_ok'] = df['volume'] > df['volume_ma'] * 1.5

    df['long_signal'] = (df['high'] > df['dc_upper'].shift(1)) & df['vol_ok']
    df['short_signal'] = (df['low'] < df['dc_lower'].shift(1)) & df['vol_ok']

    trades = []
    position = None

    for i in range(max(period, 20), len(df)):
        bar = df.iloc[i]
        atr = bar['atr']
        if pd.isna(atr) or atr == 0:
            continue

        if position is not None:
            t = trades[-1]
            if position == 'long':
                stop = t.entry_price - atr_stop * atr
                if bar['low'] <= stop:
                    t.exit_time, t.exit_price = bar.name, stop
                    t.pnl_pct = (stop - t.entry_price) / t.entry_price * 100
                    position = None
                elif bar['low'] <= bar['dc_mid']:
                    t.exit_time, t.exit_price = bar.name, bar['dc_mid']
                    t.pnl_pct = (bar['dc_mid'] - t.entry_price) / t.entry_price * 100
                    position = None
            else:
                stop = t.entry_price + atr_stop * atr
                if bar['high'] >= stop:
                    t.exit_time, t.exit_price = bar.name, stop
                    t.pnl_pct = (t.entry_price - stop) / t.entry_price * 100
                    position = None
                elif bar['high'] >= bar['dc_mid']:
                    t.exit_time, t.exit_price = bar.name, bar['dc_mid']
                    t.pnl_pct = (t.entry_price - bar['dc_mid']) / t.entry_price * 100
                    position = None
        else:
            if bar['long_signal']:
                trades.append(Trade(bar.name, bar['high'], 'long'))
                position = 'long'
            elif bar['short_signal']:
                trades.append(Trade(bar.name, bar['low'], 'short'))
                position = 'short'

    if position and trades:
        t = trades[-1]
        if t.exit_time is None:
            t.exit_time = df.index[-1]
            t.exit_price = df.iloc[-1]['close']
            t.pnl_pct = ((t.exit_price - t.entry_price) / t.entry_price * 100
                        if t.direction == 'long' else
                        (t.entry_price - t.exit_price) / t.entry_price * 100)

    return trades


def calculate_metrics(trades: List[Trade], df: pd.DataFrame) -> dict:
    """Calculate performance metrics."""
    completed = [t for t in trades if t.pnl_pct is not None]
    if not completed:
        return {'trades': 0, 'total_return': 0, 'sharpe': 0}

    pnl = np.array([t.pnl_pct for t in completed])
    total = np.sum(pnl)
    sharpe = (np.mean(pnl) / np.std(pnl)) * np.sqrt(252) if np.std(pnl) > 0 else 0
    wr = (pnl > 0).sum() / len(pnl) * 100

    cumsum = np.cumsum(pnl)
    max_dd = np.min(cumsum - np.maximum.accumulate(cumsum))

    es_return = (df.iloc[-1]['close'] / df.iloc[0]['close'] - 1) * 100

    return {
        'trades': len(completed),
        'win_rate': wr,
        'total_return': total,
        'sharpe': sharpe,
        'max_dd': max_dd,
        'es_return': es_return,
        'alpha': total - es_return
    }


if __name__ == '__main__':
    print("="*70)
    print("ES FUTURES MULTI-TIMEFRAME TEST")
    print("="*70)

    # Load 15-min data
    df_15m = load_es_data()
    print(f"\n15-min: {len(df_15m)} bars from {df_15m.index[0]} to {df_15m.index[-1]}")

    # Resample to 30-min and 1-hour
    df_30m = resample_ohlcv(df_15m, '30min')
    df_1h = resample_ohlcv(df_15m, '1H')
    print(f"30-min: {len(df_30m)} bars")
    print(f"1-hour: {len(df_1h)} bars")

    # ES benchmark
    es_ret = (df_15m.iloc[-1]['close'] / df_15m.iloc[0]['close'] - 1) * 100
    print(f"\nES Buy & Hold: {es_ret:.1f}%")

    print("\n" + "="*70)
    print("KALMAN FILTER - Best params from earlier: vel=1.5, stop=1.5, target=4.0")
    print("="*70)

    results = []
    for name, df in [('15-min', df_15m), ('30-min', df_30m), ('1-hour', df_1h)]:
        trades = backtest_kalman(df, vel_thresh=1.5, atr_stop=1.5, atr_target=4.0)
        metrics = calculate_metrics(trades, df)
        results.append((name, 'Kalman', metrics))
        print(f"\n{name}: {metrics['trades']} trades, {metrics['win_rate']:.1f}% WR, "
              f"Return={metrics['total_return']:.1f}%, Sharpe={metrics['sharpe']:.2f}, "
              f"Alpha={metrics['alpha']:.1f}%")

    print("\n" + "="*70)
    print("HULL MA - Best params: period=50, stop=1.5, target=4.0")
    print("="*70)

    for name, df in [('15-min', df_15m), ('30-min', df_30m), ('1-hour', df_1h)]:
        trades = backtest_hull(df, period=50, atr_stop=1.5, atr_target=4.0)
        metrics = calculate_metrics(trades, df)
        results.append((name, 'Hull', metrics))
        print(f"\n{name}: {metrics['trades']} trades, {metrics['win_rate']:.1f}% WR, "
              f"Return={metrics['total_return']:.1f}%, Sharpe={metrics['sharpe']:.2f}, "
              f"Alpha={metrics['alpha']:.1f}%")

    print("\n" + "="*70)
    print("DONCHIAN CHANNEL - Best params: period=50, stop=2.0, mid exit")
    print("="*70)

    for name, df in [('15-min', df_15m), ('30-min', df_30m), ('1-hour', df_1h)]:
        trades = backtest_donchian(df, period=50, atr_stop=2.0)
        metrics = calculate_metrics(trades, df)
        results.append((name, 'Donchian', metrics))
        print(f"\n{name}: {metrics['trades']} trades, {metrics['win_rate']:.1f}% WR, "
              f"Return={metrics['total_return']:.1f}%, Sharpe={metrics['sharpe']:.2f}, "
              f"Alpha={metrics['alpha']:.1f}%")

    print("\n" + "="*70)
    print("SUMMARY BY TIMEFRAME")
    print("="*70)
    print(f"\n{'Strategy':<12} {'Timeframe':<10} {'Trades':>7} {'Return':>10} {'Sharpe':>8} {'Alpha':>10}")
    print("-"*60)

    for name, strategy, m in sorted(results, key=lambda x: -x[2]['alpha']):
        print(f"{strategy:<12} {name:<10} {m['trades']:>7} {m['total_return']:>9.1f}% "
              f"{m['sharpe']:>7.2f} {m['alpha']:>+9.1f}%")

    # Find best
    best = max(results, key=lambda x: x[2]['alpha'])
    print(f"\nBest: {best[1]} on {best[0]} with alpha={best[2]['alpha']:.1f}%")
    print(f"       vs ES buy-and-hold of {es_ret:.1f}%")

    if best[2]['alpha'] > 0:
        print("\n>>> BEATS BENCHMARK!")
    else:
        print("\n>>> Still underperforms buy-and-hold")
