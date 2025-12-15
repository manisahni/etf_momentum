"""
Strategy: Hull MA + Donchian Channel Breakouts

Two trend-following strategies on ES Futures 15-min:

1. Hull MA - Faster, smoother moving average
   - Less lag than EMA, good for trend detection
   - Signal: Hull MA direction change

2. Donchian Channel Breakouts + Volume
   - Classic breakout strategy
   - Signal: Price breaks N-period high/low with volume confirmation

Data: ES Futures 15-min (same as Gaussian/Kalman tests)
"""

import pandas as pd
import numpy as np
import sqlite3
from dataclasses import dataclass
from typing import List, Optional
import warnings
warnings.filterwarnings('ignore')

DB_PATH = '/Users/nish_macbook/development/trading/emini-futures/data/emini_futures_15min.db'
TABLE_NAME = 'es_15min_bars'


@dataclass
class Trade:
    entry_time: object
    entry_price: float
    direction: str
    strategy: str
    exit_time: Optional[object] = None
    exit_price: Optional[float] = None
    exit_reason: Optional[str] = None
    pnl: Optional[float] = None
    pnl_pct: Optional[float] = None


def load_es_data() -> pd.DataFrame:
    """Load ES futures 15-min data."""
    conn = sqlite3.connect(DB_PATH)
    query = f"SELECT timestamp, open, high, low, close, volume FROM {TABLE_NAME} ORDER BY timestamp"
    df = pd.read_sql_query(query, conn)
    conn.close()
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df = df.set_index('timestamp')
    print(f"Loaded {len(df)} bars from {df.index[0]} to {df.index[-1]}")
    return df


def hull_ma(prices: pd.Series, period: int = 20) -> pd.Series:
    """
    Calculate Hull Moving Average.

    HMA = WMA(2 * WMA(price, n/2) - WMA(price, n), sqrt(n))

    Features:
    - Much less lag than SMA/EMA
    - Smoother than raw price
    - Good for trend detection
    """
    half_period = int(period / 2)
    sqrt_period = int(np.sqrt(period))

    # WMA helper
    def wma(series, n):
        weights = np.arange(1, n + 1)
        return series.rolling(n).apply(lambda x: np.dot(x, weights) / weights.sum(), raw=True)

    wma_half = wma(prices, half_period)
    wma_full = wma(prices, period)
    raw_hma = 2 * wma_half - wma_full
    hma = wma(raw_hma, sqrt_period)

    return hma


def donchian_channel(df: pd.DataFrame, period: int = 20) -> pd.DataFrame:
    """
    Calculate Donchian Channel (N-period high/low).
    """
    df = df.copy()
    df['dc_upper'] = df['high'].rolling(period).max()
    df['dc_lower'] = df['low'].rolling(period).min()
    df['dc_mid'] = (df['dc_upper'] + df['dc_lower']) / 2
    return df


def calculate_atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """Calculate Average True Range."""
    high_low = df['high'] - df['low']
    high_close = abs(df['high'] - df['close'].shift())
    low_close = abs(df['low'] - df['close'].shift())
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    return tr.rolling(period).mean()


# ========== HULL MA STRATEGY ==========

def hull_ma_signals(df: pd.DataFrame, period: int = 20,
                    use_volume: bool = True) -> pd.DataFrame:
    """Generate Hull MA trend-following signals."""
    df = df.copy()
    df['hma'] = hull_ma(df['close'], period)
    df['hma_slope'] = df['hma'].diff()
    df['hma_prev_slope'] = df['hma_slope'].shift(1)

    # Volume filter
    if use_volume:
        df['volume_ma'] = df['volume'].rolling(20).mean()
        df['vol_ok'] = df['volume'] > df['volume_ma']
    else:
        df['vol_ok'] = True

    # Signal: slope change from negative to positive (bullish)
    df['long_signal'] = (df['hma_slope'] > 0) & (df['hma_prev_slope'] <= 0) & df['vol_ok']
    df['short_signal'] = (df['hma_slope'] < 0) & (df['hma_prev_slope'] >= 0) & df['vol_ok']

    return df


def backtest_hull_ma(df: pd.DataFrame, period: int = 20,
                     use_volume: bool = True,
                     atr_stop: float = 2.0,
                     atr_target: float = 3.0) -> List[Trade]:
    """Backtest Hull MA strategy."""
    signal_df = hull_ma_signals(df, period, use_volume)
    signal_df['atr'] = calculate_atr(signal_df)

    trades = []
    position = None

    for i in range(max(period, 14), len(signal_df)):
        bar = signal_df.iloc[i]
        atr = bar['atr']

        if pd.isna(atr) or atr == 0:
            continue

        # Check exit if in position
        if position is not None:
            current_trade = trades[-1]

            if position == 'long':
                stop = current_trade.entry_price - atr_stop * atr
                target = current_trade.entry_price + atr_target * atr

                if bar['low'] <= stop:
                    current_trade.exit_time = bar.name
                    current_trade.exit_price = stop
                    current_trade.exit_reason = 'stop'
                    current_trade.pnl = stop - current_trade.entry_price
                    current_trade.pnl_pct = current_trade.pnl / current_trade.entry_price * 100
                    position = None
                    continue

                if bar['high'] >= target:
                    current_trade.exit_time = bar.name
                    current_trade.exit_price = target
                    current_trade.exit_reason = 'target'
                    current_trade.pnl = target - current_trade.entry_price
                    current_trade.pnl_pct = current_trade.pnl / current_trade.entry_price * 100
                    position = None
                    continue

                if bar['short_signal']:
                    current_trade.exit_time = bar.name
                    current_trade.exit_price = bar['close']
                    current_trade.exit_reason = 'signal'
                    current_trade.pnl = bar['close'] - current_trade.entry_price
                    current_trade.pnl_pct = current_trade.pnl / current_trade.entry_price * 100
                    trades.append(Trade(bar.name, bar['close'], 'short', 'hull_ma'))
                    position = 'short'
                    continue

            else:  # short
                stop = current_trade.entry_price + atr_stop * atr
                target = current_trade.entry_price - atr_target * atr

                if bar['high'] >= stop:
                    current_trade.exit_time = bar.name
                    current_trade.exit_price = stop
                    current_trade.exit_reason = 'stop'
                    current_trade.pnl = current_trade.entry_price - stop
                    current_trade.pnl_pct = current_trade.pnl / current_trade.entry_price * 100
                    position = None
                    continue

                if bar['low'] <= target:
                    current_trade.exit_time = bar.name
                    current_trade.exit_price = target
                    current_trade.exit_reason = 'target'
                    current_trade.pnl = current_trade.entry_price - target
                    current_trade.pnl_pct = current_trade.pnl / current_trade.entry_price * 100
                    position = None
                    continue

                if bar['long_signal']:
                    current_trade.exit_time = bar.name
                    current_trade.exit_price = bar['close']
                    current_trade.exit_reason = 'signal'
                    current_trade.pnl = current_trade.entry_price - bar['close']
                    current_trade.pnl_pct = current_trade.pnl / current_trade.entry_price * 100
                    trades.append(Trade(bar.name, bar['close'], 'long', 'hull_ma'))
                    position = 'long'
                    continue

        # New entry
        if position is None:
            if bar['long_signal']:
                trades.append(Trade(bar.name, bar['close'], 'long', 'hull_ma'))
                position = 'long'
            elif bar['short_signal']:
                trades.append(Trade(bar.name, bar['close'], 'short', 'hull_ma'))
                position = 'short'

    # Close open position
    if position and trades:
        t = trades[-1]
        if t.exit_time is None:
            t.exit_time = signal_df.index[-1]
            t.exit_price = signal_df.iloc[-1]['close']
            t.exit_reason = 'eod'
            t.pnl = (t.exit_price - t.entry_price) if t.direction == 'long' else (t.entry_price - t.exit_price)
            t.pnl_pct = t.pnl / t.entry_price * 100

    return trades


# ========== DONCHIAN CHANNEL STRATEGY ==========

def donchian_signals(df: pd.DataFrame, period: int = 20,
                     use_volume: bool = True) -> pd.DataFrame:
    """Generate Donchian Channel breakout signals."""
    df = donchian_channel(df, period)

    if use_volume:
        df['volume_ma'] = df['volume'].rolling(20).mean()
        df['vol_ok'] = df['volume'] > df['volume_ma'] * 1.5  # Higher volume threshold for breakouts
    else:
        df['vol_ok'] = True

    # Breakout signals
    df['long_signal'] = (df['high'] > df['dc_upper'].shift(1)) & df['vol_ok']
    df['short_signal'] = (df['low'] < df['dc_lower'].shift(1)) & df['vol_ok']

    return df


def backtest_donchian(df: pd.DataFrame, period: int = 20,
                      use_volume: bool = True,
                      atr_stop: float = 2.0,
                      exit_mid: bool = True) -> List[Trade]:
    """
    Backtest Donchian Channel breakout.

    Entry: Break of N-period high/low with volume
    Exit: Price touches middle band OR ATR stop
    """
    signal_df = donchian_signals(df, period, use_volume)
    signal_df['atr'] = calculate_atr(signal_df)

    trades = []
    position = None

    for i in range(max(period, 14), len(signal_df)):
        bar = signal_df.iloc[i]
        atr = bar['atr']

        if pd.isna(atr) or atr == 0:
            continue

        # Check exit if in position
        if position is not None:
            current_trade = trades[-1]

            if position == 'long':
                stop = current_trade.entry_price - atr_stop * atr

                if bar['low'] <= stop:
                    current_trade.exit_time = bar.name
                    current_trade.exit_price = stop
                    current_trade.exit_reason = 'stop'
                    current_trade.pnl = stop - current_trade.entry_price
                    current_trade.pnl_pct = current_trade.pnl / current_trade.entry_price * 100
                    position = None
                    continue

                # Exit at middle band
                if exit_mid and bar['low'] <= bar['dc_mid']:
                    current_trade.exit_time = bar.name
                    current_trade.exit_price = bar['dc_mid']
                    current_trade.exit_reason = 'mid_band'
                    current_trade.pnl = bar['dc_mid'] - current_trade.entry_price
                    current_trade.pnl_pct = current_trade.pnl / current_trade.entry_price * 100
                    position = None
                    continue

            else:  # short
                stop = current_trade.entry_price + atr_stop * atr

                if bar['high'] >= stop:
                    current_trade.exit_time = bar.name
                    current_trade.exit_price = stop
                    current_trade.exit_reason = 'stop'
                    current_trade.pnl = current_trade.entry_price - stop
                    current_trade.pnl_pct = current_trade.pnl / current_trade.entry_price * 100
                    position = None
                    continue

                if exit_mid and bar['high'] >= bar['dc_mid']:
                    current_trade.exit_time = bar.name
                    current_trade.exit_price = bar['dc_mid']
                    current_trade.exit_reason = 'mid_band'
                    current_trade.pnl = current_trade.entry_price - bar['dc_mid']
                    current_trade.pnl_pct = current_trade.pnl / current_trade.entry_price * 100
                    position = None
                    continue

        # New entry
        if position is None:
            if bar['long_signal']:
                trades.append(Trade(bar.name, bar['high'], 'long', 'donchian'))
                position = 'long'
            elif bar['short_signal']:
                trades.append(Trade(bar.name, bar['low'], 'short', 'donchian'))
                position = 'short'

    # Close open position
    if position and trades:
        t = trades[-1]
        if t.exit_time is None:
            t.exit_time = signal_df.index[-1]
            t.exit_price = signal_df.iloc[-1]['close']
            t.exit_reason = 'eod'
            t.pnl = (t.exit_price - t.entry_price) if t.direction == 'long' else (t.entry_price - t.exit_price)
            t.pnl_pct = t.pnl / t.entry_price * 100

    return trades


# ========== ANALYSIS ==========

def calculate_metrics(trades: List[Trade], df: pd.DataFrame) -> dict:
    """Calculate performance metrics."""
    completed = [t for t in trades if t.pnl_pct is not None]
    if not completed:
        return {}

    pnl_pcts = np.array([t.pnl_pct for t in completed])
    wins = pnl_pcts[pnl_pcts > 0]
    losses = pnl_pcts[pnl_pcts < 0]

    total_return = np.sum(pnl_pcts)
    days = (df.index[-1] - df.index[0]).days
    years = days / 365
    annual_return = total_return / years if years > 0 else 0

    sharpe = (np.mean(pnl_pcts) / np.std(pnl_pcts)) * np.sqrt(252) if np.std(pnl_pcts) > 0 else 0

    cumsum = np.cumsum(pnl_pcts)
    running_max = np.maximum.accumulate(cumsum)
    max_dd = np.min(cumsum - running_max)

    es_return = (df.iloc[-1]['close'] / df.iloc[0]['close'] - 1) * 100

    return {
        'trades': len(completed),
        'win_rate': len(wins) / len(pnl_pcts) * 100,
        'total_return': total_return,
        'annual_return': annual_return,
        'sharpe': sharpe,
        'max_dd': max_dd,
        'avg_win': np.mean(wins) if len(wins) > 0 else 0,
        'avg_loss': np.mean(losses) if len(losses) > 0 else 0,
        'es_return': es_return,
        'years': years
    }


def print_results(metrics: dict, name: str):
    """Print formatted results."""
    if not metrics:
        print(f"\n{name}: No trades generated")
        return

    print(f"\n{'='*60}")
    print(f"RESULTS: {name}")
    print(f"{'='*60}")
    print(f"Trades: {metrics['trades']}, Win Rate: {metrics['win_rate']:.1f}%")
    print(f"Total Return: {metrics['total_return']:.1f}%")
    print(f"Annual Return: {metrics['annual_return']:.1f}%")
    print(f"Sharpe: {metrics['sharpe']:.2f}, Max DD: {metrics['max_dd']:.1f}%")
    print(f"ES Buy & Hold: {metrics['es_return']:.1f}%")
    print(f"Alpha: {metrics['total_return'] - metrics['es_return']:.1f}%")


def parameter_sweep_hull(df: pd.DataFrame):
    """Sweep Hull MA parameters."""
    print("\n" + "="*60)
    print("HULL MA PARAMETER SWEEP")
    print("="*60)

    results = []
    for period in [10, 15, 20, 30, 50]:
        for use_vol in [False, True]:
            for atr_stop in [1.5, 2.0, 2.5]:
                for atr_target in [2.0, 3.0, 4.0]:
                    trades = backtest_hull_ma(df, period, use_vol, atr_stop, atr_target)
                    metrics = calculate_metrics(trades, df)
                    if metrics and metrics['trades'] >= 30:
                        results.append({
                            'period': period,
                            'vol': use_vol,
                            'stop': atr_stop,
                            'target': atr_target,
                            **metrics
                        })

    results.sort(key=lambda x: -x['sharpe'])

    print(f"\n{'Period':>7} {'Vol':>5} {'Stop':>5} {'Tgt':>5} {'Trades':>7} {'WR':>6} {'Return':>8} {'Sharpe':>7}")
    print("-"*60)
    for r in results[:10]:
        v = 'Y' if r['vol'] else 'N'
        print(f"{r['period']:7d} {v:>5} {r['stop']:5.1f} {r['target']:5.1f} "
              f"{r['trades']:7d} {r['win_rate']:5.1f}% {r['total_return']:7.1f}% {r['sharpe']:6.2f}")

    return results


def parameter_sweep_donchian(df: pd.DataFrame):
    """Sweep Donchian parameters."""
    print("\n" + "="*60)
    print("DONCHIAN CHANNEL PARAMETER SWEEP")
    print("="*60)

    results = []
    for period in [10, 20, 30, 50]:
        for use_vol in [False, True]:
            for atr_stop in [1.5, 2.0, 2.5, 3.0]:
                for exit_mid in [False, True]:
                    trades = backtest_donchian(df, period, use_vol, atr_stop, exit_mid)
                    metrics = calculate_metrics(trades, df)
                    if metrics and metrics['trades'] >= 30:
                        results.append({
                            'period': period,
                            'vol': use_vol,
                            'stop': atr_stop,
                            'exit_mid': exit_mid,
                            **metrics
                        })

    results.sort(key=lambda x: -x['sharpe'])

    print(f"\n{'Period':>7} {'Vol':>5} {'Stop':>5} {'Mid':>5} {'Trades':>7} {'WR':>6} {'Return':>8} {'Sharpe':>7}")
    print("-"*60)
    for r in results[:10]:
        v = 'Y' if r['vol'] else 'N'
        m = 'Y' if r['exit_mid'] else 'N'
        print(f"{r['period']:7d} {v:>5} {r['stop']:5.1f} {m:>5} "
              f"{r['trades']:7d} {r['win_rate']:5.1f}% {r['total_return']:7.1f}% {r['sharpe']:6.2f}")

    return results


if __name__ == '__main__':
    print("="*60)
    print("HULL MA & DONCHIAN CHANNEL BACKTESTS")
    print("="*60)

    df = load_es_data()

    # Hull MA default test
    print("\n--- Hull MA (Default: Period=20, Volume=True) ---")
    hull_trades = backtest_hull_ma(df, period=20, use_volume=True)
    hull_metrics = calculate_metrics(hull_trades, df)
    print_results(hull_metrics, "Hull MA Default")

    # Donchian default test
    print("\n--- Donchian Channel (Default: Period=20, Volume=True) ---")
    donchian_trades = backtest_donchian(df, period=20, use_volume=True)
    donchian_metrics = calculate_metrics(donchian_trades, df)
    print_results(donchian_metrics, "Donchian Default")

    # Parameter sweeps
    hull_results = parameter_sweep_hull(df)
    donchian_results = parameter_sweep_donchian(df)

    # Summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)

    if hull_results:
        best_hull = hull_results[0]
        print(f"\nBest Hull MA: Period={best_hull['period']}, Vol={'Y' if best_hull['vol'] else 'N'}")
        print(f"  Sharpe: {best_hull['sharpe']:.2f}, Return: {best_hull['total_return']:.1f}%")
        print(f"  Alpha: {best_hull['total_return'] - best_hull['es_return']:.1f}%")

    if donchian_results:
        best_donchian = donchian_results[0]
        print(f"\nBest Donchian: Period={best_donchian['period']}, Vol={'Y' if best_donchian['vol'] else 'N'}")
        print(f"  Sharpe: {best_donchian['sharpe']:.2f}, Return: {best_donchian['total_return']:.1f}%")
        print(f"  Alpha: {best_donchian['total_return'] - best_donchian['es_return']:.1f}%")
