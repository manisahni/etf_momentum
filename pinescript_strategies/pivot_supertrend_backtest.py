"""
Pivot Point SuperTrend Strategy Backtest
========================================
Translated from TradingView Pine Script to Python

Original: "Pivot Point SuperTrend Strategy [ES Optimized]"
Optimized parameters found via grid search on real market data.

Usage:
    python pivot_supertrend_backtest.py --instrument ES --mode long_short
    python pivot_supertrend_backtest.py --instrument SPY --mode long_only
"""

import pandas as pd
import numpy as np
import sqlite3
import argparse
from pathlib import Path
from dataclasses import dataclass
from typing import Literal, Optional


# =============================================================================
# CONFIGURATION
# =============================================================================

# Optimized parameters by instrument (found via grid search)
OPTIMAL_PARAMS = {
    'ES': {'prd': 5, 'factor': 2.0, 'atr_period': 23},
    'SPY': {'prd': 6, 'factor': 3.5, 'atr_period': 20},
    'DEFAULT': {'prd': 2, 'factor': 3.0, 'atr_period': 23},  # Original Pine Script
}

# Data paths
ES_DATA_PATH = '/Users/nish_macbook/development/trading/emini-futures/data/emini_futures_15min.db'
SPY_DATA_PATH = '/Users/nish_macbook/development/trading/0dte-intraday/data/alpaca_spy_1min_bars/alpaca_spy_1min'


# =============================================================================
# DATA LOADING
# =============================================================================

def load_es_data() -> pd.DataFrame:
    """Load ES futures 15-min data from SQLite database"""
    conn = sqlite3.connect(ES_DATA_PATH)
    df = pd.read_sql("SELECT * FROM es_15min_bars ORDER BY timestamp", conn)
    conn.close()

    df['timestamp'] = pd.to_datetime(df['timestamp'].str.replace(r'[-+]\d{2}:\d{2}$', '', regex=True))
    df = df.sort_values('timestamp').reset_index(drop=True)
    df = df.rename(columns={'timestamp': 'date'})
    df = df[df['volume'] > 0].reset_index(drop=True)

    return df


def load_spy_data(resample: str = '15min') -> pd.DataFrame:
    """Load SPY 1-min data and resample to desired timeframe"""
    base_path = Path(SPY_DATA_PATH)

    all_dfs = []
    for year in ['2020', '2021', '2022', '2023', '2024', '2025']:
        year_path = base_path / year
        if year_path.exists():
            for f in sorted(year_path.glob('*.parquet')):
                try:
                    all_dfs.append(pd.read_parquet(f, engine='fastparquet'))
                except:
                    pass

    df = pd.concat(all_dfs, ignore_index=True)
    df['timestamp'] = pd.to_datetime(df['timestamp']).dt.tz_localize(None)
    df = df.sort_values('timestamp').set_index('timestamp')

    # Resample
    df = df.resample(resample).agg({
        'open': 'first', 'high': 'max', 'low': 'min',
        'close': 'last', 'volume': 'sum'
    }).dropna().reset_index()
    df = df.rename(columns={'timestamp': 'date'})

    # RTH filter (9:30-16:00)
    df['hour'] = df['date'].dt.hour
    df['minute'] = df['date'].dt.minute
    df = df[
        ((df['hour'] == 9) & (df['minute'] >= 30)) |
        ((df['hour'] >= 10) & (df['hour'] < 16))
    ].reset_index(drop=True)

    return df


# =============================================================================
# PIVOT POINT SUPERTREND STRATEGY
# =============================================================================

def pine_pivothigh(high_series: pd.Series, left_bars: int, right_bars: int) -> pd.Series:
    """
    Pine Script ta.pivothigh equivalent
    Returns pivot high value at confirmation bar (delayed by right_bars)
    """
    result = pd.Series(index=high_series.index, dtype=float)
    result[:] = np.nan

    for i in range(left_bars + right_bars, len(high_series)):
        check_idx = i - right_bars
        check_val = high_series.iloc[check_idx]
        left_max = high_series.iloc[check_idx - left_bars:check_idx].max()
        right_max = high_series.iloc[check_idx + 1:check_idx + 1 + right_bars].max()

        if check_val >= left_max and check_val >= right_max:
            result.iloc[i] = check_val

    return result


def pine_pivotlow(low_series: pd.Series, left_bars: int, right_bars: int) -> pd.Series:
    """
    Pine Script ta.pivotlow equivalent
    Returns pivot low value at confirmation bar (delayed by right_bars)
    """
    result = pd.Series(index=low_series.index, dtype=float)
    result[:] = np.nan

    for i in range(left_bars + right_bars, len(low_series)):
        check_idx = i - right_bars
        check_val = low_series.iloc[check_idx]
        left_min = low_series.iloc[check_idx - left_bars:check_idx].min()
        right_min = low_series.iloc[check_idx + 1:check_idx + 1 + right_bars].min()

        if check_val <= left_min and check_val <= right_min:
            result.iloc[i] = check_val

    return result


def pivot_supertrend(
    df: pd.DataFrame,
    prd: int = 2,
    factor: float = 3.0,
    atr_period: int = 23,
    use_volume_filter: bool = False,
    volume_ma_length: int = 20,
) -> pd.DataFrame:
    """
    Pivot Point SuperTrend Strategy

    Args:
        df: DataFrame with OHLCV data
        prd: Pivot point period (lookback/lookforward)
        factor: ATR multiplier for bands
        atr_period: ATR calculation period
        use_volume_filter: Enable volume filter
        volume_ma_length: Volume MA length for filter

    Returns:
        DataFrame with signals added
    """
    df = df.copy()

    # Pivot Point Detection
    df['ph'] = pine_pivothigh(df['high'], prd, prd)
    df['pl'] = pine_pivotlow(df['low'], prd, prd)

    # Center Line Calculation (EMA-style smoothing)
    center = np.full(len(df), np.nan)
    for i in range(len(df)):
        ph = df['ph'].iloc[i]
        pl = df['pl'].iloc[i]
        lastpp = ph if not np.isnan(ph) else (pl if not np.isnan(pl) else np.nan)

        if not np.isnan(lastpp):
            if i == 0 or np.isnan(center[i-1]):
                center[i] = lastpp
            else:
                center[i] = (center[i-1] * 2 + lastpp) / 3
        elif i > 0:
            center[i] = center[i-1]

    df['center'] = center

    # ATR Calculation
    df['tr'] = np.maximum(
        df['high'] - df['low'],
        np.maximum(
            abs(df['high'] - df['close'].shift(1)),
            abs(df['low'] - df['close'].shift(1))
        )
    )
    df['atr'] = df['tr'].rolling(window=atr_period).mean()

    # ATR Bands
    df['up'] = df['center'] - (factor * df['atr'])
    df['dn'] = df['center'] + (factor * df['atr'])

    # Trend Detection with Trailing Stop Logic
    trend = np.zeros(len(df))
    tup = np.full(len(df), np.nan)
    tdown = np.full(len(df), np.nan)

    for i in range(1, len(df)):
        up_val = df['up'].iloc[i]
        dn_val = df['dn'].iloc[i]
        close_prev = df['close'].iloc[i-1]
        close_curr = df['close'].iloc[i]

        # TUp calculation
        if np.isnan(tup[i-1]):
            tup[i] = up_val
        elif close_prev > tup[i-1]:
            tup[i] = max(up_val, tup[i-1]) if not np.isnan(up_val) else tup[i-1]
        else:
            tup[i] = up_val if not np.isnan(up_val) else tup[i-1]

        # TDown calculation
        if np.isnan(tdown[i-1]):
            tdown[i] = dn_val
        elif close_prev < tdown[i-1]:
            tdown[i] = min(dn_val, tdown[i-1]) if not np.isnan(dn_val) else tdown[i-1]
        else:
            tdown[i] = dn_val if not np.isnan(dn_val) else tdown[i-1]

        # Trend determination
        if not np.isnan(tdown[i-1]) and close_curr > tdown[i-1]:
            trend[i] = 1
        elif not np.isnan(tup[i-1]) and close_curr < tup[i-1]:
            trend[i] = -1
        else:
            trend[i] = trend[i-1] if trend[i-1] != 0 else 1

    df['trend'] = trend
    df['tup'] = tup
    df['tdown'] = tdown
    df['trailing_stop'] = np.where(df['trend'] == 1, df['tup'], df['tdown'])

    # Volume Filter (optional)
    if use_volume_filter:
        df['vol_ma'] = df['volume'].rolling(window=volume_ma_length).mean()
        df['volume_ok'] = df['volume'] > df['vol_ma']
    else:
        df['volume_ok'] = True

    # Signals (trend change)
    df['buy_signal'] = (df['trend'] == 1) & (df['trend'].shift(1) == -1) & df['volume_ok']
    df['sell_signal'] = (df['trend'] == -1) & (df['trend'].shift(1) == 1) & df['volume_ok']

    return df


# =============================================================================
# BACKTEST ENGINE
# =============================================================================

@dataclass
class BacktestResult:
    strategy_name: str
    total_pnl: float
    total_return: float
    sharpe_ratio: float
    max_drawdown: float
    num_trades: int
    win_rate: float
    profit_factor: float
    avg_win: float
    avg_loss: float
    trades: pd.DataFrame


def run_backtest(
    df: pd.DataFrame,
    strategy_name: str,
    initial_capital: float = 100000,
    point_value: float = 50,  # ES=$50, SPY shares=100
    allow_short: bool = True,
) -> BacktestResult:
    """Run backtest on DataFrame with buy_signal/sell_signal columns"""

    position = 0
    equity = initial_capital
    entry_price = 0
    trades = []
    equity_curve = [initial_capital]

    for i in range(1, len(df)):
        buy = df['buy_signal'].iloc[i]
        sell = df['sell_signal'].iloc[i]
        price = df['close'].iloc[i]
        date = df['date'].iloc[i]

        if buy and position != 1:
            if position == -1:  # Close short
                pnl = (entry_price - price) * point_value
                equity += pnl
                trades.append({
                    'entry_date': entry_date, 'exit_date': date,
                    'direction': 'short', 'entry_price': entry_price,
                    'exit_price': price, 'pnl': pnl
                })
            position = 1
            entry_price = price
            entry_date = date

        elif sell and position != -1:
            if position == 1:  # Close long
                pnl = (price - entry_price) * point_value
                equity += pnl
                trades.append({
                    'entry_date': entry_date, 'exit_date': date,
                    'direction': 'long', 'entry_price': entry_price,
                    'exit_price': price, 'pnl': pnl
                })
            if allow_short:
                position = -1
                entry_price = price
                entry_date = date
            else:
                position = 0

        equity_curve.append(equity)

    # Close final position
    if position == 1:
        pnl = (df['close'].iloc[-1] - entry_price) * point_value
        equity += pnl
        trades.append({
            'entry_date': entry_date, 'exit_date': df['date'].iloc[-1],
            'direction': 'long', 'entry_price': entry_price,
            'exit_price': df['close'].iloc[-1], 'pnl': pnl
        })
    elif position == -1:
        pnl = (entry_price - df['close'].iloc[-1]) * point_value
        equity += pnl
        trades.append({
            'entry_date': entry_date, 'exit_date': df['date'].iloc[-1],
            'direction': 'short', 'entry_price': entry_price,
            'exit_price': df['close'].iloc[-1], 'pnl': pnl
        })

    # Calculate metrics
    trades_df = pd.DataFrame(trades) if trades else pd.DataFrame()
    equity_series = pd.Series(equity_curve)

    total_pnl = equity - initial_capital
    total_return = total_pnl / initial_capital

    # Sharpe (annualized for 15-min bars)
    returns = equity_series.pct_change().dropna()
    sharpe = returns.mean() / returns.std() * np.sqrt(6800) if returns.std() > 0 else 0

    # Max drawdown
    rolling_max = equity_series.cummax()
    drawdown = (equity_series - rolling_max) / rolling_max
    max_dd = drawdown.min()

    # Trade statistics
    if len(trades_df) > 0:
        win_rate = (trades_df['pnl'] > 0).mean()
        winners = trades_df[trades_df['pnl'] > 0]['pnl']
        losers = trades_df[trades_df['pnl'] < 0]['pnl']
        avg_win = winners.mean() if len(winners) > 0 else 0
        avg_loss = losers.mean() if len(losers) > 0 else 0
        profit_factor = winners.sum() / abs(losers.sum()) if losers.sum() != 0 else float('inf')
    else:
        win_rate = avg_win = avg_loss = profit_factor = 0

    return BacktestResult(
        strategy_name=strategy_name,
        total_pnl=total_pnl,
        total_return=total_return,
        sharpe_ratio=sharpe,
        max_drawdown=max_dd,
        num_trades=len(trades_df),
        win_rate=win_rate,
        profit_factor=profit_factor,
        avg_win=avg_win,
        avg_loss=avg_loss,
        trades=trades_df
    )


def print_results(result: BacktestResult):
    """Print formatted backtest results"""
    print(f"\n{'='*60}")
    print(f"  {result.strategy_name}")
    print(f"{'='*60}")
    print(f"  Total P&L:        ${result.total_pnl:>12,.0f}")
    print(f"  Total Return:     {result.total_return:>12.1%}")
    print(f"  Sharpe Ratio:     {result.sharpe_ratio:>12.2f}")
    print(f"  Max Drawdown:     {result.max_drawdown:>12.1%}")
    print(f"  {'-'*56}")
    print(f"  Trades:           {result.num_trades:>12}")
    print(f"  Win Rate:         {result.win_rate:>12.1%}")
    print(f"  Avg Win:          ${result.avg_win:>11,.0f}")
    print(f"  Avg Loss:         ${result.avg_loss:>11,.0f}")
    pf_str = f"{result.profit_factor:.2f}" if result.profit_factor < 100 else "inf"
    print(f"  Profit Factor:    {pf_str:>12}")
    print(f"{'='*60}")


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description='Pivot Point SuperTrend Backtest')
    parser.add_argument('--instrument', type=str, default='ES', choices=['ES', 'SPY'],
                       help='Instrument to backtest')
    parser.add_argument('--mode', type=str, default='long_only',
                       choices=['long_only', 'long_short'],
                       help='Trading mode')
    parser.add_argument('--params', type=str, default='optimized',
                       choices=['optimized', 'original'],
                       help='Parameter set to use')
    parser.add_argument('--prd', type=int, help='Override pivot period')
    parser.add_argument('--factor', type=float, help='Override ATR factor')
    parser.add_argument('--atr', type=int, help='Override ATR period')
    parser.add_argument('--save-trades', type=str, help='Save trades to CSV file')

    args = parser.parse_args()

    # Load data
    print(f"\nLoading {args.instrument} data...")
    if args.instrument == 'ES':
        df = load_es_data()
        point_value = 50
    else:
        df = load_spy_data()
        point_value = 100  # 100 shares

    print(f"Loaded {len(df):,} bars from {df['date'].min()} to {df['date'].max()}")

    # Get parameters
    if args.params == 'optimized':
        params = OPTIMAL_PARAMS.get(args.instrument, OPTIMAL_PARAMS['DEFAULT'])
    else:
        params = OPTIMAL_PARAMS['DEFAULT']

    # Override with command line args
    if args.prd: params['prd'] = args.prd
    if args.factor: params['factor'] = args.factor
    if args.atr: params['atr_period'] = args.atr

    print(f"\nParameters: prd={params['prd']}, factor={params['factor']}, atr={params['atr_period']}")

    # Run strategy
    df_signals = pivot_supertrend(df, **params)
    print(f"Signals: {df_signals['buy_signal'].sum()} buys, {df_signals['sell_signal'].sum()} sells")

    # Run backtest
    allow_short = args.mode == 'long_short'
    mode_str = "Long/Short" if allow_short else "Long Only"
    strategy_name = f"Pivot SuperTrend ({params['prd']},{params['factor']},{params['atr_period']}) [{mode_str}]"

    result = run_backtest(
        df_signals,
        strategy_name=strategy_name,
        point_value=point_value,
        allow_short=allow_short
    )

    print_results(result)

    # Save trades if requested
    if args.save_trades and len(result.trades) > 0:
        result.trades.to_csv(args.save_trades, index=False)
        print(f"\nTrades saved to: {args.save_trades}")

    # Buy & Hold comparison
    bh_pnl = (df['close'].iloc[-1] - df['close'].iloc[0]) * point_value
    print(f"\nBuy & Hold: ${bh_pnl:,.0f} ({bh_pnl/100000:.1%})")


if __name__ == "__main__":
    main()
