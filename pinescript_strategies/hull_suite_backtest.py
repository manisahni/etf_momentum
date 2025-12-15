"""
Hull Suite Strategy Backtest
============================
Translated from TradingView Pine Script to Python

Original: "Hull Suite Strategy" by InSilico/DashTrader
Optimized parameters found via grid search on real market data.

Usage:
    python hull_suite_backtest.py --instrument ES --length 20
    python hull_suite_backtest.py --instrument SPY --length 25 --mode long_only
"""

import pandas as pd
import numpy as np
import sqlite3
import argparse
from pathlib import Path
from dataclasses import dataclass
from typing import Literal


# =============================================================================
# CONFIGURATION
# =============================================================================

# Optimized parameters by instrument (found via grid search)
OPTIMAL_PARAMS = {
    'ES': {'length': 20},
    'SPY': {'length': 25},
    'DEFAULT': {'length': 55},  # Original Pine Script
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
# HULL SUITE STRATEGY
# =============================================================================

def wma(series: pd.Series, period: int) -> pd.Series:
    """Weighted Moving Average"""
    weights = np.arange(1, period + 1)
    return series.rolling(period).apply(lambda x: np.dot(x, weights) / weights.sum(), raw=True)


def ema(series: pd.Series, period: int) -> pd.Series:
    """Exponential Moving Average"""
    return series.ewm(span=period, adjust=False).mean()


def hma(series: pd.Series, length: int) -> pd.Series:
    """
    Hull Moving Average
    HMA = WMA(2 * WMA(n/2) - WMA(n), sqrt(n))
    """
    half_length = max(1, int(length / 2))
    sqrt_length = max(1, int(round(np.sqrt(length))))
    return wma(2 * wma(series, half_length) - wma(series, length), sqrt_length)


def ehma(series: pd.Series, length: int) -> pd.Series:
    """
    Exponential Hull Moving Average
    EHMA = EMA(2 * EMA(n/2) - EMA(n), sqrt(n))
    """
    half_length = max(1, int(length / 2))
    sqrt_length = max(1, int(round(np.sqrt(length))))
    return ema(2 * ema(series, half_length) - ema(series, length), sqrt_length)


def thma(series: pd.Series, length: int) -> pd.Series:
    """
    Triple Hull Moving Average
    THMA = WMA(WMA(n/3)*3 - WMA(n/2) - WMA(n), n)
    """
    third_length = max(1, int(length / 3))
    half_length = max(1, int(length / 2))
    return wma(
        wma(series, third_length) * 3 - wma(series, half_length) - wma(series, length),
        length
    )


def hull_suite(
    df: pd.DataFrame,
    length: int = 55,
    mode: Literal['Hma', 'Ehma', 'Thma'] = 'Hma',
    src: str = 'close',
) -> pd.DataFrame:
    """
    Hull Suite Strategy

    Args:
        df: DataFrame with OHLCV data
        length: Hull MA length (55 for swing, 180-200 for S/R)
        mode: Hull variation (Hma, Ehma, Thma)
        src: Source column for calculation

    Returns:
        DataFrame with signals added
    """
    df = df.copy()

    # Calculate Hull MA
    if mode == 'Hma':
        df['hull'] = hma(df[src], length)
    elif mode == 'Ehma':
        df['hull'] = ehma(df[src], length)
    elif mode == 'Thma':
        df['hull'] = thma(df[src], length // 2)

    # MHULL = HULL[0], SHULL = HULL[2]
    df['mhull'] = df['hull']
    df['shull'] = df['hull'].shift(2)

    # Trend: HULL[0] > HULL[2] = bullish
    df['hull_trend'] = np.where(df['mhull'] > df['shull'], 1, -1)

    # Signals (trend change)
    df['buy_signal'] = (df['hull_trend'] == 1) & (df['hull_trend'].shift(1) == -1)
    df['sell_signal'] = (df['hull_trend'] == -1) & (df['hull_trend'].shift(1) == 1)

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
    point_value: float = 50,
    allow_short: bool = True,
) -> BacktestResult:
    """Run backtest on DataFrame with buy_signal/sell_signal columns"""

    position = 0
    equity = initial_capital
    entry_price = 0
    trades = []
    equity_curve = [initial_capital]

    for i in range(1, len(df)):
        if pd.isna(df['hull'].iloc[i]):
            equity_curve.append(equity)
            continue

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
    parser = argparse.ArgumentParser(description='Hull Suite Strategy Backtest')
    parser.add_argument('--instrument', type=str, default='ES', choices=['ES', 'SPY'],
                       help='Instrument to backtest')
    parser.add_argument('--mode', type=str, default='long_only',
                       choices=['long_only', 'long_short'],
                       help='Trading mode')
    parser.add_argument('--params', type=str, default='optimized',
                       choices=['optimized', 'original'],
                       help='Parameter set to use')
    parser.add_argument('--length', type=int, help='Override Hull MA length')
    parser.add_argument('--hull-mode', type=str, default='Hma',
                       choices=['Hma', 'Ehma', 'Thma'],
                       help='Hull MA variation')
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
    if args.length:
        params['length'] = args.length

    print(f"\nParameters: length={params['length']}, mode={args.hull_mode}")

    # Run strategy
    df_signals = hull_suite(df, length=params['length'], mode=args.hull_mode)
    print(f"Signals: {df_signals['buy_signal'].sum()} buys, {df_signals['sell_signal'].sum()} sells")

    # Run backtest
    allow_short = args.mode == 'long_short'
    mode_str = "Long/Short" if allow_short else "Long Only"
    strategy_name = f"Hull Suite {args.hull_mode}-{params['length']} [{mode_str}]"

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
