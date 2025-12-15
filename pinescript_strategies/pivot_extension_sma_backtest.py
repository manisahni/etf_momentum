"""
Pivot Extension Strategy with SMA Regime Filter - ES 15-min Backtest
=====================================================================
Translated from TradingView Pine Script to Python with regime filtering.

Original Pine Script:
    leftBars = input(4, "Pivot Lookback Left")
    rightBars = input(2, "Pivot Lookback Right")
    ph = ta.pivothigh(leftBars, rightBars)
    pl = ta.pivotlow(leftBars, rightBars)
    if (not na(pl)) strategy.entry("PivExtLE", strategy.long)
    if (not na(ph)) strategy.entry("PivExtSE", strategy.short)

Strategy Logic:
- Long entry on pivot low confirmation (after right_bars candles confirm the low)
- Exit on pivot high confirmation
- Optional SMA regime filter to trade with trend or in ranging conditions

Optimal Parameters (ES 15-min, Jan 2024 - Nov 2025):
    left_bars=6, right_bars=1, sma_length=50, regime='mean_revert'
    Result: 69.3% return, 2.32 Sharpe, -6.5% MaxDD, 477 trades, 61.6% win rate

Key Finding:
    The mean_revert regime filter improves Sharpe by 58% over unfiltered signals.
    Pivot reversals are most reliable when price is near equilibrium (ranging).

Usage:
    # Run with optimal parameters
    python pivot_extension_sma_backtest.py --left-bars 6 --right-bars 1 \\
        --sma-length 50 --regime mean_revert

    # Run parameter optimization
    python pivot_extension_sma_backtest.py --optimize
"""

import pandas as pd
import numpy as np
import sqlite3
import argparse
from dataclasses import dataclass
from typing import Literal, Optional
from itertools import product


# =============================================================================
# CONFIGURATION
# =============================================================================

ES_DATA_PATH = '/Users/nish_macbook/development/trading/emini-futures/data/emini_futures_15min.db'

# Optimal parameters (discovered via grid search on ES 15-min, Jan 2024 - Nov 2025)
OPTIMAL_PARAMS = {
    'left_bars': 6,
    'right_bars': 1,
    'sma_length': 50,
    'regime_mode': 'mean_revert',  # Best: 69.3% return, 2.32 Sharpe, -6.5% MaxDD
}

# Default parameters (original Pine Script)
DEFAULT_PARAMS = {
    'left_bars': 4,
    'right_bars': 2,
    'sma_length': 200,
    'regime_mode': 'none',  # none, trend_follow, mean_revert
}


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


# =============================================================================
# PIVOT DETECTION (CAUSAL - NO LOOK-AHEAD)
# =============================================================================

def pivot_high(high: pd.Series, left: int, right: int) -> pd.Series:
    """
    Detect pivot highs - signal fires AFTER confirmation (no look-ahead).

    At bar i, we check if bar (i - right) is a pivot high by comparing it to:
    - left bars before it
    - right bars after it (including bar i-1)

    This is causal because we only use data up to bar i.
    """
    result = pd.Series(index=high.index, dtype=float)
    result[:] = np.nan

    for i in range(left + right, len(high)):
        pivot_idx = i - right
        pivot_val = high.iloc[pivot_idx]

        # Left window: bars before the pivot
        left_window = high.iloc[pivot_idx - left:pivot_idx]
        # Right window: bars after pivot up to (but not including) current bar
        right_window = high.iloc[pivot_idx + 1:i + 1]

        if len(left_window) == left and len(right_window) == right:
            if pivot_val >= left_window.max() and pivot_val > right_window.max():
                result.iloc[i] = pivot_val

    return result


def pivot_low(low: pd.Series, left: int, right: int) -> pd.Series:
    """
    Detect pivot lows - signal fires AFTER confirmation (no look-ahead).
    """
    result = pd.Series(index=low.index, dtype=float)
    result[:] = np.nan

    for i in range(left + right, len(low)):
        pivot_idx = i - right
        pivot_val = low.iloc[pivot_idx]

        left_window = low.iloc[pivot_idx - left:pivot_idx]
        right_window = low.iloc[pivot_idx + 1:i + 1]

        if len(left_window) == left and len(right_window) == right:
            if pivot_val <= left_window.min() and pivot_val < right_window.min():
                result.iloc[i] = pivot_val

    return result


# =============================================================================
# PIVOT EXTENSION STRATEGY
# =============================================================================

def pivot_extension_strategy(
    df: pd.DataFrame,
    left_bars: int = 4,
    right_bars: int = 2,
    sma_length: int = 200,
    regime_mode: Literal['none', 'trend_follow', 'mean_revert'] = 'none',
) -> pd.DataFrame:
    """
    Pivot Extension Strategy with SMA Regime Filter.

    Args:
        df: DataFrame with OHLCV data
        left_bars: Bars to look back for pivot detection
        right_bars: Bars to confirm pivot
        sma_length: SMA period for regime filter
        regime_mode:
            - 'none': No filtering
            - 'trend_follow': Long only above SMA, short only below
            - 'mean_revert': Only trade when price near SMA (within 1%)

    Returns:
        DataFrame with signals added
    """
    df = df.copy()

    # Calculate SMA
    df['sma'] = df['close'].rolling(sma_length).mean()

    # Detect pivots
    df['pivot_high'] = pivot_high(df['high'], left_bars, right_bars)
    df['pivot_low'] = pivot_low(df['low'], left_bars, right_bars)

    # Raw signals (pivot detected = signal fires)
    df['raw_long'] = ~df['pivot_low'].isna()
    df['raw_short'] = ~df['pivot_high'].isna()

    # Apply regime filter
    if regime_mode == 'none':
        df['buy_signal'] = df['raw_long']
        df['sell_signal'] = df['raw_short']

    elif regime_mode == 'trend_follow':
        # Long only when above SMA, exit (or short) when below
        df['above_sma'] = df['close'] > df['sma']
        df['buy_signal'] = df['raw_long'] & df['above_sma']
        df['sell_signal'] = df['raw_short']  # Exit on any pivot high

    elif regime_mode == 'mean_revert':
        # Only trade when price is within 1% of SMA (ranging)
        df['near_sma'] = (df['close'] - df['sma']).abs() / df['sma'] < 0.01
        df['buy_signal'] = df['raw_long'] & df['near_sma']
        df['sell_signal'] = df['raw_short'] & df['near_sma']

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
    allow_short: bool = False,
) -> BacktestResult:
    """Run backtest on DataFrame with buy_signal/sell_signal columns"""

    position = 0
    equity = initial_capital
    entry_price = 0
    entry_date = None
    trades = []
    equity_curve = [initial_capital]

    for i in range(1, len(df)):
        if pd.isna(df['sma'].iloc[i]):
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

    # Sharpe (annualized for 15-min bars, ~6800 RTH bars/year)
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
# PARAMETER OPTIMIZATION
# =============================================================================

def run_optimization(df: pd.DataFrame) -> pd.DataFrame:
    """Grid search parameter optimization"""

    # Parameter grid
    left_bars_range = [2, 3, 4, 5, 6]
    right_bars_range = [1, 2, 3]
    sma_lengths = [20, 50, 100, 200]
    regime_modes = ['none', 'trend_follow', 'mean_revert']

    results = []
    total_combos = len(left_bars_range) * len(right_bars_range) * len(sma_lengths) * len(regime_modes)

    print(f"\nRunning optimization ({total_combos} combinations)...")

    for i, (left, right, sma, regime) in enumerate(product(
        left_bars_range, right_bars_range, sma_lengths, regime_modes
    )):
        if (i + 1) % 20 == 0:
            print(f"  Progress: {i+1}/{total_combos}")

        try:
            df_signals = pivot_extension_strategy(
                df, left_bars=left, right_bars=right,
                sma_length=sma, regime_mode=regime
            )

            result = run_backtest(
                df_signals,
                strategy_name=f"L{left}_R{right}_SMA{sma}_{regime}",
                allow_short=False
            )

            results.append({
                'left_bars': left,
                'right_bars': right,
                'sma_length': sma,
                'regime': regime,
                'sharpe': result.sharpe_ratio,
                'return': result.total_return,
                'max_dd': result.max_drawdown,
                'trades': result.num_trades,
                'win_rate': result.win_rate,
                'pnl': result.total_pnl,
            })
        except Exception as e:
            pass

    results_df = pd.DataFrame(results)
    return results_df


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description='Pivot Extension Strategy with SMA Filter')
    parser.add_argument('--params', type=str, default='default',
                       choices=['default', 'optimal'],
                       help='Parameter preset: default (original Pine) or optimal (best Sharpe)')
    parser.add_argument('--left-bars', type=int, help='Override pivot lookback left')
    parser.add_argument('--right-bars', type=int, help='Override pivot lookback right')
    parser.add_argument('--sma-length', type=int, help='Override SMA length')
    parser.add_argument('--regime', type=str,
                       choices=['none', 'trend_follow', 'mean_revert'],
                       help='Override regime filter mode')
    parser.add_argument('--allow-short', action='store_true', help='Allow short positions')
    parser.add_argument('--optimize', action='store_true', help='Run parameter optimization')
    parser.add_argument('--save-trades', type=str, help='Save trades to CSV')

    args = parser.parse_args()

    # Load data
    print("\nLoading ES 15-min data...")
    df = load_es_data()
    print(f"Loaded {len(df):,} bars from {df['date'].min()} to {df['date'].max()}")

    if args.optimize:
        # Run optimization
        results_df = run_optimization(df)

        print("\n" + "="*80)
        print("TOP 10 CONFIGURATIONS BY SHARPE RATIO")
        print("="*80)

        top_results = results_df.nlargest(10, 'sharpe')
        for i, row in top_results.iterrows():
            print(f"\n{row['left_bars']}/{row['right_bars']} SMA{row['sma_length']} {row['regime']}:")
            print(f"  Sharpe: {row['sharpe']:.2f}, Return: {row['return']:.1%}, "
                  f"MaxDD: {row['max_dd']:.1%}, Trades: {row['trades']}")

        # Compare regimes
        print("\n" + "="*80)
        print("REGIME COMPARISON (Average across all configs)")
        print("="*80)

        regime_summary = results_df.groupby('regime').agg({
            'sharpe': 'mean',
            'return': 'mean',
            'max_dd': 'mean',
            'trades': 'mean'
        }).round(3)
        print(regime_summary)

        # Save full results
        results_df.to_csv('pivot_extension_optimization.csv', index=False)
        print("\nFull results saved to: pivot_extension_optimization.csv")

    else:
        # Get base parameters from preset
        if args.params == 'optimal':
            params = OPTIMAL_PARAMS.copy()
            print("\nUsing OPTIMAL parameters (best Sharpe from grid search)")
        else:
            params = DEFAULT_PARAMS.copy()
            print("\nUsing DEFAULT parameters (original Pine Script)")

        # Override with command line args if specified
        if args.left_bars is not None:
            params['left_bars'] = args.left_bars
        if args.right_bars is not None:
            params['right_bars'] = args.right_bars
        if args.sma_length is not None:
            params['sma_length'] = args.sma_length
        if args.regime is not None:
            params['regime_mode'] = args.regime

        print(f"Parameters: left={params['left_bars']}, right={params['right_bars']}, "
              f"sma={params['sma_length']}, regime={params['regime_mode']}")

        df_signals = pivot_extension_strategy(
            df,
            left_bars=params['left_bars'],
            right_bars=params['right_bars'],
            sma_length=params['sma_length'],
            regime_mode=params['regime_mode']
        )

        n_longs = df_signals['buy_signal'].sum()
        n_shorts = df_signals['sell_signal'].sum()
        print(f"Signals: {n_longs} longs, {n_shorts} exits/shorts")

        mode_str = "Long/Short" if args.allow_short else "Long Only"
        strategy_name = f"Pivot Extension {params['left_bars']}/{params['right_bars']} SMA{params['sma_length']} {params['regime_mode']} [{mode_str}]"

        result = run_backtest(
            df_signals,
            strategy_name=strategy_name,
            allow_short=args.allow_short
        )

        print_results(result)

        if args.save_trades and len(result.trades) > 0:
            result.trades.to_csv(args.save_trades, index=False)
            print(f"\nTrades saved to: {args.save_trades}")

        # Buy & Hold comparison
        bh_pnl = (df['close'].iloc[-1] - df['close'].iloc[0]) * 50
        print(f"\nBuy & Hold: ${bh_pnl:,.0f} ({bh_pnl/100000:.1%})")


if __name__ == "__main__":
    main()
