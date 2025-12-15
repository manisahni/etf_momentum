"""
MES 15-Min RSI Mean Reversion - Enhanced with Frequency-Friendly Filters

Tests Volume, Time-of-Day, and Volatility filters while prioritizing trade frequency.
Uses "exclusion" approach (skip bad conditions) rather than "inclusion" (require good conditions).

Baseline: RSI<45, +1.0%/-0.15%, 6 bars = Sharpe 5.02, 28% annual, $13,176/contract
"""

import pandas as pd
import numpy as np
import sqlite3
from datetime import time
from typing import Dict, List, Tuple
import warnings
warnings.filterwarnings('ignore')

# Data path
DB_PATH = '/Users/nish_macbook/development/trading/emini-futures/data/emini_futures_15min.db'
MES_POINT_VALUE = 5.0


def load_es_15min_data(rth_only: bool = True) -> pd.DataFrame:
    """Load ES 15-min data from SQLite database."""
    conn = sqlite3.connect(DB_PATH)
    query = """
    SELECT timestamp, open, high, low, close, volume
    FROM es_15min_bars
    ORDER BY timestamp
    """
    df = pd.read_sql_query(query, conn)
    conn.close()

    df['timestamp'] = pd.to_datetime(df['timestamp'], utc=True)
    df['timestamp'] = df['timestamp'].dt.tz_convert('America/New_York')
    df = df.set_index('timestamp').sort_index()

    if rth_only:
        df = df.between_time('09:30', '15:59')

    return df


def calculate_rsi(prices: pd.Series, period: int = 14) -> pd.Series:
    """Calculate RSI indicator."""
    delta = prices.diff()
    gain = delta.where(delta > 0, 0).rolling(period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(period).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))


def calculate_atr(df: pd.DataFrame, period: int = 20) -> pd.Series:
    """Calculate Average True Range."""
    tr = pd.concat([
        df['high'] - df['low'],
        abs(df['high'] - df['close'].shift(1)),
        abs(df['low'] - df['close'].shift(1))
    ], axis=1).max(axis=1)
    return tr.rolling(period).mean()


def backtest_with_filters(
    df: pd.DataFrame,
    # RSI parameters
    rsi_entry: float = 45,
    profit_target: float = 1.0,
    stop_loss: float = 0.15,
    max_bars: int = 6,
    # Volume filter
    volume_min_mult: float = None,  # Skip if volume < this * 20-bar MA
    # Time filter
    skip_first_mins: int = None,    # Skip first N mins after open
    skip_last_mins: int = None,     # Skip last N mins before close
    # Volatility filter
    atr_max_mult: float = None,     # Skip if ATR > this * 50-bar MA
) -> Dict:
    """
    Run backtest with optional filters.
    All filters use "exclusion" logic - skip bad conditions, keep the rest.
    """
    df = df.copy()
    df['rsi'] = calculate_rsi(df['close'], 14)
    df['volume_ma'] = df['volume'].rolling(20).mean()
    df['atr'] = calculate_atr(df, 20)
    df['atr_ma'] = df['atr'].rolling(50).mean()
    df = df.dropna()

    trades = []
    position = None
    skipped = {'volume': 0, 'time': 0, 'atr': 0}

    for i in range(1, len(df)):
        bar = df.iloc[i]
        bar_time = bar.name.time()

        # Manage existing position
        if position is not None:
            pnl_pct = (bar['close'] - position['entry']) / position['entry'] * 100
            bars_held = i - position['entry_idx']

            exit_reason = None
            if pnl_pct >= profit_target:
                exit_reason = 'target'
                final_pnl = profit_target
            elif pnl_pct <= -stop_loss:
                exit_reason = 'stop'
                final_pnl = -stop_loss
            elif bars_held >= max_bars:
                exit_reason = 'time'
                final_pnl = pnl_pct

            if exit_reason:
                dollar_pnl = position['entry'] * (final_pnl / 100) * MES_POINT_VALUE
                trades.append({
                    'entry_date': position['entry_date'],
                    'entry_price': position['entry'],
                    'exit_date': bar.name,
                    'exit_price': bar['close'],
                    'pnl_pct': final_pnl,
                    'pnl_dollars': dollar_pnl,
                    'bars_held': bars_held,
                    'exit_reason': exit_reason
                })
                position = None
                continue

        # Check for new entry
        if position is None and bar['rsi'] < rsi_entry:
            # Apply filters (exclusion logic)

            # Volume filter: skip dead volume
            if volume_min_mult is not None:
                if bar['volume'] < bar['volume_ma'] * volume_min_mult:
                    skipped['volume'] += 1
                    continue

            # Time filter: skip first/last minutes
            if skip_first_mins is not None:
                # Calculate cutoff time after market open (9:30)
                cutoff_hour = 9
                cutoff_min = 30 + skip_first_mins
                while cutoff_min >= 60:
                    cutoff_hour += 1
                    cutoff_min -= 60
                cutoff = time(cutoff_hour, cutoff_min)
                if bar_time < cutoff:
                    skipped['time'] += 1
                    continue

            if skip_last_mins is not None:
                # Calculate cutoff time before market close (16:00)
                cutoff_hour = 15
                cutoff_min = 60 - skip_last_mins
                while cutoff_min < 0:
                    cutoff_hour -= 1
                    cutoff_min += 60
                cutoff = time(cutoff_hour, cutoff_min)
                if bar_time >= cutoff:
                    skipped['time'] += 1
                    continue

            # ATR filter: skip extreme volatility
            if atr_max_mult is not None:
                if bar['atr'] > bar['atr_ma'] * atr_max_mult:
                    skipped['atr'] += 1
                    continue

            # Entry passed all filters
            position = {
                'entry': bar['close'],
                'entry_date': bar.name,
                'entry_idx': i
            }

    return calculate_metrics(trades, df, skipped)


def calculate_metrics(trades: List[Dict], df: pd.DataFrame, skipped: Dict = None) -> Dict:
    """Calculate performance metrics from trades."""
    if not trades:
        return {'trades': 0, 'annual_return': 0, 'sharpe': 0, 'total_dollar_pnl': 0, 'skipped': skipped}

    pnl_pcts = np.array([t['pnl_pct'] for t in trades])
    pnl_dollars = np.array([t['pnl_dollars'] for t in trades])

    total_mult = 1.0
    for pnl in pnl_pcts:
        total_mult *= (1 + pnl / 100)
    total_return = (total_mult - 1) * 100

    first_date = trades[0]['entry_date']
    last_date = trades[-1]['exit_date']
    years = (last_date - first_date).days / 365

    annual_return = (total_mult ** (1/years) - 1) * 100 if years > 0 else 0

    trades_per_year = len(trades) / years if years > 0 else 0
    if np.std(pnl_pcts) > 0 and trades_per_year > 0:
        sharpe = (np.mean(pnl_pcts) / np.std(pnl_pcts)) * np.sqrt(trades_per_year)
    else:
        sharpe = 0

    wins = pnl_pcts[pnl_pcts > 0]
    losses = pnl_pcts[pnl_pcts < 0]
    win_rate = len(wins) / len(pnl_pcts) * 100

    cumsum = np.cumsum(pnl_pcts)
    running_max = np.maximum.accumulate(cumsum)
    max_dd = np.min(cumsum - running_max)

    return {
        'trades': len(trades),
        'win_rate': win_rate,
        'total_return': total_return,
        'annual_return': annual_return,
        'sharpe': sharpe,
        'max_dd': max_dd,
        'total_dollar_pnl': np.sum(pnl_dollars),
        'avg_dollar_pnl': np.mean(pnl_dollars),
        'skipped': skipped
    }


def run_filter_sweep(df: pd.DataFrame) -> List[Dict]:
    """Test all filter combinations and return results."""
    results = []

    # Baseline
    baseline = backtest_with_filters(df)
    baseline_trades = baseline['trades']

    results.append({
        'name': 'Baseline (No Filters)',
        'volume_min': None,
        'skip_first': None,
        'skip_last': None,
        'atr_max': None,
        **baseline,
        'trades_pct': 100.0,
        'freq_sharpe_score': baseline['sharpe']
    })

    # Volume filter variations
    for vol_mult in [0.3, 0.5, 0.7, 1.0]:
        m = backtest_with_filters(df, volume_min_mult=vol_mult)
        trades_pct = m['trades'] / baseline_trades * 100
        results.append({
            'name': f'Volume > {vol_mult}x avg',
            'volume_min': vol_mult,
            'skip_first': None,
            'skip_last': None,
            'atr_max': None,
            **m,
            'trades_pct': trades_pct,
            'freq_sharpe_score': (trades_pct / 100) * m['sharpe']
        })

    # Time filter variations
    for skip_mins in [15, 30, 45]:
        m = backtest_with_filters(df, skip_first_mins=skip_mins, skip_last_mins=skip_mins)
        trades_pct = m['trades'] / baseline_trades * 100
        results.append({
            'name': f'Skip first/last {skip_mins}m',
            'volume_min': None,
            'skip_first': skip_mins,
            'skip_last': skip_mins,
            'atr_max': None,
            **m,
            'trades_pct': trades_pct,
            'freq_sharpe_score': (trades_pct / 100) * m['sharpe']
        })

    # ATR filter variations
    for atr_mult in [1.5, 2.0, 2.5, 3.0]:
        m = backtest_with_filters(df, atr_max_mult=atr_mult)
        trades_pct = m['trades'] / baseline_trades * 100
        results.append({
            'name': f'ATR < {atr_mult}x avg',
            'volume_min': None,
            'skip_first': None,
            'skip_last': None,
            'atr_max': atr_mult,
            **m,
            'trades_pct': trades_pct,
            'freq_sharpe_score': (trades_pct / 100) * m['sharpe']
        })

    return results


def run_combination_sweep(df: pd.DataFrame, best_volume: float, best_time: int, best_atr: float) -> List[Dict]:
    """Test combinations of best individual filters."""
    results = []
    baseline = backtest_with_filters(df)
    baseline_trades = baseline['trades']

    # Volume + Time
    m = backtest_with_filters(df, volume_min_mult=best_volume,
                              skip_first_mins=best_time, skip_last_mins=best_time)
    trades_pct = m['trades'] / baseline_trades * 100
    results.append({
        'name': f'Vol>{best_volume}x + Skip {best_time}m',
        **m,
        'trades_pct': trades_pct,
        'freq_sharpe_score': (trades_pct / 100) * m['sharpe']
    })

    # Volume + ATR
    m = backtest_with_filters(df, volume_min_mult=best_volume, atr_max_mult=best_atr)
    trades_pct = m['trades'] / baseline_trades * 100
    results.append({
        'name': f'Vol>{best_volume}x + ATR<{best_atr}x',
        **m,
        'trades_pct': trades_pct,
        'freq_sharpe_score': (trades_pct / 100) * m['sharpe']
    })

    # Time + ATR
    m = backtest_with_filters(df, skip_first_mins=best_time, skip_last_mins=best_time,
                              atr_max_mult=best_atr)
    trades_pct = m['trades'] / baseline_trades * 100
    results.append({
        'name': f'Skip {best_time}m + ATR<{best_atr}x',
        **m,
        'trades_pct': trades_pct,
        'freq_sharpe_score': (trades_pct / 100) * m['sharpe']
    })

    # All three
    m = backtest_with_filters(df, volume_min_mult=best_volume,
                              skip_first_mins=best_time, skip_last_mins=best_time,
                              atr_max_mult=best_atr)
    trades_pct = m['trades'] / baseline_trades * 100
    results.append({
        'name': f'All Three Combined',
        **m,
        'trades_pct': trades_pct,
        'freq_sharpe_score': (trades_pct / 100) * m['sharpe']
    })

    return results


if __name__ == '__main__':
    print("=" * 90)
    print("MES 15-MIN RSI MEAN REVERSION - FILTER OPTIMIZATION")
    print("Priority: Maximize Trade Frequency While Improving Quality")
    print("=" * 90)

    # Load data
    df = load_es_15min_data()
    print(f"\nData: {len(df):,} bars ({df.index[0].date()} to {df.index[-1].date()})")

    # Phase 1: Individual filter tests
    print("\n" + "=" * 90)
    print("PHASE 1: INDIVIDUAL FILTER TESTS")
    print("=" * 90)

    results = run_filter_sweep(df)

    print(f"\n{'Configuration':<30} {'Trades':>7} {'Trade%':>7} {'Sharpe':>7} {'Annual':>8} {'$P&L':>10} {'Score':>7}")
    print("-" * 90)

    for r in results:
        print(f"{r['name']:<30} {r['trades']:>7d} {r['trades_pct']:>6.1f}% {r['sharpe']:>6.2f} "
              f"{r['annual_return']:>7.1f}% ${r['total_dollar_pnl']:>9,.0f} {r['freq_sharpe_score']:>6.2f}")

    # Find best of each filter type (by freq_sharpe_score)
    volume_results = [r for r in results if r.get('volume_min') is not None]
    time_results = [r for r in results if r.get('skip_first') is not None]
    atr_results = [r for r in results if r.get('atr_max') is not None]

    best_volume = max(volume_results, key=lambda x: x['freq_sharpe_score']) if volume_results else None
    best_time = max(time_results, key=lambda x: x['freq_sharpe_score']) if time_results else None
    best_atr = max(atr_results, key=lambda x: x['freq_sharpe_score']) if atr_results else None

    print("\n" + "-" * 90)
    print("BEST INDIVIDUAL FILTERS (by Trade% × Sharpe Score):")
    if best_volume:
        print(f"  Volume: {best_volume['name']} → Score {best_volume['freq_sharpe_score']:.2f}")
    if best_time:
        print(f"  Time:   {best_time['name']} → Score {best_time['freq_sharpe_score']:.2f}")
    if best_atr:
        print(f"  ATR:    {best_atr['name']} → Score {best_atr['freq_sharpe_score']:.2f}")

    # Phase 2: Combination tests
    if best_volume and best_time and best_atr:
        print("\n" + "=" * 90)
        print("PHASE 2: FILTER COMBINATIONS")
        print("=" * 90)

        combo_results = run_combination_sweep(
            df,
            best_volume['volume_min'],
            best_time['skip_first'],
            best_atr['atr_max']
        )

        print(f"\n{'Configuration':<35} {'Trades':>7} {'Trade%':>7} {'Sharpe':>7} {'Annual':>8} {'$P&L':>10} {'Score':>7}")
        print("-" * 90)

        for r in combo_results:
            print(f"{r['name']:<35} {r['trades']:>7d} {r['trades_pct']:>6.1f}% {r['sharpe']:>6.2f} "
                  f"{r['annual_return']:>7.1f}% ${r['total_dollar_pnl']:>9,.0f} {r['freq_sharpe_score']:>6.2f}")

    # Final recommendation
    print("\n" + "=" * 90)
    print("FINAL RECOMMENDATION")
    print("=" * 90)

    all_results = results + (combo_results if best_volume and best_time and best_atr else [])

    # Filter to configs that keep >= 85% of trades
    high_freq = [r for r in all_results if r['trades_pct'] >= 85]

    if high_freq:
        best_overall = max(high_freq, key=lambda x: x['sharpe'])
        baseline = results[0]

        print(f"\n🏆 WINNER (≥85% trade retention): {best_overall['name']}")
        print(f"\n{'Metric':<20} {'Baseline':>12} {'Enhanced':>12} {'Change':>12}")
        print("-" * 60)
        print(f"{'Trades':<20} {baseline['trades']:>12d} {best_overall['trades']:>12d} {best_overall['trades'] - baseline['trades']:>+12d}")
        print(f"{'Trade Retention':<20} {'100%':>12} {best_overall['trades_pct']:>11.1f}% {best_overall['trades_pct'] - 100:>+11.1f}%")
        print(f"{'Sharpe':<20} {baseline['sharpe']:>12.2f} {best_overall['sharpe']:>12.2f} {best_overall['sharpe'] - baseline['sharpe']:>+12.2f}")
        print(f"{'Annual Return':<20} {baseline['annual_return']:>11.1f}% {best_overall['annual_return']:>11.1f}% {best_overall['annual_return'] - baseline['annual_return']:>+11.1f}%")
        print(f"{'$P&L/Contract':<20} ${baseline['total_dollar_pnl']:>11,.0f} ${best_overall['total_dollar_pnl']:>11,.0f} ${best_overall['total_dollar_pnl'] - baseline['total_dollar_pnl']:>+11,.0f}")
    else:
        print("\nNo configuration maintains ≥85% trades. Consider looser thresholds.")
