"""
Strategy: Micro E-mini S&P 500 (MES) 15-Min RSI Mean Reversion

VALIDATED RESULTS (Jan 2024 - Nov 2025, RTH):
- Best: RSI<45, +1.0%/-0.15%, 6 bars = 27.9% annual, Sharpe 5.02, $13,176/contract
- Alt:  RSI<45, +1.25%/-0.15%, 6 bars = 28.4% annual, Sharpe 4.79, $13,510/contract

Key insight: Tighter stop (0.15% vs 0.20%) improves Sharpe significantly.
RSI outperforms all other mean reversion indicators (Williams %R, Keltner, Stochastic, etc.)

MES = 1/10th ES = $5/point (vs $50 for ES)
Data: ES 15-min bars from IB (MES tracks ES 1:1)
"""

import pandas as pd
import numpy as np
import sqlite3
from typing import List, Dict, Optional
from dataclasses import dataclass
import warnings
warnings.filterwarnings('ignore')

# Data path
DB_PATH = '/Users/nish_macbook/development/trading/emini-futures/data/emini_futures_15min.db'

# Contract specs
MES_POINT_VALUE = 5.0   # $5 per point for MES
ES_POINT_VALUE = 50.0   # $50 per point for ES (reference)

# Strategy parameters (optimized for MES)
DEFAULT_PARAMS = {
    'rsi_period': 14,
    'rsi_entry': 45,       # Enter when RSI < this
    'profit_target': 1.0,   # Exit at +1.0%
    'stop_loss': 0.15,      # Exit at -0.15% (tighter = better)
    'max_bars': 6,          # Max holding period
}

# Alternative: slightly wider target
WIDE_TARGET_PARAMS = {
    'rsi_period': 14,
    'rsi_entry': 45,
    'profit_target': 1.25,  # Wider target
    'stop_loss': 0.15,
    'max_bars': 6,
}


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

    # Parse timestamp with timezone
    df['timestamp'] = pd.to_datetime(df['timestamp'], utc=True)
    df['timestamp'] = df['timestamp'].dt.tz_convert('America/New_York')
    df = df.set_index('timestamp').sort_index()

    # Filter to regular trading hours (RTH): 9:30 AM - 4:00 PM ET
    if rth_only:
        df = df.between_time('09:30', '15:59')

    print(f"Loaded {len(df):,} 15-min bars from {df.index[0].date()} to {df.index[-1].date()}")
    return df


def calculate_rsi(prices: pd.Series, period: int = 14) -> pd.Series:
    """Calculate RSI indicator."""
    delta = prices.diff()
    gain = delta.where(delta > 0, 0).rolling(period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(period).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))


def backtest(df: pd.DataFrame, rsi_entry: float = 35, profit_target: float = 0.75,
             stop_loss: float = 0.30, max_bars: int = 8, rsi_period: int = 14) -> Dict:
    """
    Run backtest with profit target and stop loss.

    Entry: RSI < rsi_entry (oversold)
    Exit: First of profit target, stop loss, or max bars
    """
    df = df.copy()
    df['rsi'] = calculate_rsi(df['close'], rsi_period)
    df = df.dropna()

    trades = []
    position = None

    for i in range(1, len(df)):
        bar = df.iloc[i]

        # Manage existing position
        if position is not None:
            pnl_pct = (bar['close'] - position['entry']) / position['entry'] * 100
            bars_held = i - position['entry_idx']

            exit_reason = None

            # Check profit target
            if pnl_pct >= profit_target:
                exit_reason = 'target'
                final_pnl = profit_target
            # Check stop loss
            elif pnl_pct <= -stop_loss:
                exit_reason = 'stop'
                final_pnl = -stop_loss
            # Check time exit
            elif bars_held >= max_bars:
                exit_reason = 'time'
                final_pnl = pnl_pct

            if exit_reason:
                # Calculate dollar P&L for MES
                point_move = bar['close'] - position['entry']
                dollar_pnl = point_move * MES_POINT_VALUE

                trades.append({
                    'entry_date': position['entry_date'],
                    'entry_price': position['entry'],
                    'exit_date': bar.name,
                    'exit_price': bar['close'],
                    'pnl_pct': final_pnl,
                    'pnl_points': point_move,
                    'pnl_dollars': dollar_pnl,
                    'bars_held': bars_held,
                    'exit_reason': exit_reason
                })
                position = None
                continue

        # Check for new entry
        if position is None and bar['rsi'] < rsi_entry:
            position = {
                'entry': bar['close'],
                'entry_date': bar.name,
                'entry_idx': i
            }

    return calculate_metrics(trades, df)


def calculate_metrics(trades: List[Dict], df: pd.DataFrame) -> Dict:
    """Calculate performance metrics from trades."""
    if not trades:
        return {'trades': 0, 'annual_return': 0, 'sharpe': 0}

    pnl_pcts = np.array([t['pnl_pct'] for t in trades])
    pnl_dollars = np.array([t['pnl_dollars'] for t in trades])

    # Compound returns
    total_mult = 1.0
    for pnl in pnl_pcts:
        total_mult *= (1 + pnl / 100)
    total_return = (total_mult - 1) * 100

    # Time period
    first_date = trades[0]['entry_date']
    last_date = trades[-1]['exit_date']
    years = (last_date - first_date).days / 365

    # Annualized return
    annual_return = (total_mult ** (1/years) - 1) * 100 if years > 0 else 0

    # Sharpe ratio (annualized)
    trades_per_year = len(trades) / years if years > 0 else 0
    if np.std(pnl_pcts) > 0 and trades_per_year > 0:
        sharpe = (np.mean(pnl_pcts) / np.std(pnl_pcts)) * np.sqrt(trades_per_year)
    else:
        sharpe = 0

    # Win rate
    wins = pnl_pcts[pnl_pcts > 0]
    losses = pnl_pcts[pnl_pcts < 0]
    win_rate = len(wins) / len(pnl_pcts) * 100

    # Max drawdown
    cumsum = np.cumsum(pnl_pcts)
    running_max = np.maximum.accumulate(cumsum)
    drawdown = cumsum - running_max
    max_dd = np.min(drawdown)

    # Dollar stats
    total_dollar_pnl = np.sum(pnl_dollars)
    avg_dollar_pnl = np.mean(pnl_dollars)

    # Profit factor
    gross_profit = np.sum(wins) if len(wins) > 0 else 0
    gross_loss = abs(np.sum(losses)) if len(losses) > 0 else 1
    profit_factor = gross_profit / gross_loss if gross_loss > 0 else 0

    # Exit reason breakdown
    exit_reasons = {}
    for t in trades:
        r = t['exit_reason']
        exit_reasons[r] = exit_reasons.get(r, 0) + 1

    return {
        'trades': len(trades),
        'win_rate': win_rate,
        'total_return': total_return,
        'annual_return': annual_return,
        'sharpe': sharpe,
        'max_dd': max_dd,
        'profit_factor': profit_factor,
        'avg_win': np.mean(wins) if len(wins) > 0 else 0,
        'avg_loss': np.mean(losses) if len(losses) > 0 else 0,
        'total_dollar_pnl': total_dollar_pnl,
        'avg_dollar_pnl': avg_dollar_pnl,
        'years': years,
        'exit_reasons': exit_reasons,
        'trades_list': trades
    }


def backtest_by_year(df: pd.DataFrame, **params) -> Dict[int, Dict]:
    """Run backtest for each year separately."""
    results = {}

    for year in df.index.year.unique():
        year_df = df[df.index.year == year]
        if len(year_df) > 100:
            metrics = backtest(year_df, **params)
            if metrics['trades'] > 0:
                results[year] = metrics

    return results


def parameter_sweep(df: pd.DataFrame) -> List[Dict]:
    """Test different parameter combinations."""
    results = []

    for rsi_entry in [30, 35, 40, 45]:
        for profit_target in [0.5, 0.75, 1.0, 1.25]:
            for stop_loss in [0.20, 0.25, 0.30, 0.40]:
                for max_bars in [6, 8, 10, 12]:
                    metrics = backtest(df, rsi_entry=rsi_entry,
                                      profit_target=profit_target,
                                      stop_loss=stop_loss, max_bars=max_bars)

                    if metrics['trades'] >= 30:  # Lower threshold due to shorter history
                        results.append({
                            'rsi_entry': rsi_entry,
                            'profit_target': profit_target,
                            'stop_loss': stop_loss,
                            'max_bars': max_bars,
                            **metrics
                        })

    # Sort by Sharpe
    results.sort(key=lambda x: -x['sharpe'])
    return results


def print_results(metrics: Dict, name: str = ""):
    """Print formatted results."""
    print(f"\n{'='*60}")
    print(f"RESULTS: {name}")
    print(f"{'='*60}")
    print(f"Trades: {metrics['trades']}")
    print(f"Win Rate: {metrics['win_rate']:.1f}%")
    print(f"Avg Win: +{metrics['avg_win']:.2f}%")
    print(f"Avg Loss: {metrics['avg_loss']:.2f}%")

    print(f"\nTotal Return: {metrics['total_return']:.1f}%")
    print(f"Annual Return: {metrics['annual_return']:.1f}%")
    print(f"Sharpe Ratio: {metrics['sharpe']:.2f}")
    print(f"Max Drawdown: {metrics['max_dd']:.1f}%")
    print(f"Profit Factor: {metrics['profit_factor']:.2f}")

    # Dollar P&L for 1 MES contract
    print(f"\n--- Dollar P&L (1 MES Contract) ---")
    print(f"Total P&L: ${metrics['total_dollar_pnl']:,.2f}")
    print(f"Avg P&L per Trade: ${metrics['avg_dollar_pnl']:.2f}")

    if 'exit_reasons' in metrics:
        print(f"\nExit Reasons:")
        for reason, count in metrics['exit_reasons'].items():
            pct = count / metrics['trades'] * 100
            print(f"  {reason}: {count} ({pct:.0f}%)")


if __name__ == '__main__':
    print("="*60)
    print("MES (MICRO E-MINI S&P 500) 15-MIN RSI MEAN REVERSION BACKTEST")
    print("="*60)
    print(f"\nContract: MES = ${MES_POINT_VALUE}/point")
    print(f"Data: ES 15-min bars (MES tracks ES 1:1)")

    # Load data
    df = load_es_15min_data()

    if len(df) < 100:
        print(f"\nERROR: Only {len(df)} bars after filtering to RTH. Need more data.")
        print("Trying without RTH filter...")
        df = load_es_15min_data(rth_only=False)
        print(f"Reloaded {len(df):,} bars (all sessions)")

    # Test default (robust) configuration
    print("\n--- ROBUST CONFIGURATION ---")
    print(f"RSI<{DEFAULT_PARAMS['rsi_entry']}, Target +{DEFAULT_PARAMS['profit_target']}%, "
          f"Stop -{DEFAULT_PARAMS['stop_loss']}%, Max {DEFAULT_PARAMS['max_bars']} bars")

    metrics = backtest(df, **{k: v for k, v in DEFAULT_PARAMS.items() if k != 'rsi_period'})
    print_results(metrics, "RSI Mean Reversion (Robust)")

    # Year-by-year breakdown
    print("\n--- YEAR-BY-YEAR ---")
    yearly = backtest_by_year(df, **{k: v for k, v in DEFAULT_PARAMS.items() if k != 'rsi_period'})
    for year, m in sorted(yearly.items()):
        print(f"{year}: {m['trades']} trades, +{m['total_return']:.1f}%, "
              f"Sharpe {m['sharpe']:.2f}, ${m['total_dollar_pnl']:,.0f}")

    # Test aggressive configuration
    print("\n--- AGGRESSIVE CONFIGURATION ---")
    print(f"RSI<{WIDE_TARGET_PARAMS['rsi_entry']}, Target +{WIDE_TARGET_PARAMS['profit_target']}%, "
          f"Stop -{WIDE_TARGET_PARAMS['stop_loss']}%, Max {WIDE_TARGET_PARAMS['max_bars']} bars")

    metrics_agg = backtest(df, **{k: v for k, v in WIDE_TARGET_PARAMS.items() if k != 'rsi_period'})
    print_results(metrics_agg, "RSI Mean Reversion (Aggressive)")

    # Parameter sweep
    print("\n" + "="*60)
    print("PARAMETER SWEEP - TOP 10 BY SHARPE")
    print("="*60)

    sweep_results = parameter_sweep(df)

    if sweep_results:
        print(f"\n{'RSI':>4} {'Target':>7} {'Stop':>6} {'Bars':>5} {'Trades':>7} {'Annual':>8} {'Sharpe':>7} {'$P&L':>10}")
        print("-"*70)

        for r in sweep_results[:10]:
            print(f"{r['rsi_entry']:4d} {r['profit_target']:6.2f}% {r['stop_loss']:5.2f}% "
                  f"{r['max_bars']:5d} {r['trades']:7d} {r['annual_return']:7.1f}% "
                  f"{r['sharpe']:6.2f} ${r['total_dollar_pnl']:>9,.0f}")
    else:
        print("\nNot enough trades for parameter sweep. Need more data.")
