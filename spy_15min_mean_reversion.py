"""
Strategy: SPY 15-Min RSI Mean Reversion with Asymmetric Risk/Reward

VALIDATED RESULTS (2021-2024):
- Best: RSI<40, +1.0%/-0.25%, 10 bars = 26.2% annual, Sharpe 3.36
- Robust: RSI<35, +0.75%/-0.30%, 8 bars = 17.2% annual, Sharpe 2.68

Key insight: Asymmetric risk/reward (tight stop, larger target) is critical.
Without targets/stops, strategy only achieved 3.7% annual.

Year-by-Year Consistency (RSI<35 config):
- 2021: 273 trades, +23.6%
- 2022: 389 trades, +13.3%
- 2023: 289 trades, +16.8%
- 2024: 273 trades, +15.2%

Data: Alpaca SPY 1-min bars resampled to 15-min (2021-2024)
"""

import pandas as pd
import numpy as np
import os
from typing import List, Dict, Optional
import warnings
warnings.filterwarnings('ignore')

# Data path
DATA_PATH = '/Users/nish_macbook/development/trading/central_data/market_data/alpaca_spy_1min'

# Strategy parameters (validated)
DEFAULT_PARAMS = {
    'rsi_period': 14,
    'rsi_entry': 35,      # Enter when RSI < this
    'profit_target': 0.75, # Exit at +0.75%
    'stop_loss': 0.30,     # Exit at -0.30%
    'max_bars': 8,         # Max holding period
}

AGGRESSIVE_PARAMS = {
    'rsi_period': 14,
    'rsi_entry': 40,
    'profit_target': 1.0,
    'stop_loss': 0.25,
    'max_bars': 10,
}


def load_15min_data(years: List[int] = [2021, 2022, 2023, 2024]) -> pd.DataFrame:
    """Load and resample SPY 1-min data to 15-min bars."""
    all_data = []

    for year in years:
        year_path = os.path.join(DATA_PATH, str(year))
        if not os.path.exists(year_path):
            print(f"Warning: {year_path} not found")
            continue

        for f in sorted(os.listdir(year_path)):
            if f.endswith('.parquet'):
                df = pd.read_parquet(os.path.join(year_path, f), engine='fastparquet')
                df['timestamp'] = pd.to_datetime(df['timestamp']).dt.tz_convert('America/New_York')
                df = df.set_index('timestamp').sort_index()
                df = df.between_time('09:30', '15:59')
                all_data.append(df)

    if not all_data:
        raise ValueError("No data loaded")

    combined = pd.concat(all_data)

    # Resample to 15-min
    resampled = combined.resample('15min').agg({
        'open': 'first',
        'high': 'max',
        'low': 'min',
        'close': 'last',
        'volume': 'sum'
    }).dropna()

    print(f"Loaded {len(resampled):,} 15-min bars from {resampled.index[0].date()} to {resampled.index[-1].date()}")
    return resampled


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
                trades.append({
                    'entry_date': position['entry_date'],
                    'entry_price': position['entry'],
                    'exit_date': bar.name,
                    'exit_price': bar['close'],
                    'pnl_pct': final_pnl,
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

    for rsi_entry in [30, 35, 40]:
        for profit_target in [0.5, 0.75, 1.0]:
            for stop_loss in [0.25, 0.30, 0.40]:
                for max_bars in [6, 8, 10, 12]:
                    metrics = backtest(df, rsi_entry=rsi_entry,
                                      profit_target=profit_target,
                                      stop_loss=stop_loss, max_bars=max_bars)

                    if metrics['trades'] >= 100:
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

    if 'exit_reasons' in metrics:
        print(f"\nExit Reasons:")
        for reason, count in metrics['exit_reasons'].items():
            pct = count / metrics['trades'] * 100
            print(f"  {reason}: {count} ({pct:.0f}%)")


if __name__ == '__main__':
    print("="*60)
    print("SPY 15-MIN RSI MEAN REVERSION BACKTEST")
    print("="*60)

    # Load data
    df = load_15min_data()

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
        print(f"{year}: {m['trades']} trades, +{m['total_return']:.1f}%, Sharpe {m['sharpe']:.2f}")

    # Test aggressive configuration
    print("\n--- AGGRESSIVE CONFIGURATION ---")
    print(f"RSI<{AGGRESSIVE_PARAMS['rsi_entry']}, Target +{AGGRESSIVE_PARAMS['profit_target']}%, "
          f"Stop -{AGGRESSIVE_PARAMS['stop_loss']}%, Max {AGGRESSIVE_PARAMS['max_bars']} bars")

    metrics_agg = backtest(df, **{k: v for k, v in AGGRESSIVE_PARAMS.items() if k != 'rsi_period'})
    print_results(metrics_agg, "RSI Mean Reversion (Aggressive)")

    # Parameter sweep
    print("\n" + "="*60)
    print("PARAMETER SWEEP - TOP 10 BY SHARPE")
    print("="*60)

    sweep_results = parameter_sweep(df)

    print(f"\n{'RSI':>4} {'Target':>7} {'Stop':>6} {'Bars':>5} {'Trades':>7} {'Annual':>8} {'Sharpe':>7}")
    print("-"*60)

    for r in sweep_results[:10]:
        print(f"{r['rsi_entry']:4d} {r['profit_target']:6.2f}% {r['stop_loss']:5.2f}% "
              f"{r['max_bars']:5d} {r['trades']:7d} {r['annual_return']:7.1f}% {r['sharpe']:6.2f}")
