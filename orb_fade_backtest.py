"""
Strategy #1: 0DTE ORB Fade / False Break Reversal

Edge: Opening breakouts fail ~30-50% depending on regime
Trigger: ORB break <0.25% beyond range then price re-enters ORB
Entry/Exit: Fade back toward ORB midpoint; exit on VWAP or range edge
Best Regime: Low-moderate volatility, non-trend days

Data: Alpaca SPY 1-min bars (2020-2025)
"""

import pandas as pd
import numpy as np
import os
from datetime import datetime, time
from dataclasses import dataclass
from typing import List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

# Configuration
DATA_PATH = '/Users/nish_macbook/development/trading/central_data/market_data/alpaca_spy_1min'
ORB_MINUTES = 15  # First 15 minutes for Opening Range
FALSE_BREAK_THRESHOLD = 0.0025  # 0.25% max extension beyond ORB
MARKET_OPEN = time(9, 30)  # ET
MARKET_CLOSE = time(16, 0)  # ET


@dataclass
class Trade:
    entry_time: datetime
    entry_price: float
    direction: str  # 'long' or 'short'
    orb_high: float
    orb_low: float
    orb_mid: float
    exit_time: Optional[datetime] = None
    exit_price: Optional[float] = None
    exit_reason: Optional[str] = None
    pnl: Optional[float] = None
    pnl_pct: Optional[float] = None


def load_day_data(filepath: str) -> pd.DataFrame:
    """Load single day of SPY 1-min data."""
    df = pd.read_parquet(filepath, engine='fastparquet')
    df['timestamp'] = pd.to_datetime(df['timestamp'])

    # Convert UTC to ET
    df['timestamp'] = df['timestamp'].dt.tz_convert('America/New_York')
    df = df.set_index('timestamp').sort_index()

    return df


def get_market_hours(df: pd.DataFrame) -> pd.DataFrame:
    """Filter to regular market hours (9:30 - 16:00 ET)."""
    df = df.copy()
    df['time'] = df.index.time
    mask = (df['time'] >= MARKET_OPEN) & (df['time'] < MARKET_CLOSE)
    return df[mask].drop(columns=['time'])


def calculate_orb(df: pd.DataFrame, minutes: int = 15) -> Tuple[float, float, float]:
    """Calculate Opening Range (high, low, midpoint) for first N minutes."""
    market_df = get_market_hours(df)
    if len(market_df) < minutes:
        return None, None, None

    orb_data = market_df.iloc[:minutes]
    orb_high = orb_data['high'].max()
    orb_low = orb_data['low'].min()
    orb_mid = (orb_high + orb_low) / 2

    return orb_high, orb_low, orb_mid


def calculate_vwap(df: pd.DataFrame) -> pd.Series:
    """Calculate cumulative VWAP from market open."""
    market_df = get_market_hours(df)
    typical_price = (market_df['high'] + market_df['low'] + market_df['close']) / 3
    cum_vol = market_df['volume'].cumsum()
    cum_tp_vol = (typical_price * market_df['volume']).cumsum()
    vwap = cum_tp_vol / cum_vol
    return vwap


def detect_false_break(df: pd.DataFrame, orb_high: float, orb_low: float,
                       threshold: float = FALSE_BREAK_THRESHOLD) -> List[dict]:
    """
    Detect false breakouts where price:
    1. Breaks ORB by less than threshold
    2. Re-enters the ORB range

    Returns list of potential trade signals.
    """
    market_df = get_market_hours(df)
    orb_range = orb_high - orb_low
    max_extension = orb_range * (threshold / 0.01)  # Convert % to price

    signals = []
    in_breakout = None  # 'up' or 'down' or None
    breakout_extreme = None

    # Start after ORB period
    for i in range(ORB_MINUTES, len(market_df)):
        bar = market_df.iloc[i]
        prev_bar = market_df.iloc[i-1]

        # Check for upside breakout
        if bar['high'] > orb_high and in_breakout != 'up':
            extension = bar['high'] - orb_high
            if extension <= max_extension:
                in_breakout = 'up'
                breakout_extreme = bar['high']

        # Check for downside breakout
        elif bar['low'] < orb_low and in_breakout != 'down':
            extension = orb_low - bar['low']
            if extension <= max_extension:
                in_breakout = 'down'
                breakout_extreme = bar['low']

        # Check for re-entry (false breakout signal)
        if in_breakout == 'up' and bar['close'] < orb_high:
            signals.append({
                'time': bar.name,
                'price': bar['close'],
                'direction': 'short',  # Fade the failed upside break
                'breakout_type': 'up',
                'extension': breakout_extreme - orb_high
            })
            in_breakout = None

        elif in_breakout == 'down' and bar['close'] > orb_low:
            signals.append({
                'time': bar.name,
                'price': bar['close'],
                'direction': 'long',  # Fade the failed downside break
                'breakout_type': 'down',
                'extension': orb_low - breakout_extreme
            })
            in_breakout = None

    return signals


def execute_trade(signal: dict, df: pd.DataFrame, orb_high: float,
                  orb_low: float, orb_mid: float) -> Trade:
    """
    Execute trade based on signal:
    - Entry: At signal price
    - Target: ORB midpoint
    - Stop: Beyond ORB range (opposite edge)
    """
    market_df = get_market_hours(df)
    vwap = calculate_vwap(df)

    trade = Trade(
        entry_time=signal['time'],
        entry_price=signal['price'],
        direction=signal['direction'],
        orb_high=orb_high,
        orb_low=orb_low,
        orb_mid=orb_mid
    )

    # Find exit
    entry_idx = market_df.index.get_loc(signal['time'])

    for i in range(entry_idx + 1, len(market_df)):
        bar = market_df.iloc[i]
        current_vwap = vwap.iloc[i] if i < len(vwap) else vwap.iloc[-1]

        if trade.direction == 'short':
            # Target: ORB midpoint or VWAP (whichever is closer)
            target = min(orb_mid, current_vwap)
            stop = orb_high + (orb_high - orb_low) * 0.5  # 50% above range

            if bar['low'] <= target:
                trade.exit_time = bar.name
                trade.exit_price = target
                trade.exit_reason = 'target'
                break
            elif bar['high'] >= stop:
                trade.exit_time = bar.name
                trade.exit_price = stop
                trade.exit_reason = 'stop'
                break

        else:  # long
            target = max(orb_mid, current_vwap)
            stop = orb_low - (orb_high - orb_low) * 0.5  # 50% below range

            if bar['high'] >= target:
                trade.exit_time = bar.name
                trade.exit_price = target
                trade.exit_reason = 'target'
                break
            elif bar['low'] <= stop:
                trade.exit_time = bar.name
                trade.exit_price = stop
                trade.exit_reason = 'stop'
                break

    # EOD exit if no target/stop hit
    if trade.exit_time is None:
        trade.exit_time = market_df.index[-1]
        trade.exit_price = market_df.iloc[-1]['close']
        trade.exit_reason = 'eod'

    # Calculate P&L
    if trade.direction == 'short':
        trade.pnl = trade.entry_price - trade.exit_price
    else:
        trade.pnl = trade.exit_price - trade.entry_price

    trade.pnl_pct = (trade.pnl / trade.entry_price) * 100

    return trade


def backtest_day(filepath: str) -> List[Trade]:
    """Run ORB Fade strategy for a single day."""
    try:
        df = load_day_data(filepath)
        market_df = get_market_hours(df)

        if len(market_df) < ORB_MINUTES + 10:
            return []

        orb_high, orb_low, orb_mid = calculate_orb(df, ORB_MINUTES)
        if orb_high is None:
            return []

        signals = detect_false_break(df, orb_high, orb_low)

        trades = []
        for signal in signals[:1]:  # Take only first signal per day
            trade = execute_trade(signal, df, orb_high, orb_low, orb_mid)
            trades.append(trade)

        return trades

    except Exception as e:
        print(f"Error processing {filepath}: {e}")
        return []


def run_backtest(years: List[int] = [2022, 2023, 2024]) -> pd.DataFrame:
    """Run backtest across multiple years."""
    all_trades = []

    for year in years:
        year_path = os.path.join(DATA_PATH, str(year))
        if not os.path.exists(year_path):
            print(f"Year {year} data not found")
            continue

        files = sorted([f for f in os.listdir(year_path) if f.endswith('.parquet')])
        print(f"Processing {year}: {len(files)} days...")

        for f in files:
            trades = backtest_day(os.path.join(year_path, f))
            all_trades.extend(trades)

    if not all_trades:
        return pd.DataFrame()

    # Convert to DataFrame
    results = pd.DataFrame([{
        'entry_time': t.entry_time,
        'entry_price': t.entry_price,
        'direction': t.direction,
        'exit_time': t.exit_time,
        'exit_price': t.exit_price,
        'exit_reason': t.exit_reason,
        'pnl': t.pnl,
        'pnl_pct': t.pnl_pct,
        'orb_high': t.orb_high,
        'orb_low': t.orb_low,
        'orb_mid': t.orb_mid
    } for t in all_trades])

    return results


def calculate_metrics(results: pd.DataFrame) -> dict:
    """Calculate strategy performance metrics."""
    if len(results) == 0:
        return {}

    total_trades = len(results)
    winners = results[results['pnl'] > 0]
    losers = results[results['pnl'] < 0]

    win_rate = len(winners) / total_trades * 100

    avg_win = winners['pnl_pct'].mean() if len(winners) > 0 else 0
    avg_loss = losers['pnl_pct'].mean() if len(losers) > 0 else 0

    total_pnl_pct = results['pnl_pct'].sum()
    avg_pnl_pct = results['pnl_pct'].mean()

    # Calculate Sharpe (annualized, assuming ~252 trading days)
    daily_returns = results['pnl_pct']
    sharpe = (daily_returns.mean() / daily_returns.std()) * np.sqrt(252) if daily_returns.std() > 0 else 0

    # Max drawdown
    cumulative = results['pnl_pct'].cumsum()
    running_max = cumulative.cummax()
    drawdown = cumulative - running_max
    max_drawdown = drawdown.min()

    # Profit factor
    gross_profit = winners['pnl_pct'].sum() if len(winners) > 0 else 0
    gross_loss = abs(losers['pnl_pct'].sum()) if len(losers) > 0 else 1
    profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf')

    # Exit reason breakdown
    exit_reasons = results['exit_reason'].value_counts().to_dict()

    return {
        'total_trades': total_trades,
        'win_rate': win_rate,
        'avg_win_pct': avg_win,
        'avg_loss_pct': avg_loss,
        'total_return_pct': total_pnl_pct,
        'avg_return_pct': avg_pnl_pct,
        'sharpe_ratio': sharpe,
        'max_drawdown_pct': max_drawdown,
        'profit_factor': profit_factor,
        'exit_reasons': exit_reasons
    }


def print_results(metrics: dict, results: pd.DataFrame):
    """Print formatted backtest results."""
    print("\n" + "="*60)
    print("STRATEGY #1: 0DTE ORB FADE - BACKTEST RESULTS")
    print("="*60)

    print(f"\nTotal Trades: {metrics['total_trades']}")
    print(f"Win Rate: {metrics['win_rate']:.1f}%")
    print(f"Avg Win: +{metrics['avg_win_pct']:.3f}%")
    print(f"Avg Loss: {metrics['avg_loss_pct']:.3f}%")
    print(f"\nTotal Return: {metrics['total_return_pct']:.2f}%")
    print(f"Avg Return/Trade: {metrics['avg_return_pct']:.3f}%")
    print(f"Sharpe Ratio: {metrics['sharpe_ratio']:.2f}")
    print(f"Max Drawdown: {metrics['max_drawdown_pct']:.2f}%")
    print(f"Profit Factor: {metrics['profit_factor']:.2f}")

    print(f"\nExit Reasons: {metrics['exit_reasons']}")

    # By direction
    print("\nBy Direction:")
    for direction in ['long', 'short']:
        dir_trades = results[results['direction'] == direction]
        if len(dir_trades) > 0:
            dir_win_rate = len(dir_trades[dir_trades['pnl'] > 0]) / len(dir_trades) * 100
            print(f"  {direction.upper()}: {len(dir_trades)} trades, {dir_win_rate:.1f}% win rate")

    # By year
    print("\nBy Year:")
    results['year'] = results['entry_time'].dt.year
    for year in sorted(results['year'].unique()):
        year_trades = results[results['year'] == year]
        year_pnl = year_trades['pnl_pct'].sum()
        print(f"  {year}: {len(year_trades)} trades, {year_pnl:.2f}% return")


if __name__ == '__main__':
    print("Running ORB Fade Backtest...")
    print(f"Parameters: ORB={ORB_MINUTES}min, False Break Threshold={FALSE_BREAK_THRESHOLD*100}%")

    results = run_backtest(years=[2022, 2023, 2024])

    if len(results) > 0:
        metrics = calculate_metrics(results)
        print_results(metrics, results)

        # Save results
        output_path = '/Users/nish_macbook/development/trading/strategy_backtests/results'
        os.makedirs(output_path, exist_ok=True)
        results.to_csv(f'{output_path}/orb_fade_results.csv', index=False)
        print(f"\nResults saved to {output_path}/orb_fade_results.csv")
    else:
        print("No trades generated!")
