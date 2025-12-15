"""
Strategy #3: 0DTE Gap → ORB Trend System

Edge: Large gaps predict strong directional ORB continuation
Trigger: Gap >0.35% with aligned ORB break
Entry/Exit: Trade in direction of ORB; exit at prior day H/L or VWAP loss
Best Regime: High volatility, macro news days

Data: Alpaca SPY 1-min bars (2020-2025)
"""

import pandas as pd
import numpy as np
import os
from datetime import time
from dataclasses import dataclass
from typing import Optional
import warnings
warnings.filterwarnings('ignore')

DATA_PATH = '/Users/nish_macbook/development/trading/central_data/market_data/alpaca_spy_1min'

# Parameters
ORB_MINUTES = 15
GAP_THRESHOLD = 0.0035  # 0.35% minimum gap
MARKET_OPEN = time(9, 30)
MARKET_CLOSE = time(16, 0)


@dataclass
class Trade:
    date: str
    gap_pct: float
    direction: str
    entry_time: object
    entry_price: float
    orb_high: float
    orb_low: float
    exit_time: Optional[object] = None
    exit_price: Optional[float] = None
    exit_reason: Optional[str] = None
    pnl: Optional[float] = None
    pnl_pct: Optional[float] = None


def load_day_data(filepath: str) -> pd.DataFrame:
    df = pd.read_parquet(filepath, engine='fastparquet')
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df['timestamp'] = df['timestamp'].dt.tz_convert('America/New_York')
    df = df.set_index('timestamp').sort_index()
    return df


def get_market_hours(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df['time'] = df.index.time
    mask = (df['time'] >= MARKET_OPEN) & (df['time'] < MARKET_CLOSE)
    return df[mask].drop(columns=['time'])


def calculate_gap(today_open: float, prev_close: float) -> float:
    """Calculate overnight gap percentage."""
    return (today_open - prev_close) / prev_close


def calculate_orb(df: pd.DataFrame, minutes: int = 15):
    """Calculate Opening Range (high, low) for first N minutes."""
    market_df = get_market_hours(df)
    if len(market_df) < minutes:
        return None, None
    orb_data = market_df.iloc[:minutes]
    return orb_data['high'].max(), orb_data['low'].min()


def calculate_vwap(df: pd.DataFrame) -> pd.Series:
    """Calculate cumulative VWAP from market open."""
    market_df = get_market_hours(df)
    typical_price = (market_df['high'] + market_df['low'] + market_df['close']) / 3
    cum_vol = market_df['volume'].cumsum()
    cum_tp_vol = (typical_price * market_df['volume']).cumsum()
    vwap = cum_tp_vol / cum_vol
    return vwap


def check_orb_break(df: pd.DataFrame, orb_high: float, orb_low: float, gap_direction: str):
    """
    Check if price breaks ORB in direction of gap.
    Returns break time and price if break occurs, else None.
    """
    market_df = get_market_hours(df)

    # Look for break after ORB period
    for i in range(ORB_MINUTES, min(ORB_MINUTES + 30, len(market_df))):  # Within 30 min after ORB
        bar = market_df.iloc[i]

        if gap_direction == 'up' and bar['high'] > orb_high:
            return bar.name, orb_high, 'long'
        elif gap_direction == 'down' and bar['low'] < orb_low:
            return bar.name, orb_low, 'short'

    return None, None, None


def execute_trade(entry_time, entry_price: float, direction: str,
                  df: pd.DataFrame, orb_high: float, orb_low: float,
                  prev_high: float, prev_low: float) -> Trade:
    """
    Execute trend trade:
    - Long: Exit at prior day high or VWAP loss
    - Short: Exit at prior day low or VWAP loss
    """
    market_df = get_market_hours(df)
    vwap = calculate_vwap(df)

    trade = Trade(
        date=str(entry_time.date()),
        gap_pct=0,  # Will be filled later
        direction=direction,
        entry_time=entry_time,
        entry_price=entry_price,
        orb_high=orb_high,
        orb_low=orb_low
    )

    entry_idx = market_df.index.get_loc(entry_time)

    for i in range(entry_idx + 1, len(market_df)):
        bar = market_df.iloc[i]
        current_vwap = vwap.iloc[i] if i < len(vwap) else vwap.iloc[-1]

        if direction == 'long':
            # Target: Prior day high
            # Stop: Close below VWAP
            if bar['high'] >= prev_high:
                trade.exit_time = bar.name
                trade.exit_price = prev_high
                trade.exit_reason = 'target'
                break
            elif bar['close'] < current_vwap:
                trade.exit_time = bar.name
                trade.exit_price = bar['close']
                trade.exit_reason = 'vwap_stop'
                break
        else:  # short
            if bar['low'] <= prev_low:
                trade.exit_time = bar.name
                trade.exit_price = prev_low
                trade.exit_reason = 'target'
                break
            elif bar['close'] > current_vwap:
                trade.exit_time = bar.name
                trade.exit_price = bar['close']
                trade.exit_reason = 'vwap_stop'
                break

    # EOD exit
    if trade.exit_time is None:
        trade.exit_time = market_df.index[-1]
        trade.exit_price = market_df.iloc[-1]['close']
        trade.exit_reason = 'eod'

    # Calculate P&L
    if direction == 'long':
        trade.pnl = trade.exit_price - trade.entry_price
    else:
        trade.pnl = trade.entry_price - trade.exit_price

    trade.pnl_pct = (trade.pnl / trade.entry_price) * 100

    return trade


def run_backtest(years: list = [2021, 2022, 2023, 2024], gap_threshold: float = GAP_THRESHOLD):
    """Run Gap → ORB Trend backtest."""
    all_trades = []
    prev_close = None
    prev_high = None
    prev_low = None

    for year in years:
        year_path = os.path.join(DATA_PATH, str(year))
        if not os.path.exists(year_path):
            continue

        files = sorted([f for f in os.listdir(year_path) if f.endswith('.parquet')])
        print(f"Processing {year}: {len(files)} days...")

        for f in files:
            try:
                filepath = os.path.join(year_path, f)
                df = load_day_data(filepath)
                market_df = get_market_hours(df)

                if len(market_df) < ORB_MINUTES + 10:
                    prev_close = market_df.iloc[-1]['close'] if len(market_df) > 0 else prev_close
                    prev_high = market_df['high'].max() if len(market_df) > 0 else prev_high
                    prev_low = market_df['low'].min() if len(market_df) > 0 else prev_low
                    continue

                # Get today's open
                today_open = market_df.iloc[0]['open']

                # Check for gap
                if prev_close is not None:
                    gap = calculate_gap(today_open, prev_close)
                    gap_direction = 'up' if gap > 0 else 'down'

                    # Only trade if gap exceeds threshold
                    if abs(gap) >= gap_threshold:
                        orb_high, orb_low = calculate_orb(df, ORB_MINUTES)

                        if orb_high is not None:
                            # Check for aligned ORB break
                            break_time, break_price, direction = check_orb_break(
                                df, orb_high, orb_low, gap_direction
                            )

                            if break_time is not None and prev_high is not None:
                                trade = execute_trade(
                                    break_time, break_price, direction,
                                    df, orb_high, orb_low,
                                    prev_high, prev_low
                                )
                                trade.gap_pct = gap * 100
                                all_trades.append(trade)

                # Update previous day values
                prev_close = market_df.iloc[-1]['close']
                prev_high = market_df['high'].max()
                prev_low = market_df['low'].min()

            except Exception as e:
                continue

    return all_trades


def analyze_results(trades: list, name: str = ""):
    """Analyze and print backtest results."""
    if not trades:
        print("No trades!")
        return {}

    pnl_pcts = np.array([t.pnl_pct for t in trades])
    wins = pnl_pcts[pnl_pcts > 0]
    losses = pnl_pcts[pnl_pcts < 0]

    win_rate = len(wins) / len(pnl_pcts) * 100
    sharpe = (np.mean(pnl_pcts) / np.std(pnl_pcts)) * np.sqrt(252) if np.std(pnl_pcts) > 0 else 0

    cumsum = np.cumsum(pnl_pcts)
    running_max = np.maximum.accumulate(cumsum)
    drawdown = cumsum - running_max
    max_dd = np.min(drawdown)

    gross_profit = np.sum(wins) if len(wins) > 0 else 0
    gross_loss = abs(np.sum(losses)) if len(losses) > 0 else 1
    pf = gross_profit / gross_loss if gross_loss > 0 else float('inf')

    print(f"\n{'='*60}")
    print(f"RESULTS {name}")
    print(f"{'='*60}")
    print(f"Total Trades: {len(trades)}")
    print(f"Win Rate: {win_rate:.1f}%")
    print(f"Avg Win: +{np.mean(wins):.3f}%" if len(wins) > 0 else "Avg Win: N/A")
    print(f"Avg Loss: {np.mean(losses):.3f}%" if len(losses) > 0 else "Avg Loss: N/A")
    print(f"\nTotal Return: {np.sum(pnl_pcts):.2f}%")
    print(f"Avg Return/Trade: {np.mean(pnl_pcts):.3f}%")
    print(f"Sharpe Ratio: {sharpe:.2f}")
    print(f"Max Drawdown: {max_dd:.2f}%")
    print(f"Profit Factor: {pf:.2f}")

    # By year
    print("\nBy Year:")
    for year in [2021, 2022, 2023, 2024]:
        year_trades = [t for t in trades if int(t.date[:4]) == year]
        if year_trades:
            year_pnl = sum(t.pnl_pct for t in year_trades)
            year_wr = len([t for t in year_trades if t.pnl_pct > 0]) / len(year_trades) * 100
            print(f"  {year}: {len(year_trades)} trades, {year_wr:.1f}% WR, {year_pnl:.2f}% return")

    # By direction
    print("\nBy Direction:")
    for direction in ['long', 'short']:
        dir_trades = [t for t in trades if t.direction == direction]
        if dir_trades:
            dir_pnl = sum(t.pnl_pct for t in dir_trades)
            dir_wr = len([t for t in dir_trades if t.pnl_pct > 0]) / len(dir_trades) * 100
            print(f"  {direction.upper()}: {len(dir_trades)} trades, {dir_wr:.1f}% WR, {dir_pnl:.2f}% return")

    # Exit reasons
    print("\nExit Reasons:")
    reasons = {}
    for t in trades:
        reasons[t.exit_reason] = reasons.get(t.exit_reason, 0) + 1
    for r, c in sorted(reasons.items(), key=lambda x: -x[1]):
        print(f"  {r}: {c} ({c/len(trades)*100:.1f}%)")

    return {
        'trades': len(trades),
        'win_rate': win_rate,
        'sharpe': sharpe,
        'total_return': np.sum(pnl_pcts),
        'max_dd': max_dd,
        'pf': pf
    }


if __name__ == '__main__':
    print("="*60)
    print("STRATEGY #3: GAP → ORB TREND SYSTEM")
    print("="*60)
    print(f"Parameters: ORB={ORB_MINUTES}min, Gap Threshold={GAP_THRESHOLD*100}%")

    trades = run_backtest()
    analyze_results(trades)

    # Try different gap thresholds
    print("\n" + "="*60)
    print("PARAMETER SWEEP - Gap Threshold")
    print("="*60)

    for gap in [0.002, 0.003, 0.0035, 0.004, 0.005, 0.0075, 0.01]:
        trades = run_backtest(gap_threshold=gap)
        if trades:
            pnl = np.array([t.pnl_pct for t in trades])
            sharpe = (np.mean(pnl) / np.std(pnl)) * np.sqrt(252) if np.std(pnl) > 0 else 0
            wr = len(pnl[pnl > 0]) / len(pnl) * 100
            print(f"Gap {gap*100:.2f}%: {len(trades)} trades, {wr:.1f}% WR, {np.sum(pnl):.2f}% return, Sharpe {sharpe:.2f}")
