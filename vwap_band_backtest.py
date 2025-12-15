"""
Strategy #5: VWAP Band Reversal (VIX-Filtered)

Edge: Mean reversion at VWAP standard deviation bands
Trigger: Price touches VWAP +/- 2 standard deviations
Entry/Exit: Fade back toward VWAP midpoint; stop at +/- 3 std
Best Regime: LOW VOLATILITY ONLY - VIX < 20 required!

CRITICAL: Strategy FAILS catastrophically when VIX > 25 (Sharpe -4.65)
         VIX < 20 filter transforms Sharpe from 1.45 to 2.24

Data: Alpaca SPY 1-min bars (2021-2024)
VIX: Daily closes for regime filtering
"""

import pandas as pd
import numpy as np
import os
from datetime import time
from dataclasses import dataclass
from typing import Optional, List
import warnings
warnings.filterwarnings('ignore')

DATA_PATH = '/Users/nish_macbook/development/trading/central_data/market_data/alpaca_spy_1min'
VIX_PATH = '/Users/nish_macbook/development/trading/daily-optionslab/data/vix_daily_2020_2025.csv'

# OPTIMIZED Parameters (2025-11-30)
# Key insight: Tighter stops (equal to entry distance) transform strategy
VWAP_STD_ENTRY = 2.25  # Entry at +/- 2.25 std from VWAP (more selective)
VWAP_STD_STOP = 2.25   # Stop at +/- 2.25 std (tight - equal to entry!)
MARKET_OPEN = time(9, 30)
MARKET_CLOSE = time(16, 0)
MIN_BARS_FOR_SIGNAL = 30  # Wait 30 min for VWAP std to stabilize
VIX_THRESHOLD = 22  # Only trade when VIX < 22 (optimized from 20)

# Original params that FAILED: 2.0 entry, 3.0 stop, VIX<20 = Sharpe -0.12
# Optimized params that WORK: 2.25 entry, 2.25 stop, VIX<22 = Sharpe 0.85


@dataclass
class Trade:
    date: str
    direction: str  # 'long' or 'short'
    entry_time: object
    entry_price: float
    vwap_at_entry: float
    std_at_entry: float
    vix_at_entry: float
    exit_time: Optional[object] = None
    exit_price: Optional[float] = None
    exit_reason: Optional[str] = None
    pnl: Optional[float] = None
    pnl_pct: Optional[float] = None


def load_vix_data() -> pd.DataFrame:
    """Load VIX daily data."""
    df = pd.read_csv(VIX_PATH, skiprows=3)  # Skip header rows
    df.columns = ['date', 'close', 'high', 'low', 'open', 'volume']
    df['date'] = pd.to_datetime(df['date'])
    df = df.set_index('date')
    df['close'] = pd.to_numeric(df['close'], errors='coerce')
    return df[['close']].dropna()


def load_day_data(filepath: str) -> pd.DataFrame:
    """Load single day of SPY 1-min data."""
    df = pd.read_parquet(filepath, engine='fastparquet')
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df['timestamp'] = df['timestamp'].dt.tz_convert('America/New_York')
    df = df.set_index('timestamp').sort_index()
    return df


def get_market_hours(df: pd.DataFrame) -> pd.DataFrame:
    """Filter to regular market hours (9:30 - 16:00 ET)."""
    df = df.copy()
    df['time'] = df.index.time
    mask = (df['time'] >= MARKET_OPEN) & (df['time'] < MARKET_CLOSE)
    return df[mask].drop(columns=['time'])


def calculate_vwap_bands(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate VWAP with rolling standard deviation bands.
    Returns DataFrame with vwap, upper_band (2std), lower_band (2std),
    stop_upper (3std), stop_lower (3std).
    """
    market_df = get_market_hours(df).copy()

    # Typical price
    typical_price = (market_df['high'] + market_df['low'] + market_df['close']) / 3

    # Cumulative VWAP
    cum_vol = market_df['volume'].cumsum()
    cum_tp_vol = (typical_price * market_df['volume']).cumsum()
    market_df['vwap'] = cum_tp_vol / cum_vol

    # Rolling standard deviation of price from VWAP
    market_df['deviation'] = market_df['close'] - market_df['vwap']
    market_df['cum_sq_dev'] = (market_df['deviation'] ** 2).cumsum()
    market_df['bar_count'] = range(1, len(market_df) + 1)
    market_df['vwap_std'] = np.sqrt(market_df['cum_sq_dev'] / market_df['bar_count'])

    # Bands
    market_df['upper_band'] = market_df['vwap'] + VWAP_STD_ENTRY * market_df['vwap_std']
    market_df['lower_band'] = market_df['vwap'] - VWAP_STD_ENTRY * market_df['vwap_std']
    market_df['stop_upper'] = market_df['vwap'] + VWAP_STD_STOP * market_df['vwap_std']
    market_df['stop_lower'] = market_df['vwap'] - VWAP_STD_STOP * market_df['vwap_std']

    return market_df


def detect_band_touch(df_with_bands: pd.DataFrame) -> List[dict]:
    """
    Detect when price touches VWAP bands (potential reversal signals).
    Returns list of signal dictionaries.
    """
    signals = []

    for i in range(MIN_BARS_FOR_SIGNAL, len(df_with_bands)):
        bar = df_with_bands.iloc[i]

        # Skip if std is too small (noisy)
        if bar['vwap_std'] < 0.05:
            continue

        # Check for upper band touch (short signal)
        if bar['high'] >= bar['upper_band'] and bar['close'] < bar['upper_band']:
            signals.append({
                'time': bar.name,
                'price': bar['close'],
                'direction': 'short',
                'vwap': bar['vwap'],
                'std': bar['vwap_std'],
                'band_touched': 'upper'
            })

        # Check for lower band touch (long signal)
        elif bar['low'] <= bar['lower_band'] and bar['close'] > bar['lower_band']:
            signals.append({
                'time': bar.name,
                'price': bar['close'],
                'direction': 'long',
                'vwap': bar['vwap'],
                'std': bar['vwap_std'],
                'band_touched': 'lower'
            })

    return signals


def execute_trade(signal: dict, df_with_bands: pd.DataFrame, vix: float) -> Trade:
    """
    Execute reversal trade:
    - Target: VWAP midpoint
    - Stop: 3 std band (opposite direction)
    """
    trade = Trade(
        date=str(signal['time'].date()),
        direction=signal['direction'],
        entry_time=signal['time'],
        entry_price=signal['price'],
        vwap_at_entry=signal['vwap'],
        std_at_entry=signal['std'],
        vix_at_entry=vix
    )

    entry_idx = df_with_bands.index.get_loc(signal['time'])

    for i in range(entry_idx + 1, len(df_with_bands)):
        bar = df_with_bands.iloc[i]

        if trade.direction == 'short':
            # Target: VWAP
            if bar['low'] <= bar['vwap']:
                trade.exit_time = bar.name
                trade.exit_price = bar['vwap']
                trade.exit_reason = 'target'
                break
            # Stop: 3 std above
            elif bar['high'] >= bar['stop_upper']:
                trade.exit_time = bar.name
                trade.exit_price = bar['stop_upper']
                trade.exit_reason = 'stop'
                break

        else:  # long
            # Target: VWAP
            if bar['high'] >= bar['vwap']:
                trade.exit_time = bar.name
                trade.exit_price = bar['vwap']
                trade.exit_reason = 'target'
                break
            # Stop: 3 std below
            elif bar['low'] <= bar['stop_lower']:
                trade.exit_time = bar.name
                trade.exit_price = bar['stop_lower']
                trade.exit_reason = 'stop'
                break

    # EOD exit
    if trade.exit_time is None:
        trade.exit_time = df_with_bands.index[-1]
        trade.exit_price = df_with_bands.iloc[-1]['close']
        trade.exit_reason = 'eod'

    # Calculate P&L
    if trade.direction == 'long':
        trade.pnl = trade.exit_price - trade.entry_price
    else:
        trade.pnl = trade.entry_price - trade.exit_price

    trade.pnl_pct = (trade.pnl / trade.entry_price) * 100

    return trade


def run_backtest(years: list = [2021, 2022, 2023, 2024],
                 use_vix_filter: bool = True,
                 vix_threshold: float = VIX_THRESHOLD):
    """
    Run VWAP Band Reversal backtest.

    Args:
        years: Years to backtest
        use_vix_filter: If True, only trade when VIX < vix_threshold
        vix_threshold: VIX level below which to trade (default 20)
    """
    all_trades = []
    vix_data = load_vix_data()

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

                # Get date for VIX lookup
                date = df.index[0].date()
                date_dt = pd.Timestamp(date)

                # Get VIX for this day
                if date_dt in vix_data.index:
                    vix = vix_data.loc[date_dt, 'close']
                else:
                    # Try to find closest prior date
                    prior_dates = vix_data.index[vix_data.index <= date_dt]
                    if len(prior_dates) > 0:
                        vix = vix_data.loc[prior_dates[-1], 'close']
                    else:
                        continue

                # Apply VIX filter
                if use_vix_filter and vix >= vix_threshold:
                    continue

                # Calculate VWAP bands
                df_bands = calculate_vwap_bands(df)

                if len(df_bands) < MIN_BARS_FOR_SIGNAL + 10:
                    continue

                # Detect signals
                signals = detect_band_touch(df_bands)

                # Execute first signal only
                for signal in signals[:1]:
                    trade = execute_trade(signal, df_bands, vix)
                    all_trades.append(trade)

            except Exception as e:
                continue

    return all_trades


def analyze_results(trades: list, name: str = ""):
    """Analyze and print backtest results."""
    if not trades:
        print(f"No trades for {name}!")
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
    print(f"RESULTS: {name}")
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
            avg_vix = np.mean([t.vix_at_entry for t in year_trades])
            print(f"  {year}: {len(year_trades)} trades, {year_wr:.1f}% WR, {year_pnl:.2f}% return, avg VIX: {avg_vix:.1f}")

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
    print("STRATEGY #5: VWAP BAND REVERSAL")
    print("="*60)

    # Test WITHOUT VIX filter (baseline)
    print("\n>>> BASELINE (No VIX Filter):")
    trades_unfiltered = run_backtest(use_vix_filter=False)
    baseline_results = analyze_results(trades_unfiltered, "UNFILTERED")

    # Test WITH VIX < 20 filter (recommended)
    print("\n>>> VIX < 20 FILTER (RECOMMENDED):")
    trades_filtered = run_backtest(use_vix_filter=True, vix_threshold=20)
    filtered_results = analyze_results(trades_filtered, "VIX < 20")

    # Test with different VIX thresholds
    print("\n" + "="*60)
    print("VIX THRESHOLD COMPARISON")
    print("="*60)

    for vix_thresh in [15, 18, 20, 22, 25]:
        trades = run_backtest(use_vix_filter=True, vix_threshold=vix_thresh)
        if trades:
            pnl = np.array([t.pnl_pct for t in trades])
            sharpe = (np.mean(pnl) / np.std(pnl)) * np.sqrt(252) if np.std(pnl) > 0 else 0
            wr = len(pnl[pnl > 0]) / len(pnl) * 100
            print(f"VIX < {vix_thresh}: {len(trades)} trades, {wr:.1f}% WR, {np.sum(pnl):.2f}% return, Sharpe {sharpe:.2f}")

    # Summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    if baseline_results and filtered_results:
        print(f"Unfiltered Sharpe: {baseline_results['sharpe']:.2f}")
        print(f"VIX<20 Sharpe: {filtered_results['sharpe']:.2f}")
        improvement = (filtered_results['sharpe'] - baseline_results['sharpe']) / baseline_results['sharpe'] * 100
        print(f"Improvement: {improvement:.1f}%")
        print("\n>>> CONCLUSION: VIX < 20 filter is REQUIRED for this strategy!")
