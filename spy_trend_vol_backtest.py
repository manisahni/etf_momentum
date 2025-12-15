"""
Strategy: SPY Trend Following + Volatility Filter

Inspired by successful Gaussian Filter strategy (204% return, Sharpe 1.68)
Key patterns from winners:
1. Trend-following (not mean reversion)
2. Volatility/volume filters
3. Simple indicators

Hypothesis: Simple EMA crossover with volatility regime filter can beat SPY buy & hold

Data: yfinance SPY daily (2010-2024)
"""

import pandas as pd
import numpy as np
import yfinance as yf
from dataclasses import dataclass
from typing import List, Optional
import warnings
warnings.filterwarnings('ignore')


@dataclass
class Trade:
    entry_date: object
    entry_price: float
    direction: str
    exit_date: Optional[object] = None
    exit_price: Optional[float] = None
    exit_reason: Optional[str] = None
    pnl_pct: Optional[float] = None


def download_data(start='2010-01-01', end='2024-12-01'):
    """Download SPY data with volume."""
    print("Downloading SPY data...")
    df = yf.download('SPY', start=start, end=end, auto_adjust=True, progress=False)

    # Handle MultiIndex columns from yfinance
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    df = df.dropna()
    print(f"Data: {df.index[0].date()} to {df.index[-1].date()}, {len(df)} days")
    return df


def calculate_indicators(df: pd.DataFrame, fast_ema: int = 10, slow_ema: int = 30,
                         vol_lookback: int = 20) -> pd.DataFrame:
    """Calculate EMA crossover and volatility indicators."""
    df = df.copy()

    # EMA crossover
    df['ema_fast'] = df['Close'].ewm(span=fast_ema, adjust=False).mean()
    df['ema_slow'] = df['Close'].ewm(span=slow_ema, adjust=False).mean()

    # Trend direction
    df['trend'] = np.where(df['ema_fast'] > df['ema_slow'], 1, -1)
    df['trend_change'] = df['trend'].diff()

    # Volatility (rolling std of returns)
    df['returns'] = df['Close'].pct_change()
    df['volatility'] = df['returns'].rolling(vol_lookback).std() * np.sqrt(252)

    # Volume filter
    df['volume_ma'] = df['Volume'].rolling(20).mean()
    df['volume_above_avg'] = df['Volume'] > df['volume_ma']

    return df


def generate_signals(df: pd.DataFrame, use_vol_filter: bool = True,
                     vol_threshold: float = 0.25) -> pd.DataFrame:
    """
    Generate trading signals.

    Long: EMA fast crosses above slow
    Exit: EMA fast crosses below slow

    Vol filter: Only trade when volatility > threshold (trending markets)
    """
    df = df.copy()

    # Base signals from EMA crossover
    df['long_signal'] = (df['trend_change'] == 2)  # Crossed above
    df['exit_signal'] = (df['trend_change'] == -2)  # Crossed below

    if use_vol_filter:
        # Only enter when volatility is moderate (not too high, not too low)
        vol_ok = (df['volatility'] > vol_threshold) & (df['volatility'] < 0.50)
        df['long_signal'] = df['long_signal'] & vol_ok

    return df


def backtest_long_only(df: pd.DataFrame) -> List[Trade]:
    """
    Long-only trend following backtest.
    Entry: Long signal
    Exit: Exit signal or stop
    """
    trades = []
    position = None

    for i in range(1, len(df)):
        bar = df.iloc[i]
        prev_bar = df.iloc[i-1]

        # Check for exit if in position
        if position is not None:
            current_trade = trades[-1]

            # Exit on signal
            if bar['exit_signal']:
                current_trade.exit_date = bar.name
                current_trade.exit_price = bar['Close']
                current_trade.exit_reason = 'signal'
                current_trade.pnl_pct = (current_trade.exit_price / current_trade.entry_price - 1) * 100
                position = None
                continue

            # Trailing stop: exit if drops 5% from entry
            if bar['Close'] < current_trade.entry_price * 0.95:
                current_trade.exit_date = bar.name
                current_trade.exit_price = bar['Close']
                current_trade.exit_reason = 'stop'
                current_trade.pnl_pct = (current_trade.exit_price / current_trade.entry_price - 1) * 100
                position = None
                continue

        # Check for new entry if flat
        if position is None and bar['long_signal']:
            trades.append(Trade(
                entry_date=bar.name,
                entry_price=bar['Close'],
                direction='long'
            ))
            position = 'long'

    # Close open position at end
    if position is not None and trades:
        current_trade = trades[-1]
        if current_trade.exit_date is None:
            last_bar = df.iloc[-1]
            current_trade.exit_date = last_bar.name
            current_trade.exit_price = last_bar['Close']
            current_trade.exit_reason = 'eod'
            current_trade.pnl_pct = (current_trade.exit_price / current_trade.entry_price - 1) * 100

    return trades


def calculate_metrics(trades: List[Trade], df: pd.DataFrame) -> dict:
    """Calculate performance metrics."""
    if not trades:
        return {}

    completed = [t for t in trades if t.pnl_pct is not None]
    if not completed:
        return {}

    pnl_pcts = np.array([t.pnl_pct for t in completed])
    wins = pnl_pcts[pnl_pcts > 0]
    losses = pnl_pcts[pnl_pcts < 0]

    # Total return (compounded)
    total_mult = 1.0
    for pnl in pnl_pcts:
        total_mult *= (1 + pnl/100)
    total_return = (total_mult - 1) * 100

    # Years
    first_date = completed[0].entry_date
    last_date = completed[-1].exit_date
    years = (last_date - first_date).days / 365

    annual_return = (total_mult ** (1/years) - 1) * 100 if years > 0 else 0

    # Sharpe (using trade returns)
    sharpe = (np.mean(pnl_pcts) / np.std(pnl_pcts)) * np.sqrt(len(completed)/years) if np.std(pnl_pcts) > 0 else 0

    # Max drawdown
    cumsum = np.cumsum(pnl_pcts)
    running_max = np.maximum.accumulate(cumsum)
    drawdown = cumsum - running_max
    max_dd = np.min(drawdown)

    # Win rate
    win_rate = len(wins) / len(pnl_pcts) * 100

    # SPY benchmark
    spy_start = df.loc[first_date:].iloc[0]['Close']
    spy_end = df.loc[:last_date].iloc[-1]['Close']
    spy_return = (spy_end / spy_start - 1) * 100
    spy_annual = ((spy_end / spy_start) ** (1/years) - 1) * 100 if years > 0 else 0

    # Time in market
    days_in_market = sum((t.exit_date - t.entry_date).days for t in completed)
    total_days = (last_date - first_date).days
    pct_in_market = days_in_market / total_days * 100 if total_days > 0 else 0

    return {
        'trades': len(completed),
        'win_rate': win_rate,
        'total_return': total_return,
        'annual_return': annual_return,
        'sharpe': sharpe,
        'max_dd': max_dd,
        'avg_win': np.mean(wins) if len(wins) > 0 else 0,
        'avg_loss': np.mean(losses) if len(losses) > 0 else 0,
        'spy_return': spy_return,
        'spy_annual': spy_annual,
        'years': years,
        'pct_in_market': pct_in_market
    }


def print_results(metrics: dict, name: str = ""):
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
    print(f"\nTime in Market: {metrics['pct_in_market']:.1f}%")
    print(f"\n--- BENCHMARK ---")
    print(f"SPY Buy & Hold: {metrics['spy_return']:.1f}% ({metrics['spy_annual']:.1f}% annual)")
    print(f"Alpha (annual): {metrics['annual_return'] - metrics['spy_annual']:.1f}%")


def parameter_sweep(df: pd.DataFrame) -> List[dict]:
    """Test different parameter combinations."""
    print("\n" + "="*60)
    print("PARAMETER SWEEP")
    print("="*60)

    results = []

    for fast_ema in [5, 10, 15, 20]:
        for slow_ema in [20, 30, 50, 100, 200]:
            if fast_ema >= slow_ema:
                continue

            for vol_thresh in [0.0, 0.15, 0.20, 0.25, 0.30]:
                use_vol = vol_thresh > 0

                # Calculate indicators
                signal_df = calculate_indicators(df, fast_ema=fast_ema, slow_ema=slow_ema)
                signal_df = generate_signals(signal_df, use_vol_filter=use_vol, vol_threshold=vol_thresh)

                # Backtest
                trades = backtest_long_only(signal_df)
                metrics = calculate_metrics(trades, df)

                if not metrics or metrics['trades'] < 10:
                    continue

                results.append({
                    'fast': fast_ema,
                    'slow': slow_ema,
                    'vol_thresh': vol_thresh,
                    **metrics
                })

    # Sort by annual return
    results.sort(key=lambda x: -x['annual_return'])

    print(f"\n{'Fast':>5} {'Slow':>5} {'Vol':>5} {'Trades':>7} {'WR':>6} {'Annual':>8} {'Alpha':>8} {'Sharpe':>7}")
    print("-"*60)

    for r in results[:15]:
        alpha = r['annual_return'] - r['spy_annual']
        v = f"{r['vol_thresh']:.2f}" if r['vol_thresh'] > 0 else "None"
        print(f"{r['fast']:5d} {r['slow']:5d} {v:>5} {r['trades']:7d} {r['win_rate']:5.1f}% "
              f"{r['annual_return']:7.1f}% {alpha:+7.1f}% {r['sharpe']:6.2f}")

    return results


if __name__ == '__main__':
    print("="*60)
    print("SPY TREND FOLLOWING + VOLATILITY FILTER")
    print("="*60)

    # Download data
    df = download_data()

    # Default configuration
    print("\nTesting default: EMA 10/30, Vol > 0.25")
    signal_df = calculate_indicators(df, fast_ema=10, slow_ema=30)
    signal_df = generate_signals(signal_df, use_vol_filter=True, vol_threshold=0.25)

    trades = backtest_long_only(signal_df)
    metrics = calculate_metrics(trades, df)
    print_results(metrics, "EMA 10/30 + Vol Filter")

    # Test without vol filter
    print("\n" + "-"*60)
    print("Testing without vol filter...")
    signal_df_no_vol = generate_signals(signal_df, use_vol_filter=False)
    trades_no_vol = backtest_long_only(signal_df_no_vol)
    metrics_no_vol = calculate_metrics(trades_no_vol, df)
    print_results(metrics_no_vol, "EMA 10/30 (No Vol Filter)")

    # Parameter sweep
    sweep_results = parameter_sweep(df)

    if sweep_results:
        best = sweep_results[0]
        print(f"\nBest Configuration:")
        print(f"  Fast EMA: {best['fast']}, Slow EMA: {best['slow']}")
        print(f"  Vol Threshold: {best['vol_thresh']}")
        print(f"  Annual Return: {best['annual_return']:.1f}%")
        print(f"  Alpha vs SPY: {best['annual_return'] - best['spy_annual']:.1f}%")
        print(f"  Sharpe: {best['sharpe']:.2f}")
