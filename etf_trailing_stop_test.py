"""
ETF Trailing Stop Test - Compare fixed vs trailing stops on MEGA_UNIVERSE
"""

import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime
from expanded_universe_test import UNIVERSES, backtest_rotation, download_universe
import warnings
warnings.filterwarnings('ignore')


def test_etf_stops():
    """Compare fixed vs trailing stops on large ETF universe."""

    # Use MEGA_UNIVERSE (58 ETFs)
    universe = UNIVERSES['MEGA_UNIVERSE']

    print("=" * 70)
    print("ETF TRAILING STOP TEST - MEGA_UNIVERSE (58 ETFs)")
    print("=" * 70)

    # Download data
    prices, available = download_universe(universe)
    print(f"\nAvailable ETFs: {len(available)}")

    # Test configurations
    configs = [
        {'name': '3% Fixed Stop', 'stop_loss': 0.03, 'trailing': False},
        {'name': '3% Trailing Stop', 'stop_loss': 0.03, 'trailing': True},
        {'name': '5% Fixed Stop', 'stop_loss': 0.05, 'trailing': False},
        {'name': '5% Trailing Stop', 'stop_loss': 0.05, 'trailing': True},
        {'name': '10% Fixed Stop', 'stop_loss': 0.10, 'trailing': False},
        {'name': '10% Trailing Stop', 'stop_loss': 0.10, 'trailing': True},
        {'name': 'No Stop', 'stop_loss': 0.50, 'trailing': False},
    ]

    results = []

    for cfg in configs:
        print(f"\nTesting: {cfg['name']}...")

        equity, trades, yearly = backtest_rotation(
            prices, available,
            lookback=21,
            top_n=5,
            rebalance='M',
            stop_loss=cfg['stop_loss'],
            trailing_stop=cfg['trailing']
        )

        if equity is None or len(equity) == 0:
            print(f"  No valid trades")
            continue

        # Calculate metrics
        total_return = (equity['value'].iloc[-1] / 100 - 1) * 100
        returns = equity['value'].pct_change().dropna()
        sharpe = returns.mean() / returns.std() * np.sqrt(252) if returns.std() > 0 else 0
        max_dd = ((equity['value'] - equity['value'].cummax()) / equity['value'].cummax()).min() * 100

        stopped = [t for t in trades if t.exit_reason == 'stop_loss']
        winners = [t for t in trades if t.pnl_pct and t.pnl_pct > 0]

        # Per-trade stats
        avg_win = np.mean([t.pnl_pct for t in winners]) if winners else 0
        avg_loss = np.mean([t.pnl_pct for t in trades if t.pnl_pct and t.pnl_pct < 0])

        results.append({
            'config': cfg['name'],
            'return': total_return,
            'sharpe': sharpe,
            'max_dd': max_dd,
            'trades': len(trades),
            'stopped': len(stopped),
            'stop_pct': len(stopped) / len(trades) * 100 if trades else 0,
            'win_rate': len(winners) / len(trades) * 100 if trades else 0,
            'avg_win': avg_win,
            'avg_loss': avg_loss
        })

        print(f"  Return: {total_return:+.1f}% | Sharpe: {sharpe:.2f} | MaxDD: {max_dd:.1f}%")

    # Results table
    print("\n" + "=" * 90)
    print("RESULTS COMPARISON")
    print("=" * 90)
    print(f"{'Config':<20} {'Return':>10} {'Sharpe':>8} {'MaxDD':>10} {'Trades':>8} {'Stop%':>8} {'Win%':>8}")
    print("-" * 90)

    for r in results:
        print(f"{r['config']:<20} {r['return']:>+9.1f}% {r['sharpe']:>8.2f} {r['max_dd']:>+9.1f}% "
              f"{r['trades']:>8} {r['stop_pct']:>7.0f}% {r['win_rate']:>7.1f}%")

    # Per-trade analysis
    print("\n" + "=" * 90)
    print("PER-TRADE ANALYSIS")
    print("=" * 90)
    print(f"{'Config':<20} {'Avg Win':>12} {'Avg Loss':>12} {'Expectancy':>12}")
    print("-" * 90)

    for r in results:
        win_rate = r['win_rate'] / 100
        expectancy = win_rate * r['avg_win'] + (1 - win_rate) * r['avg_loss']
        print(f"{r['config']:<20} {r['avg_win']:>+11.2f}% {r['avg_loss']:>+11.2f}% {expectancy:>+11.2f}%")

    # Save results
    df = pd.DataFrame(results)
    df.to_csv('/Users/nish_macbook/development/trading/strategy_backtests/results/etf_trailing_stop_comparison.csv', index=False)
    print(f"\nResults saved to results/etf_trailing_stop_comparison.csv")

    return results


if __name__ == '__main__':
    results = test_etf_stops()
