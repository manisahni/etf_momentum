"""
ETF Ultra Universe Test - Compare MEGA (58) vs ULTRA (102) ETFs with 3% Trailing Stop
"""

import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime
from expanded_universe_test import UNIVERSES, backtest_rotation
import warnings
warnings.filterwarnings('ignore')


def main():
    print("=" * 80)
    print("ULTRA UNIVERSE TEST: 102 ETFs vs MEGA 58 ETFs")
    print("=" * 80)

    # Get universes
    mega = UNIVERSES['MEGA_UNIVERSE']
    ultra = UNIVERSES['ULTRA_UNIVERSE']

    print(f"\nMEGA_UNIVERSE: {len(mega)} ETFs")
    print(f"ULTRA_UNIVERSE: {len(ultra)} ETFs")

    # New ETFs added
    new_etfs = [t for t in ultra if t not in mega]
    print(f"\nNew ETFs added ({len(new_etfs)}): {new_etfs}")

    # Download all data
    print("\nDownloading price data...")
    all_tickers = list(set(ultra + ['SPY']))
    df = yf.download(all_tickers, start='2020-01-01',
                     end=datetime.now().strftime('%Y-%m-%d'),
                     auto_adjust=True, progress=False)

    if isinstance(df.columns, pd.MultiIndex):
        prices = df['Close']
    else:
        prices = df

    prices = prices.dropna(how='all').ffill()

    # Check availability
    available_mega = [t for t in mega if t in prices.columns]
    available_ultra = [t for t in ultra if t in prices.columns]
    print(f"\nAvailable: MEGA {len(available_mega)}/{len(mega)}, ULTRA {len(available_ultra)}/{len(ultra)}")

    # Missing ETFs
    missing = [t for t in ultra if t not in prices.columns]
    if missing:
        print(f"Missing: {missing}")

    # SPY benchmark
    spy_return = (prices['SPY'].iloc[-1] / prices['SPY'].iloc[0] - 1) * 100

    # Test configurations
    configs = [
        {'name': 'MEGA (58 ETFs)', 'universe': available_mega},
        {'name': 'ULTRA (102 ETFs)', 'universe': available_ultra},
    ]

    results = []

    for cfg in configs:
        print(f"\n{'=' * 60}")
        print(f"Testing: {cfg['name']}")
        print(f"{'=' * 60}")

        equity, trades, stats = backtest_rotation(
            prices, cfg['universe'],
            lookback=21,
            top_n=5,
            rebalance='M',
            stop_loss=0.03,
            trailing_stop=True
        )

        if equity is None or len(equity) == 0:
            print(f"  No valid trades for {cfg['name']}")
            continue

        # Calculate metrics
        total_return = (equity['value'].iloc[-1] / equity['value'].iloc[0] - 1) * 100
        daily_returns = equity['value'].pct_change().dropna()
        sharpe = daily_returns.mean() / daily_returns.std() * np.sqrt(252) if daily_returns.std() > 0 else 0
        max_dd = ((equity['value'] - equity['value'].cummax()) / equity['value'].cummax()).min() * 100

        # Trade stats
        stopped = [t for t in trades if t.exit_reason == 'stop_loss']
        winners = [t for t in trades if t.pnl_pct and t.pnl_pct > 0]

        results.append({
            'name': cfg['name'],
            'etfs': len(cfg['universe']),
            'return': total_return,
            'sharpe': sharpe,
            'max_dd': max_dd,
            'trades': len(trades),
            'stopped': len(stopped),
            'winners': len(winners),
            'alpha': total_return - spy_return
        })

        print(f"  Return: {total_return:+.1f}% | Sharpe: {sharpe:.2f} | MaxDD: {max_dd:.1f}%")
        print(f"  Trades: {len(trades)} | Stopped: {len(stopped)} | Winners: {len(winners)}")

        # Year-by-year
        print("\n  Year-by-Year:")
        equity['year'] = equity.index.year
        for year in sorted(equity['year'].unique()):
            year_data = equity[equity['year'] == year]
            if len(year_data) > 1:
                year_ret = (year_data['value'].iloc[-1] / year_data['value'].iloc[0] - 1) * 100
                year_spy = prices['SPY'][prices.index.year == year]
                spy_yr = (year_spy.iloc[-1] / year_spy.iloc[0] - 1) * 100 if len(year_spy) > 1 else 0
                print(f"    {year}: {year_ret:+.1f}% (SPY: {spy_yr:+.1f}%, Alpha: {year_ret - spy_yr:+.1f}%)")

    # Summary
    print("\n" + "=" * 80)
    print("SUMMARY: MEGA vs ULTRA UNIVERSE")
    print("=" * 80)
    print(f"\nSPY Total Return: {spy_return:+.1f}%\n")
    print(f"{'Universe':<20} {'ETFs':>6} {'Return':>10} {'Sharpe':>8} {'MaxDD':>8} {'Alpha':>10}")
    print("-" * 70)

    for r in results:
        print(f"{r['name']:<20} {r['etfs']:>6} {r['return']:>+9.1f}% {r['sharpe']:>8.2f} {r['max_dd']:>+7.1f}% {r['alpha']:>+9.1f}%")

    # New categories breakdown
    print("\n" + "=" * 80)
    print("TESTING NEW CATEGORIES INDIVIDUALLY")
    print("=" * 80)

    new_categories = ['CHINA', 'FACTOR', 'COMMODITIES', 'INDUSTRIES', 'INCOME', 'SIZE_ETFS']

    for cat in new_categories:
        cat_universe = [t for t in UNIVERSES[cat] if t in prices.columns]
        if len(cat_universe) < 3:
            print(f"\n{cat}: Only {len(cat_universe)} available, skipping")
            continue

        equity, trades, _ = backtest_rotation(
            prices, cat_universe,
            lookback=21,
            top_n=min(3, len(cat_universe)),
            rebalance='M',
            stop_loss=0.03,
            trailing_stop=True
        )

        if equity is None or len(equity) == 0:
            print(f"\n{cat}: No valid trades")
            continue

        total_return = (equity['value'].iloc[-1] / equity['value'].iloc[0] - 1) * 100
        daily_returns = equity['value'].pct_change().dropna()
        sharpe = daily_returns.mean() / daily_returns.std() * np.sqrt(252) if daily_returns.std() > 0 else 0
        max_dd = ((equity['value'] - equity['value'].cummax()) / equity['value'].cummax()).min() * 100
        stopped = len([t for t in trades if t.exit_reason == 'stop_loss'])

        print(f"\n{cat} ({len(cat_universe)} ETFs):")
        print(f"  Return: {total_return:+.1f}% | Sharpe: {sharpe:.2f} | MaxDD: {max_dd:.1f}%")
        print(f"  Trades: {len(trades)} | Stopped: {stopped}")

    return results


if __name__ == '__main__':
    results = main()
