"""
ETF 2025 Out-of-Sample Test - MEGA_UNIVERSE with 3% Trailing Stop
"""

import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime
from dataclasses import dataclass
from typing import List, Dict, Optional, Tuple
import warnings
warnings.filterwarnings('ignore')

# MEGA_UNIVERSE (58 ETFs)
MEGA_UNIVERSE = [
    # US Sectors (11)
    'XLK', 'XLV', 'XLF', 'XLY', 'XLP', 'XLE', 'XLI', 'XLB', 'XLU', 'XLRE', 'XLC',
    # High-Flying Growth/Tech (20)
    'QQQ', 'ARKK', 'ARKG', 'ARKW', 'ARKF', 'SOXX', 'SMH', 'XSD', 'IGV', 'WCLD',
    'CLOU', 'XBI', 'IBB', 'TAN', 'ICLN', 'QCLN', 'BOTZ', 'ROBO', 'FINX', 'IPAY',
    # International Developed (15)
    'EFA', 'VEA', 'EWJ', 'EWG', 'EWU', 'EWC', 'EWA', 'EWS', 'EWH', 'EWI',
    'EWP', 'EWQ', 'EWL', 'EWN', 'EWD',
    # Emerging Markets (12)
    'VWO', 'EEM', 'INDA', 'EWT', 'EWY', 'EWZ', 'EWW', 'EIDO', 'THD', 'VNM',
    'EPOL', 'TUR',
]


@dataclass
class Trade:
    entry_date: datetime
    exit_date: Optional[datetime]
    ticker: str
    entry_price: float
    exit_price: Optional[float]
    high_price: float
    stop_price: float
    exit_reason: str
    pnl_pct: Optional[float]


def backtest_2025(
    prices: pd.DataFrame,
    tickers: List[str],
    lookback: int = 21,
    top_n: int = 5,
    stop_loss: float = 0.03,
    trailing_stop: bool = True
) -> Tuple[pd.DataFrame, List[Trade], Dict]:
    """Backtest on 2025 data only."""

    available = [t for t in tickers if t in prices.columns]
    price_df = prices[available].copy()

    # Filter to 2025 only (but need lookback from late 2024)
    start_2025 = pd.Timestamp('2025-01-01')

    momentum = price_df.pct_change(lookback)

    # Monthly rebalance dates in 2025
    rebal_dates = price_df.resample('M').last().index.tolist()
    rebal_dates = [d for d in rebal_dates if d in price_df.index and d >= start_2025]

    if len(rebal_dates) < 1:
        return None, [], {}

    portfolio_value = 100.0
    positions = {}
    trades = []
    daily_values = []

    # Add end date
    end_date = price_df.index[-1]
    rebal_dates_with_end = rebal_dates + [end_date]

    for i, date in enumerate(rebal_dates_with_end[:-1]):
        next_date = rebal_dates_with_end[i + 1]

        try:
            mom = momentum.loc[date].dropna()
        except KeyError:
            continue

        if len(mom) < 1:
            continue

        # Select top N by momentum
        top_tickers = mom.nlargest(min(top_n, len(mom))).index.tolist()

        # Close positions not in new selection
        for ticker in list(positions.keys()):
            if ticker not in top_tickers:
                pos = positions[ticker]
                exit_price = price_df.loc[date, ticker]
                pnl_pct = (exit_price / pos['entry_price'] - 1) * 100

                trades.append(Trade(
                    entry_date=pos['entry_date'],
                    exit_date=date,
                    ticker=ticker,
                    entry_price=pos['entry_price'],
                    exit_price=exit_price,
                    high_price=pos['high_price'],
                    stop_price=pos['stop_price'],
                    exit_reason='rebalance',
                    pnl_pct=pnl_pct
                ))
                del positions[ticker]

        # Open new positions
        for ticker in top_tickers:
            if ticker not in positions:
                entry_price = price_df.loc[date, ticker]
                positions[ticker] = {
                    'entry_date': date,
                    'entry_price': entry_price,
                    'high_price': entry_price,
                    'stop_price': entry_price * (1 - stop_loss),
                }

        # Simulate period with stop checks
        period_dates = price_df.loc[date:next_date].index[1:]

        for day in period_dates:
            # Check stops
            for ticker in list(positions.keys()):
                pos = positions[ticker]
                try:
                    current_price = price_df.loc[day, ticker]
                except:
                    continue

                # Update trailing stop
                if trailing_stop and current_price > pos['high_price']:
                    pos['high_price'] = current_price
                    pos['stop_price'] = current_price * (1 - stop_loss)

                if current_price <= pos['stop_price']:
                    pnl_pct = (current_price / pos['entry_price'] - 1) * 100

                    trades.append(Trade(
                        entry_date=pos['entry_date'],
                        exit_date=day,
                        ticker=ticker,
                        entry_price=pos['entry_price'],
                        exit_price=current_price,
                        high_price=pos['high_price'],
                        stop_price=pos['stop_price'],
                        exit_reason='stop_loss',
                        pnl_pct=pnl_pct
                    ))
                    del positions[ticker]

            # Calculate portfolio value
            if positions:
                daily_return = 0
                weight = 1.0 / len(positions)
                for ticker, pos in positions.items():
                    try:
                        prev_idx = price_df.index.get_loc(day) - 1
                        if prev_idx >= 0:
                            prev_price = price_df.iloc[prev_idx][ticker]
                            curr_price = price_df.loc[day, ticker]
                            daily_return += (curr_price / prev_price - 1) * weight
                    except:
                        pass

                portfolio_value *= (1 + daily_return)

            daily_values.append({'date': day, 'value': portfolio_value})

    # Close remaining positions at end
    last_date = price_df.index[-1]
    for ticker, pos in list(positions.items()):
        try:
            exit_price = price_df.loc[last_date, ticker]
            pnl_pct = (exit_price / pos['entry_price'] - 1) * 100

            trades.append(Trade(
                entry_date=pos['entry_date'],
                exit_date=last_date,
                ticker=ticker,
                entry_price=pos['entry_price'],
                exit_price=exit_price,
                high_price=pos['high_price'],
                stop_price=pos['stop_price'],
                exit_reason='end_of_period',
                pnl_pct=pnl_pct
            ))
        except:
            pass

    # Create equity curve
    equity_df = pd.DataFrame(daily_values)
    if len(equity_df) == 0:
        return None, trades, {}

    equity_df['date'] = pd.to_datetime(equity_df['date'])
    equity_df = equity_df.set_index('date')

    return equity_df, trades, {}


def main():
    print("=" * 70)
    print("2025 OUT-OF-SAMPLE TEST: MEGA_UNIVERSE + 3% TRAILING STOP")
    print("=" * 70)

    # Download data (need some 2024 for lookback)
    print("\nDownloading ETF data...")
    all_tickers = MEGA_UNIVERSE + ['SPY']
    df = yf.download(all_tickers, start='2024-11-01',
                     end=datetime.now().strftime('%Y-%m-%d'),
                     auto_adjust=True, progress=False)

    if isinstance(df.columns, pd.MultiIndex):
        prices = df['Close']
    else:
        prices = df

    prices = prices.dropna(how='all').ffill()

    available = [t for t in MEGA_UNIVERSE if t in prices.columns]
    print(f"Available ETFs: {len(available)}/58")
    print(f"Date range: {prices.index[0].date()} to {prices.index[-1].date()}")

    # Get SPY return for 2025
    spy_2025 = prices['SPY'][prices.index >= '2025-01-01']
    spy_return = (spy_2025.iloc[-1] / spy_2025.iloc[0] - 1) * 100

    # Test configurations
    configs = [
        {'name': '3% Trailing (Champion)', 'stop': 0.03, 'trailing': True},
        {'name': '3% Fixed', 'stop': 0.03, 'trailing': False},
        {'name': '5% Trailing', 'stop': 0.05, 'trailing': True},
        {'name': 'No Stop', 'stop': 0.50, 'trailing': False},
    ]

    results = []

    for cfg in configs:
        equity, trades, _ = backtest_2025(
            prices, available,
            lookback=21,
            top_n=5,
            stop_loss=cfg['stop'],
            trailing_stop=cfg['trailing']
        )

        if equity is None or len(equity) == 0:
            print(f"\n{cfg['name']}: No valid trades")
            continue

        # Calculate 2025 metrics
        equity_2025 = equity[equity.index >= '2025-01-01']
        if len(equity_2025) < 2:
            continue

        total_return = (equity_2025['value'].iloc[-1] / equity_2025['value'].iloc[0] - 1) * 100
        max_dd = ((equity_2025['value'] - equity_2025['value'].cummax()) / equity_2025['value'].cummax()).min() * 100

        # Filter trades to 2025
        trades_2025 = [t for t in trades if t.entry_date.year == 2025]
        stopped = [t for t in trades_2025 if t.exit_reason == 'stop_loss']
        winners = [t for t in trades_2025 if t.pnl_pct and t.pnl_pct > 0]

        alpha = total_return - spy_return

        results.append({
            'config': cfg['name'],
            'return': total_return,
            'spy': spy_return,
            'alpha': alpha,
            'max_dd': max_dd,
            'trades': len(trades_2025),
            'stopped': len(stopped),
            'winners': len(winners),
            'win_rate': len(winners) / len(trades_2025) * 100 if trades_2025 else 0
        })

        print(f"\n{cfg['name']}:")
        print(f"  Return: {total_return:+.1f}% | SPY: {spy_return:+.1f}% | Alpha: {alpha:+.1f}%")
        print(f"  MaxDD: {max_dd:.1f}% | Trades: {len(trades_2025)} | Stopped: {len(stopped)}")

    # Results summary
    print("\n" + "=" * 80)
    print("2025 OUT-OF-SAMPLE RESULTS")
    print("=" * 80)
    print(f"\nSPY 2025 YTD: {spy_return:+.1f}%")
    print()
    print(f"{'Config':<25} {'Return':>10} {'Alpha':>10} {'MaxDD':>10} {'Trades':>8} {'Stop%':>8} {'Win%':>8}")
    print("-" * 80)

    for r in results:
        stop_pct = r['stopped'] / r['trades'] * 100 if r['trades'] > 0 else 0
        print(f"{r['config']:<25} {r['return']:>+9.1f}% {r['alpha']:>+9.1f}% {r['max_dd']:>+9.1f}% "
              f"{r['trades']:>8} {stop_pct:>7.0f}% {r['win_rate']:>7.1f}%")

    # Trade audit for champion config
    print("\n" + "=" * 80)
    print("TRADE AUDIT: 3% TRAILING STOP")
    print("=" * 80)

    equity, trades, _ = backtest_2025(
        prices, available,
        lookback=21,
        top_n=5,
        stop_loss=0.03,
        trailing_stop=True
    )

    trades_2025 = [t for t in trades if t.entry_date.year == 2025]

    # Sort by P&L
    sorted_trades = sorted([t for t in trades_2025 if t.pnl_pct], key=lambda x: x.pnl_pct, reverse=True)

    print("\nTop 5 Winners:")
    for t in sorted_trades[:5]:
        print(f"  {t.ticker}: {t.entry_date.strftime('%m/%d')} → {t.exit_date.strftime('%m/%d')} = {t.pnl_pct:+.1f}% ({t.exit_reason})")

    print("\nTop 5 Losers:")
    for t in sorted_trades[-5:]:
        print(f"  {t.ticker}: {t.entry_date.strftime('%m/%d')} → {t.exit_date.strftime('%m/%d')} = {t.pnl_pct:+.1f}% ({t.exit_reason})")

    # Monthly breakdown
    print("\nMonthly Breakdown:")
    for month in range(1, 13):
        month_trades = [t for t in trades_2025 if t.entry_date.month == month]
        if month_trades:
            month_pnl = sum(t.pnl_pct for t in month_trades if t.pnl_pct)
            month_stopped = len([t for t in month_trades if t.exit_reason == 'stop_loss'])
            print(f"  {month:02d}/2025: {len(month_trades)} trades, {month_pnl:+.1f}% P&L, {month_stopped} stopped")

    # What's currently held
    print("\nCurrently Held Positions:")
    open_positions = [t for t in trades if t.exit_reason == 'end_of_period']
    for t in open_positions:
        print(f"  {t.ticker}: Entry {t.entry_date.strftime('%m/%d')} @ ${t.entry_price:.2f} → ${t.exit_price:.2f} = {t.pnl_pct:+.1f}%")

    return results


if __name__ == '__main__':
    results = main()
