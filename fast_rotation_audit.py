"""
Fast Rotation Strategy - Trade Audit & Additional Universe Testing

Purpose: Validate realistic strategies with year-by-year breakdown and trade-level audit
"""

import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta
from dataclasses import dataclass
from typing import List, Dict, Optional, Tuple
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# ETF UNIVERSES
# ============================================================================

UNIVERSES = {
    'US_SECTORS': ['XLK', 'XLV', 'XLF', 'XLY', 'XLP', 'XLE', 'XLI', 'XLB', 'XLU', 'XLRE', 'XLC'],

    'INDUSTRIES': ['XHB', 'XRT', 'XME', 'XOP', 'KRE', 'KBE', 'ITB', 'XAR'],

    'INCOME': ['VYM', 'SCHD', 'DVY', 'HDV', 'SPHD', 'SPYD', 'VIG'],

    'BONDS': ['TLT', 'IEF', 'SHY', 'BND', 'AGG', 'LQD', 'HYG', 'JNK', 'EMB'],

    'COMMODITIES': ['GLD', 'SLV', 'GDX', 'GDXJ', 'USO', 'UNG', 'DBA', 'DBC'],

    'VOLATILITY': ['VXX', 'UVXY', 'SVXY', 'VIXY'],

    'CRYPTO': ['BITO', 'GBTC', 'COIN', 'MARA', 'RIOT', 'MSTR'],

    'SIZE': ['IWM', 'IWO', 'IWN', 'IJR', 'IJH', 'VB', 'VO', 'MDY'],

    'THEMATIC': ['ARKK', 'ARKG', 'ARKF', 'ARKW', 'QQQ', 'IGV', 'SOXX', 'SMH',
                 'XBI', 'TAN', 'ICLN', 'LIT', 'BOTZ'],
}

# ============================================================================
# DATA FUNCTIONS
# ============================================================================

def download_universe(tickers: List[str], start: str = '2020-01-01') -> pd.DataFrame:
    """Download price data for universe."""
    all_tickers = list(set(tickers + ['SPY']))

    print(f"Downloading {len(all_tickers)} tickers...")
    df = yf.download(all_tickers, start=start, end=datetime.now().strftime('%Y-%m-%d'),
                     auto_adjust=True, progress=False)

    if isinstance(df.columns, pd.MultiIndex):
        prices = df['Close']
    else:
        prices = df[['Close']].rename(columns={'Close': all_tickers[0]})

    prices = prices.dropna(how='all')

    # Fill gaps
    prices = prices.ffill()

    available = [t for t in tickers if t in prices.columns]
    print(f"Available: {len(available)}/{len(tickers)} tickers")
    print(f"Date range: {prices.index[0].date()} to {prices.index[-1].date()}")

    return prices, available


@dataclass
class Trade:
    """Individual trade record for auditing."""
    entry_date: datetime
    exit_date: Optional[datetime]
    ticker: str
    entry_price: float
    exit_price: Optional[float]
    momentum: float
    ema_value: Optional[float]
    stop_price: Optional[float]
    exit_reason: str  # 'rebalance', 'stop_loss', 'end_of_period'
    pnl_pct: Optional[float]

    def __repr__(self):
        exit_str = self.exit_date.strftime('%Y-%m-%d') if self.exit_date else 'OPEN'
        pnl_str = f"{self.pnl_pct:+.1f}%" if self.pnl_pct else 'N/A'
        exit_reason = f" ({self.exit_reason})" if self.exit_reason else ""
        return f"{self.ticker}: {self.entry_date.strftime('%Y-%m-%d')} → {exit_str} = {pnl_str}{exit_reason}"


# ============================================================================
# BACKTEST WITH TRADE AUDIT
# ============================================================================

def backtest_with_audit(
    prices: pd.DataFrame,
    tickers: List[str],
    lookback: int = 21,
    top_n: int = 3,
    rebalance: str = 'W',  # 'D', 'W', 'M'
    ema_period: Optional[int] = None,
    stop_loss: Optional[float] = None,
    verbose: bool = False
) -> Tuple[pd.DataFrame, List[Trade], Dict]:
    """
    Backtest with full trade audit trail.

    Returns:
        - equity_curve: Daily portfolio values
        - trades: List of all trades with full details
        - yearly_stats: Year-by-year breakdown
    """
    # Filter to available tickers
    available = [t for t in tickers if t in prices.columns]
    if len(available) < top_n:
        return None, [], {}

    price_df = prices[available].copy()

    # Calculate momentum
    momentum = price_df.pct_change(lookback)

    # Calculate EMA if needed
    ema_df = None
    if ema_period:
        ema_df = price_df.ewm(span=ema_period, adjust=False).mean()

    # Get rebalance dates
    if rebalance == 'D':
        rebal_dates = price_df.index.tolist()
    elif rebalance == 'W':
        rebal_dates = price_df.resample('W').last().index.tolist()
    else:  # Monthly
        rebal_dates = price_df.resample('M').last().index.tolist()

    # Filter to dates in our data
    rebal_dates = [d for d in rebal_dates if d in price_df.index]

    # Skip initial lookback period
    min_date = price_df.index[lookback + (ema_period or 0)]
    rebal_dates = [d for d in rebal_dates if d >= min_date]

    if len(rebal_dates) < 2:
        return None, [], {}

    # Track portfolio
    portfolio_value = 100.0
    positions = {}  # ticker -> {entry_price, entry_date, stop_price, shares}
    trades = []
    daily_values = []

    for i, date in enumerate(rebal_dates[:-1]):
        next_date = rebal_dates[i + 1]

        # Get momentum on rebalance date
        try:
            mom = momentum.loc[date].dropna()
        except KeyError:
            continue

        # Apply EMA filter
        if ema_df is not None:
            try:
                ema_vals = ema_df.loc[date]
                current_prices = price_df.loc[date]
                # Only consider tickers above their EMA
                above_ema = [t for t in mom.index
                            if t in current_prices.index and t in ema_vals.index
                            and current_prices[t] > ema_vals[t]]
                mom = mom[mom.index.isin(above_ema)]
            except KeyError:
                continue

        if len(mom) < 1:
            # No valid signals - hold cash
            daily_values.append({'date': date, 'value': portfolio_value})
            continue

        # Select top N
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
                    momentum=pos['momentum'],
                    ema_value=pos.get('ema_value'),
                    stop_price=pos.get('stop_price'),
                    exit_reason='rebalance',
                    pnl_pct=pnl_pct
                ))

                del positions[ticker]

        # Open new positions
        for ticker in top_tickers:
            if ticker not in positions:
                entry_price = price_df.loc[date, ticker]
                stop_price = entry_price * (1 - stop_loss) if stop_loss else None
                ema_val = ema_df.loc[date, ticker] if ema_df is not None else None

                positions[ticker] = {
                    'entry_date': date,
                    'entry_price': entry_price,
                    'stop_price': stop_price,
                    'momentum': mom.get(ticker, 0),
                    'ema_value': ema_val,
                    'shares': 1.0 / len(top_tickers)  # Equal weight
                }

        # Simulate daily with stop loss checks
        period_dates = price_df.loc[date:next_date].index[1:]  # Skip entry date

        for day in period_dates:
            # Check stop losses
            if stop_loss:
                for ticker in list(positions.keys()):
                    pos = positions[ticker]
                    current_price = price_df.loc[day, ticker]

                    if current_price <= pos['stop_price']:
                        pnl_pct = (current_price / pos['entry_price'] - 1) * 100

                        trades.append(Trade(
                            entry_date=pos['entry_date'],
                            exit_date=day,
                            ticker=ticker,
                            entry_price=pos['entry_price'],
                            exit_price=current_price,
                            momentum=pos['momentum'],
                            ema_value=pos.get('ema_value'),
                            stop_price=pos['stop_price'],
                            exit_reason='stop_loss',
                            pnl_pct=pnl_pct
                        ))

                        del positions[ticker]

            # Calculate portfolio value
            if positions:
                daily_return = 0
                for ticker, pos in positions.items():
                    prev_price = price_df.loc[:day].iloc[-2][ticker] if len(price_df.loc[:day]) > 1 else pos['entry_price']
                    curr_price = price_df.loc[day, ticker]
                    daily_return += (curr_price / prev_price - 1) * pos['shares']

                # Normalize by number of positions
                if len(positions) > 0:
                    portfolio_value *= (1 + daily_return / len(positions) * len(positions))

            daily_values.append({'date': day, 'value': portfolio_value})

    # Close remaining positions
    last_date = rebal_dates[-1]
    for ticker, pos in positions.items():
        exit_price = price_df.loc[last_date, ticker]
        pnl_pct = (exit_price / pos['entry_price'] - 1) * 100

        trades.append(Trade(
            entry_date=pos['entry_date'],
            exit_date=last_date,
            ticker=ticker,
            entry_price=pos['entry_price'],
            exit_price=exit_price,
            momentum=pos['momentum'],
            ema_value=pos.get('ema_value'),
            stop_price=pos.get('stop_price'),
            exit_reason='end_of_period',
            pnl_pct=pnl_pct
        ))

    # Create equity curve
    equity_df = pd.DataFrame(daily_values)
    if len(equity_df) == 0:
        return None, trades, {}

    equity_df['date'] = pd.to_datetime(equity_df['date'])
    equity_df = equity_df.set_index('date')

    # Calculate yearly stats
    yearly_stats = calculate_yearly_stats(equity_df, trades, prices)

    return equity_df, trades, yearly_stats


def calculate_yearly_stats(equity_df: pd.DataFrame, trades: List[Trade],
                          prices: pd.DataFrame) -> Dict:
    """Calculate year-by-year statistics."""
    stats = {}

    for year in range(2020, 2026):
        year_equity = equity_df[equity_df.index.year == year]
        year_trades = [t for t in trades if t.entry_date.year == year]

        if len(year_equity) < 2:
            continue

        start_val = year_equity['value'].iloc[0]
        end_val = year_equity['value'].iloc[-1]
        year_return = (end_val / start_val - 1) * 100

        # SPY return for comparison
        spy_year = prices['SPY'][prices['SPY'].index.year == year]
        if len(spy_year) > 1:
            spy_return = (spy_year.iloc[-1] / spy_year.iloc[0] - 1) * 100
        else:
            spy_return = 0

        # Max drawdown
        cummax = year_equity['value'].cummax()
        drawdown = (year_equity['value'] - cummax) / cummax * 100
        max_dd = drawdown.min()

        # Trade stats
        winning_trades = [t for t in year_trades if t.pnl_pct and t.pnl_pct > 0]
        stopped_out = [t for t in year_trades if t.exit_reason == 'stop_loss']

        stats[year] = {
            'return': year_return,
            'spy_return': spy_return,
            'alpha': year_return - spy_return,
            'max_dd': max_dd,
            'num_trades': len(year_trades),
            'win_rate': len(winning_trades) / len(year_trades) * 100 if year_trades else 0,
            'stop_outs': len(stopped_out),
            'avg_trade': np.mean([t.pnl_pct for t in year_trades if t.pnl_pct]) if year_trades else 0
        }

    return stats


def print_audit_report(trades: List[Trade], yearly_stats: Dict,
                       strategy_name: str, show_trades: int = 20):
    """Print detailed audit report."""
    print("\n" + "="*80)
    print(f"TRADE AUDIT REPORT: {strategy_name}")
    print("="*80)

    # Yearly breakdown
    print("\n📊 YEAR-BY-YEAR PERFORMANCE")
    print("-"*80)
    print(f"{'Year':<6} {'Return':>10} {'SPY':>10} {'Alpha':>10} {'MaxDD':>10} {'Trades':>8} {'Win%':>8} {'Stops':>8}")
    print("-"*80)

    total_return = 100
    total_spy = 100

    for year, s in sorted(yearly_stats.items()):
        print(f"{year:<6} {s['return']:>+10.1f}% {s['spy_return']:>+9.1f}% {s['alpha']:>+9.1f}% "
              f"{s['max_dd']:>+9.1f}% {s['num_trades']:>8} {s['win_rate']:>7.0f}% {s['stop_outs']:>8}")
        total_return *= (1 + s['return']/100)
        total_spy *= (1 + s['spy_return']/100)

    print("-"*80)
    print(f"{'TOTAL':<6} {(total_return-100):>+10.1f}% {(total_spy-100):>+9.1f}% {(total_return-total_spy):>+9.1f}%")

    # Trade summary
    print("\n📈 TRADE STATISTICS")
    print("-"*80)
    total_trades = len(trades)
    stopped_out = [t for t in trades if t.exit_reason == 'stop_loss']
    rebalanced = [t for t in trades if t.exit_reason == 'rebalance']
    winners = [t for t in trades if t.pnl_pct and t.pnl_pct > 0]
    losers = [t for t in trades if t.pnl_pct and t.pnl_pct < 0]

    print(f"Total Trades: {total_trades}")
    print(f"Winners: {len(winners)} ({len(winners)/total_trades*100:.1f}%)")
    print(f"Losers: {len(losers)} ({len(losers)/total_trades*100:.1f}%)")
    print(f"Stopped Out: {len(stopped_out)} ({len(stopped_out)/total_trades*100:.1f}%)")
    print(f"Rebalanced: {len(rebalanced)} ({len(rebalanced)/total_trades*100:.1f}%)")

    if winners:
        print(f"\nAvg Winner: +{np.mean([t.pnl_pct for t in winners]):.2f}%")
    if losers:
        print(f"Avg Loser: {np.mean([t.pnl_pct for t in losers]):.2f}%")
    if stopped_out:
        print(f"Avg Stop Loss: {np.mean([t.pnl_pct for t in stopped_out]):.2f}%")

    # Sample trades
    print(f"\n📋 SAMPLE TRADES (First {show_trades})")
    print("-"*80)
    for trade in trades[:show_trades]:
        print(f"  {trade}")

    # Biggest winners/losers
    print("\n🏆 TOP 5 WINNERS")
    print("-"*80)
    sorted_by_pnl = sorted([t for t in trades if t.pnl_pct], key=lambda x: x.pnl_pct, reverse=True)
    for trade in sorted_by_pnl[:5]:
        print(f"  {trade}")

    print("\n💀 TOP 5 LOSERS")
    print("-"*80)
    for trade in sorted_by_pnl[-5:]:
        print(f"  {trade}")

    # Stop loss analysis
    if stopped_out:
        print("\n🛑 STOP LOSS ANALYSIS")
        print("-"*80)
        print(f"Total Stop-Outs: {len(stopped_out)}")
        print(f"Avg Loss on Stop: {np.mean([t.pnl_pct for t in stopped_out]):.2f}%")

        # Stop-outs by ticker
        stop_by_ticker = {}
        for t in stopped_out:
            stop_by_ticker[t.ticker] = stop_by_ticker.get(t.ticker, 0) + 1

        print("\nMost Stopped Tickers:")
        for ticker, count in sorted(stop_by_ticker.items(), key=lambda x: -x[1])[:5]:
            print(f"  {ticker}: {count} times")


# ============================================================================
# MAIN EXECUTION
# ============================================================================

def run_universe_sweep():
    """Test all additional universes."""

    results = []

    # Best params from previous sweep
    params = {
        'lookback': 21,
        'top_n': 4,
        'rebalance': 'M',
        'ema_period': None,
        'stop_loss': 0.03
    }

    print("="*80)
    print("ADDITIONAL UNIVERSE TESTING")
    print("="*80)
    print(f"Parameters: {params}")
    print()

    for universe_name, tickers in UNIVERSES.items():
        print(f"\n{'='*60}")
        print(f"Testing: {universe_name}")
        print(f"{'='*60}")

        prices, available = download_universe(tickers)

        if len(available) < 2:
            print(f"Skipping {universe_name} - insufficient data")
            continue

        equity, trades, yearly = backtest_with_audit(
            prices, available,
            lookback=params['lookback'],
            top_n=min(params['top_n'], len(available)),
            rebalance=params['rebalance'],
            ema_period=params['ema_period'],
            stop_loss=params['stop_loss']
        )

        if equity is None or len(equity) == 0:
            print(f"Skipping {universe_name} - no valid trades")
            continue

        # Calculate metrics
        total_return = (equity['value'].iloc[-1] / 100 - 1) * 100

        returns = equity['value'].pct_change().dropna()
        sharpe = returns.mean() / returns.std() * np.sqrt(252) if len(returns) > 0 and returns.std() > 0 else 0

        cummax = equity['value'].cummax()
        max_dd = ((equity['value'] - cummax) / cummax).min() * 100

        results.append({
            'universe': universe_name,
            'tickers': len(available),
            'total_return': total_return,
            'sharpe': sharpe,
            'max_dd': max_dd,
            'trades': len(trades),
            'stop_outs': len([t for t in trades if t.exit_reason == 'stop_loss'])
        })

        print_audit_report(trades, yearly, universe_name, show_trades=10)

    # Summary table
    print("\n" + "="*80)
    print("UNIVERSE COMPARISON SUMMARY")
    print("="*80)
    print(f"{'Universe':<15} {'Tickers':>8} {'Return':>12} {'Sharpe':>8} {'MaxDD':>10} {'Trades':>8} {'Stops':>8}")
    print("-"*80)

    for r in sorted(results, key=lambda x: -x['sharpe']):
        print(f"{r['universe']:<15} {r['tickers']:>8} {r['total_return']:>+11.1f}% {r['sharpe']:>8.2f} "
              f"{r['max_dd']:>+9.1f}% {r['trades']:>8} {r['stop_outs']:>8}")

    return results


def run_detailed_audit_best_strategy():
    """Detailed audit of the best realistic strategy: US Sectors + 3% Stop."""

    print("="*80)
    print("DETAILED AUDIT: US SECTORS + 3% STOP LOSS")
    print("="*80)

    tickers = UNIVERSES['US_SECTORS']
    prices, available = download_universe(tickers)

    # Best params for US Sectors
    params = {
        'lookback': 21,
        'top_n': 4,
        'rebalance': 'M',
        'ema_period': None,
        'stop_loss': 0.03
    }

    print(f"\nParameters: Lookback={params['lookback']}d, Top={params['top_n']}, "
          f"Rebalance={params['rebalance']}, Stop={params['stop_loss']*100}%")

    equity, trades, yearly = backtest_with_audit(
        prices, available,
        **params
    )

    print_audit_report(trades, yearly, "US Sectors (Monthly, 3% Stop)", show_trades=30)

    return equity, trades, yearly


if __name__ == '__main__':
    # First: Detailed audit of best realistic strategy
    print("\n" + "#"*80)
    print("# PART 1: DETAILED AUDIT OF BEST STRATEGY")
    print("#"*80)
    equity, trades, yearly = run_detailed_audit_best_strategy()

    # Second: Test all additional universes
    print("\n" + "#"*80)
    print("# PART 2: ADDITIONAL UNIVERSE TESTING")
    print("#"*80)
    results = run_universe_sweep()
