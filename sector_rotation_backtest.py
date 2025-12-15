"""
Strategy: Sector ETF Rotation (6-Month Momentum)

Based on successful pattern from S&P 500 Momentum Strategy (37-80% annual returns)
Cross-sectional momentum: Buy top N sectors by 6-month returns, rebalance monthly

Hypothesis: If stock-level momentum worked, sector-level should work too
with lower transaction costs and simpler implementation.

Data: yfinance sector ETFs (2019-2024)
"""

import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Sector ETFs (11 S&P 500 sectors)
SECTOR_ETFS = {
    'XLK': 'Technology',
    'XLV': 'Healthcare',
    'XLF': 'Financials',
    'XLY': 'Consumer Discretionary',
    'XLP': 'Consumer Staples',
    'XLE': 'Energy',
    'XLI': 'Industrials',
    'XLB': 'Materials',
    'XLU': 'Utilities',
    'XLRE': 'Real Estate',
    'XLC': 'Communication Services'
}

# Parameters
LOOKBACK_MONTHS = 6  # Momentum lookback period
TOP_N = 3  # Number of sectors to hold
REBALANCE_FREQ = 'M'  # Monthly rebalancing


def download_data(start='2019-01-01', end='2024-12-01'):
    """Download sector ETF and SPY data."""
    tickers = list(SECTOR_ETFS.keys()) + ['SPY']

    print(f"Downloading {len(tickers)} tickers...")
    df = yf.download(tickers, start=start, end=end, auto_adjust=True, progress=False)

    # Get close prices
    prices = df['Close'].copy()

    # Drop any rows with all NaN
    prices = prices.dropna(how='all')

    print(f"Data: {prices.index[0].date()} to {prices.index[-1].date()}, {len(prices)} days")

    return prices


def calculate_momentum(prices, lookback_months=6):
    """Calculate N-month momentum (total return) for each ETF."""
    lookback_days = lookback_months * 21  # ~21 trading days per month

    # Simple momentum: return over lookback period
    momentum = prices.pct_change(lookback_days)

    return momentum


def get_monthly_rebalance_dates(prices):
    """Get the last trading day of each month from actual price data."""
    # Group by year-month and get the last index (date) of each group
    monthly_last = prices.groupby(pd.Grouper(freq='M')).apply(lambda x: x.index[-1] if len(x) > 0 else None)

    # Remove None values and get just the dates
    rebalance_dates = [d for d in monthly_last if d is not None and d in prices.index]

    return rebalance_dates


def backtest_sector_rotation(prices, lookback_months=6, top_n=3):
    """
    Backtest sector rotation strategy.

    Strategy:
    1. Calculate 6-month momentum for each sector
    2. At month end, rank sectors by momentum
    3. Hold top N sectors equally weighted
    4. Rebalance monthly
    """
    sector_tickers = list(SECTOR_ETFS.keys())
    sector_prices = prices[sector_tickers].copy()
    spy_prices = prices['SPY'].copy()

    # Calculate momentum
    momentum = calculate_momentum(sector_prices, lookback_months)

    # Get actual rebalance dates from data
    rebalance_dates = get_monthly_rebalance_dates(sector_prices)

    # Skip first lookback_months (need history for momentum)
    start_idx = lookback_months
    rebalance_dates = rebalance_dates[start_idx:]

    print(f"Rebalancing on {len(rebalance_dates)} month-end dates")

    # Track portfolio
    portfolio_value = 100.0
    portfolio_history = []
    holdings = {}
    trades = []

    for i, date in enumerate(rebalance_dates[:-1]):
        next_date = rebalance_dates[i + 1]

        # Get momentum on rebalance date
        try:
            sector_mom = momentum.loc[date].dropna()
        except KeyError:
            # Find closest prior date
            valid_dates = momentum.index[momentum.index <= date]
            if len(valid_dates) == 0:
                continue
            closest_date = valid_dates[-1]
            sector_mom = momentum.loc[closest_date].dropna()

        if len(sector_mom) < top_n:
            continue

        # Rank and select top N
        top_sectors = sector_mom.nlargest(top_n).index.tolist()

        # Calculate returns for holding period
        try:
            period_returns = (sector_prices.loc[next_date] / sector_prices.loc[date] - 1)
        except KeyError:
            continue

        # Equal weight portfolio return
        portfolio_return = period_returns[top_sectors].mean()

        # Update portfolio value
        portfolio_value *= (1 + portfolio_return)

        portfolio_history.append({
            'date': next_date,
            'value': portfolio_value,
            'return': portfolio_return,
            'holdings': top_sectors,
            'momentum': {s: sector_mom.get(s, 0) for s in top_sectors}
        })

        trades.append({
            'date': date,
            'sectors': top_sectors,
            'sector_names': [SECTOR_ETFS[s] for s in top_sectors],
            'momentum_scores': [sector_mom[s] for s in top_sectors]
        })

    return portfolio_history, trades


def calculate_metrics(portfolio_history, spy_prices):
    """Calculate performance metrics."""
    if not portfolio_history:
        return {}

    # Convert to DataFrame
    df = pd.DataFrame(portfolio_history)
    df['date'] = pd.to_datetime(df['date'])
    df = df.set_index('date')

    returns = df['return']

    # Total return
    total_return = (df['value'].iloc[-1] / 100 - 1) * 100

    # Annualized return
    years = (df.index[-1] - df.index[0]).days / 365
    annual_return = ((df['value'].iloc[-1] / 100) ** (1/years) - 1) * 100

    # Sharpe ratio (annualized)
    sharpe = returns.mean() / returns.std() * np.sqrt(12) if returns.std() > 0 else 0

    # Max drawdown
    cumulative = df['value']
    running_max = cumulative.cummax()
    drawdown = (cumulative - running_max) / running_max * 100
    max_dd = drawdown.min()

    # Win rate
    win_rate = (returns > 0).sum() / len(returns) * 100

    # SPY benchmark
    spy_start = spy_prices.loc[df.index[0]:].iloc[0]
    spy_end = spy_prices.loc[:df.index[-1]].iloc[-1]
    spy_total = (spy_end / spy_start - 1) * 100
    spy_annual = ((spy_end / spy_start) ** (1/years) - 1) * 100

    return {
        'total_return': total_return,
        'annual_return': annual_return,
        'sharpe': sharpe,
        'max_dd': max_dd,
        'win_rate': win_rate,
        'months': len(returns),
        'years': years,
        'spy_total': spy_total,
        'spy_annual': spy_annual
    }


def analyze_holdings(trades):
    """Analyze which sectors were held most often."""
    sector_counts = {}
    for trade in trades:
        for sector in trade['sectors']:
            name = SECTOR_ETFS[sector]
            sector_counts[name] = sector_counts.get(name, 0) + 1

    return sorted(sector_counts.items(), key=lambda x: -x[1])


def print_results(metrics, trades):
    """Print formatted results."""
    print("\n" + "="*60)
    print("SECTOR ETF ROTATION - BACKTEST RESULTS")
    print("="*60)

    print(f"\nStrategy: Top {TOP_N} sectors by {LOOKBACK_MONTHS}-month momentum")
    print(f"Period: {metrics['years']:.1f} years ({metrics['months']} months)")

    print(f"\n{'PERFORMANCE':^60}")
    print("-"*60)
    print(f"Total Return:    {metrics['total_return']:>10.1f}%")
    print(f"Annual Return:   {metrics['annual_return']:>10.1f}%")
    print(f"Sharpe Ratio:    {metrics['sharpe']:>10.2f}")
    print(f"Max Drawdown:    {metrics['max_dd']:>10.1f}%")
    print(f"Win Rate:        {metrics['win_rate']:>10.1f}%")

    print(f"\n{'BENCHMARK COMPARISON':^60}")
    print("-"*60)
    print(f"SPY Total:       {metrics['spy_total']:>10.1f}%")
    print(f"SPY Annual:      {metrics['spy_annual']:>10.1f}%")
    print(f"Alpha (annual):  {metrics['annual_return'] - metrics['spy_annual']:>10.1f}%")

    # Sector frequency analysis
    print(f"\n{'MOST HELD SECTORS':^60}")
    print("-"*60)
    holdings_analysis = analyze_holdings(trades)
    for sector, count in holdings_analysis[:6]:
        pct = count / len(trades) * 100
        print(f"  {sector:<30} {count:>3} times ({pct:.0f}%)")

    # Recent holdings
    print(f"\n{'RECENT HOLDINGS':^60}")
    print("-"*60)
    for trade in trades[-3:]:
        print(f"  {trade['date'].strftime('%Y-%m')}: {', '.join(trade['sector_names'])}")


def parameter_sweep():
    """Test different parameters."""
    prices = download_data()

    print("\n" + "="*60)
    print("PARAMETER SWEEP")
    print("="*60)

    results = []

    for lookback in [3, 6, 9, 12]:
        for top_n in [1, 2, 3, 4, 5]:
            portfolio_history, trades = backtest_sector_rotation(
                prices,
                lookback_months=lookback,
                top_n=top_n
            )

            if not portfolio_history:
                continue

            metrics = calculate_metrics(portfolio_history, prices['SPY'])

            results.append({
                'lookback': lookback,
                'top_n': top_n,
                'annual_return': metrics['annual_return'],
                'sharpe': metrics['sharpe'],
                'max_dd': metrics['max_dd'],
                'alpha': metrics['annual_return'] - metrics['spy_annual']
            })

            print(f"Lookback={lookback}m, Top={top_n}: "
                  f"Return={metrics['annual_return']:.1f}%, "
                  f"Sharpe={metrics['sharpe']:.2f}, "
                  f"Alpha={metrics['annual_return'] - metrics['spy_annual']:.1f}%")

    # Find best configuration
    if results:
        best = max(results, key=lambda x: x['sharpe'])
        print(f"\nBest by Sharpe: Lookback={best['lookback']}m, Top={best['top_n']}")
        print(f"  Return={best['annual_return']:.1f}%, Sharpe={best['sharpe']:.2f}, Alpha={best['alpha']:.1f}%")

    return results


if __name__ == '__main__':
    print("="*60)
    print("SECTOR ETF ROTATION STRATEGY")
    print("="*60)
    print(f"Parameters: {LOOKBACK_MONTHS}-month momentum, Top {TOP_N} sectors")

    # Download data
    prices = download_data()

    # Calculate SPY benchmark
    spy_total = (prices['SPY'].iloc[-1] / prices['SPY'].iloc[0] - 1) * 100
    years = (prices.index[-1] - prices.index[0]).days / 365
    spy_annual = ((prices['SPY'].iloc[-1] / prices['SPY'].iloc[0]) ** (1/years) - 1) * 100
    print(f"SPY Benchmark: {spy_total:.1f}% total, {spy_annual:.1f}% annual")

    # Run backtest
    portfolio_history, trades = backtest_sector_rotation(prices)

    # Calculate metrics
    metrics = calculate_metrics(portfolio_history, prices['SPY'])

    # Print results
    print_results(metrics, trades)

    # Parameter sweep
    print("\n")
    sweep_results = parameter_sweep()
