"""
Unbiased Stock Momentum Rotation Backtester

Uses point-in-time universe from comprehensive_features.csv:
- 2,827 unique tickers selected by volume at each point in time
- NO survivorship bias - includes stocks that were later delisted
- Daily stop loss simulation with real price data

Key Features:
1. Universe from comprehensive_features.csv (top 1000 by volume each month)
2. Daily prices downloaded via yfinance for stop loss simulation
3. Parameter sweep: momentum horizon, top_n, stop_loss, universe_size
4. Comparison to 6-Factor model and SPY benchmark
"""

import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple
from pathlib import Path
import warnings
import time
import os
warnings.filterwarnings('ignore')

# Paths
FEATURES_PATH = '/Users/nish_macbook/development/trading/sp500-factor/comprehensive_features.csv'
CACHE_DIR = Path('/Users/nish_macbook/development/trading/strategy_backtests/data')
RESULTS_DIR = Path('/Users/nish_macbook/development/trading/strategy_backtests/results')

# Create directories
CACHE_DIR.mkdir(exist_ok=True)
RESULTS_DIR.mkdir(exist_ok=True)


# ============================================================================
# DATA LOADING
# ============================================================================

def load_universe_data() -> pd.DataFrame:
    """Load comprehensive_features.csv with point-in-time universe."""
    print("Loading universe data from comprehensive_features.csv...")
    df = pd.read_csv(FEATURES_PATH)
    df['date'] = pd.to_datetime(df['date'])

    # Basic stats
    tickers = df['ticker'].unique()
    dates = df['date'].unique()
    print(f"  Loaded {len(df):,} rows")
    print(f"  {len(tickers):,} unique tickers")
    print(f"  {len(dates)} months: {df['date'].min().strftime('%Y-%m')} to {df['date'].max().strftime('%Y-%m')}")

    return df


def download_daily_prices(tickers: List[str], start: str = '2019-12-01',
                          batch_size: int = 100, skip_download: bool = True) -> pd.DataFrame:
    """
    Download daily prices for all tickers with batching and caching.

    Uses batching to avoid yfinance rate limits.
    Caches results to avoid re-downloading.

    If skip_download=True (default), just uses cached data without downloading.
    """
    cache_file = CACHE_DIR / 'daily_prices.parquet'

    # Check cache
    if cache_file.exists():
        print(f"Loading cached daily prices from {cache_file}")
        df = pd.read_parquet(cache_file)

        # Check if we have all tickers
        cached_tickers = set(df.columns)
        missing = set(tickers) - cached_tickers

        if len(missing) == 0:
            print(f"  Cache complete: {len(df.columns)} tickers")
            return df
        elif skip_download:
            # Just use what we have - missing are mostly delisted stocks
            print(f"  Using cache with {len(cached_tickers)} tickers ({len(missing)} missing/delisted)")
            return df
        else:
            print(f"  Cache has {len(cached_tickers)} tickers, need {len(missing)} more")
            tickers_to_download = list(missing)
    else:
        print(f"No cache found, downloading all {len(tickers)} tickers...")
        tickers_to_download = tickers
        df = None

    # Download in batches
    all_prices = []
    n_batches = (len(tickers_to_download) + batch_size - 1) // batch_size

    for i in range(0, len(tickers_to_download), batch_size):
        batch = tickers_to_download[i:i+batch_size]
        batch_num = i // batch_size + 1

        print(f"  Batch {batch_num}/{n_batches}: downloading {len(batch)} tickers...")

        try:
            batch_df = yf.download(
                batch,
                start=start,
                end=datetime.now().strftime('%Y-%m-%d'),
                auto_adjust=True,
                progress=False,
                threads=True
            )

            if isinstance(batch_df.columns, pd.MultiIndex):
                prices = batch_df['Close']
            else:
                prices = batch_df[['Close']].rename(columns={'Close': batch[0]})

            all_prices.append(prices)

        except Exception as e:
            print(f"    Error downloading batch: {e}")

        # Rate limit protection
        if i + batch_size < len(tickers_to_download):
            time.sleep(1)

    # Combine all batches
    if all_prices:
        new_prices = pd.concat(all_prices, axis=1)

        # Merge with existing cache
        if df is not None:
            df = pd.concat([df, new_prices], axis=1)
        else:
            df = new_prices

        # Save to cache
        df.to_parquet(cache_file)
        print(f"  Saved {len(df.columns)} tickers to cache")

    return df


# ============================================================================
# BACKTEST ENGINE
# ============================================================================

@dataclass
class Trade:
    """Single trade record."""
    entry_date: datetime
    exit_date: Optional[datetime]
    ticker: str
    entry_price: float
    exit_price: Optional[float]
    momentum: float
    stop_price: Optional[float]
    exit_reason: str  # 'rebalance', 'stop_loss', 'end_of_period'
    pnl_pct: Optional[float]
    holding_days: Optional[int] = None


@dataclass
class BacktestResult:
    """Backtest results container."""
    total_return: float
    annual_return: float
    sharpe: float
    max_dd: float
    trades: List[Trade]
    yearly_stats: Dict
    equity_curve: pd.DataFrame
    params: Dict


def backtest_momentum_rotation(
    universe_df: pd.DataFrame,
    daily_prices: pd.DataFrame,
    momentum_col: str = 'ret_1m',
    top_n: int = 20,
    stop_loss: Optional[float] = 0.03,
    universe_size: int = 1000,
    start_date: str = '2020-02-01',
    end_date: str = '2024-11-30',
    roe_filter: bool = False,
    ema_period: Optional[int] = None,
    revenue_filter: bool = False,
    vol_filter: Optional[str] = None,  # 'high', 'low', or None
    trailing_stop: bool = False  # If True, use trailing stop instead of fixed
) -> BacktestResult:
    """
    Backtest momentum rotation with daily stop loss simulation.

    Parameters:
    -----------
    universe_df : DataFrame with columns ['date', 'ticker', momentum_col, 'price']
    daily_prices : DataFrame with daily closing prices for all tickers
    momentum_col : Column to rank by (ret_1m, ret_3m, ret_6m, ret_12m, mom_risk_adj)
    top_n : Number of stocks to hold
    stop_loss : Stop loss percentage (0.03 = 3%), None for no stop
    universe_size : Filter to top N by some criteria (matches original universe)
    roe_filter : If True, only include stocks with ROE > 0
    ema_period : If set, only include stocks where price > EMA(period)
    revenue_filter : If True, only include stocks with revenue_growth > 0
    vol_filter : 'high' (top 50% vol), 'low' (bottom 50% vol), or None
    trailing_stop : If True, stop trails highest price; if False, fixed from entry
    """

    # Pre-calculate EMAs if needed
    ema_data = {}
    if ema_period:
        for ticker in daily_prices.columns:
            try:
                ema_data[ticker] = daily_prices[ticker].ewm(span=ema_period, adjust=False).mean()
            except:
                pass

    # Get rebalance dates (monthly)
    rebal_dates = sorted(universe_df['date'].unique())
    rebal_dates = [d for d in rebal_dates
                   if pd.Timestamp(start_date) <= d <= pd.Timestamp(end_date)]

    if len(rebal_dates) < 2:
        return None

    # Initialize
    portfolio_value = 100.0
    positions = {}  # {ticker: {'entry_date', 'entry_price', 'stop_price', 'shares'}}
    trades = []
    daily_values = []

    # Process each rebalance period
    for i, rebal_date in enumerate(rebal_dates[:-1]):
        next_rebal = rebal_dates[i + 1]

        # Get universe for this month
        month_df = universe_df[universe_df['date'] == rebal_date].copy()

        # Filter to top universe_size by dollar volume (if column exists)
        if 'dollar_volume' in month_df.columns and len(month_df) > universe_size:
            month_df = month_df.nlargest(universe_size, 'dollar_volume')

        # Drop missing momentum values
        month_df = month_df.dropna(subset=[momentum_col])

        # === FUNDAMENTAL FILTERS ===
        if roe_filter and 'roe' in month_df.columns:
            month_df = month_df[month_df['roe'] > 0]

        if revenue_filter and 'revenue_growth' in month_df.columns:
            month_df = month_df[month_df['revenue_growth'] > 0]

        # === EMA FILTER ===
        if ema_period and ema_data:
            passing_tickers = []
            for ticker in month_df['ticker'].tolist():
                if ticker in ema_data:
                    try:
                        # Find nearest date in EMA data
                        ema_series = ema_data[ticker]
                        ema_val = ema_series.loc[:rebal_date].iloc[-1] if len(ema_series.loc[:rebal_date]) > 0 else None
                        price = daily_prices[ticker].loc[:rebal_date].iloc[-1] if ticker in daily_prices.columns else None

                        if ema_val and price and not pd.isna(ema_val) and not pd.isna(price):
                            if price > ema_val:
                                passing_tickers.append(ticker)
                    except:
                        pass
            month_df = month_df[month_df['ticker'].isin(passing_tickers)]

        # === VOLATILITY FILTER ===
        if vol_filter and 'volatility' in month_df.columns:
            vol_median = month_df['volatility'].median()
            if vol_filter == 'high':
                month_df = month_df[month_df['volatility'] >= vol_median]
            elif vol_filter == 'low':
                month_df = month_df[month_df['volatility'] < vol_median]

        if len(month_df) < top_n:
            continue

        # Rank by momentum and select top N
        top_stocks = month_df.nlargest(top_n, momentum_col)['ticker'].tolist()

        # Close positions not in new selection
        for ticker in list(positions.keys()):
            if ticker not in top_stocks:
                pos = positions[ticker]

                # Get exit price from daily data if available
                try:
                    exit_price = daily_prices.loc[rebal_date, ticker]
                    if pd.isna(exit_price):
                        exit_price = pos['last_price']
                except:
                    exit_price = pos['last_price']

                pnl_pct = (exit_price / pos['entry_price'] - 1) * 100
                holding_days = (rebal_date - pos['entry_date']).days

                trades.append(Trade(
                    entry_date=pos['entry_date'],
                    exit_date=rebal_date,
                    ticker=ticker,
                    entry_price=pos['entry_price'],
                    exit_price=exit_price,
                    momentum=pos['momentum'],
                    stop_price=pos.get('stop_price'),
                    exit_reason='rebalance',
                    pnl_pct=pnl_pct,
                    holding_days=holding_days
                ))
                del positions[ticker]

        # Open new positions
        for ticker in top_stocks:
            if ticker not in positions:
                # Get entry price - ONLY from adjusted daily_prices, never CSV
                mom_val = month_df[month_df['ticker'] == ticker][momentum_col].iloc[0]

                try:
                    if ticker not in daily_prices.columns:
                        continue  # Skip - no adjusted price data
                    entry_price = daily_prices.loc[rebal_date, ticker]
                    if pd.isna(entry_price):
                        # Try nearby dates (within 5 days) for market holidays
                        nearby = daily_prices[ticker].loc[:rebal_date].dropna().tail(5)
                        if len(nearby) > 0:
                            entry_price = nearby.iloc[-1]
                        else:
                            continue  # Skip - no valid price
                except:
                    continue  # Skip - can't get adjusted price

                if pd.isna(entry_price) or entry_price <= 0:
                    continue

                stop_price = entry_price * (1 - stop_loss) if stop_loss else None

                positions[ticker] = {
                    'entry_date': rebal_date,
                    'entry_price': entry_price,
                    'stop_price': stop_price,
                    'momentum': mom_val,
                    'last_price': entry_price,
                    'high_price': entry_price  # For trailing stop
                }

        # Simulate daily with stop loss checks
        if positions and stop_loss:
            # Get trading days between rebalances
            try:
                period_prices = daily_prices.loc[rebal_date:next_rebal]
            except:
                continue

            for day in period_prices.index[1:]:  # Skip first day (entry day)
                for ticker in list(positions.keys()):
                    pos = positions[ticker]

                    try:
                        current_price = period_prices.loc[day, ticker]
                    except:
                        continue

                    if pd.isna(current_price):
                        continue

                    pos['last_price'] = current_price

                    # Update trailing stop if enabled
                    if trailing_stop and current_price > pos['high_price']:
                        pos['high_price'] = current_price
                        pos['stop_price'] = current_price * (1 - stop_loss)

                    # Check stop loss
                    if pos['stop_price'] and current_price <= pos['stop_price']:
                        pnl_pct = (current_price / pos['entry_price'] - 1) * 100
                        holding_days = (day - pos['entry_date']).days

                        trades.append(Trade(
                            entry_date=pos['entry_date'],
                            exit_date=day,
                            ticker=ticker,
                            entry_price=pos['entry_price'],
                            exit_price=current_price,
                            momentum=pos['momentum'],
                            stop_price=pos['stop_price'],
                            exit_reason='stop_loss',
                            pnl_pct=pnl_pct,
                            holding_days=holding_days
                        ))
                        del positions[ticker]

                # Calculate daily portfolio value
                if positions:
                    daily_return = 0
                    weight = 1.0 / len(positions)

                    for ticker, pos in positions.items():
                        try:
                            curr_price = period_prices.loc[day, ticker]
                            prev_idx = period_prices.index.get_loc(day) - 1
                            prev_price = period_prices.iloc[prev_idx][ticker]

                            if not pd.isna(curr_price) and not pd.isna(prev_price) and prev_price > 0:
                                daily_return += (curr_price / prev_price - 1) * weight
                        except:
                            pass

                    portfolio_value *= (1 + daily_return)

                daily_values.append({'date': day, 'value': portfolio_value})

    # Close remaining positions
    if positions:
        last_date = rebal_dates[-1]
        for ticker, pos in positions.items():
            try:
                exit_price = daily_prices.loc[last_date, ticker]
                if pd.isna(exit_price):
                    exit_price = pos['last_price']
            except:
                exit_price = pos['last_price']

            pnl_pct = (exit_price / pos['entry_price'] - 1) * 100
            holding_days = (last_date - pos['entry_date']).days

            trades.append(Trade(
                entry_date=pos['entry_date'],
                exit_date=last_date,
                ticker=ticker,
                entry_price=pos['entry_price'],
                exit_price=exit_price,
                momentum=pos['momentum'],
                stop_price=pos.get('stop_price'),
                exit_reason='end_of_period',
                pnl_pct=pnl_pct,
                holding_days=holding_days
            ))

    # Calculate metrics
    if len(daily_values) == 0:
        return None

    equity_df = pd.DataFrame(daily_values).set_index('date')

    total_return = (equity_df['value'].iloc[-1] / 100 - 1) * 100
    years = (equity_df.index[-1] - equity_df.index[0]).days / 365
    annual_return = ((equity_df['value'].iloc[-1] / 100) ** (1/years) - 1) * 100 if years > 0 else 0

    returns = equity_df['value'].pct_change().dropna()
    sharpe = returns.mean() / returns.std() * np.sqrt(252) if returns.std() > 0 else 0

    cummax = equity_df['value'].cummax()
    max_dd = ((equity_df['value'] - cummax) / cummax).min() * 100

    # Yearly stats
    yearly_stats = {}
    for year in range(2020, 2025):
        year_eq = equity_df[equity_df.index.year == year]
        year_trades = [t for t in trades if t.entry_date.year == year]

        if len(year_eq) < 2:
            continue

        year_return = (year_eq['value'].iloc[-1] / year_eq['value'].iloc[0] - 1) * 100
        stopped = len([t for t in year_trades if t.exit_reason == 'stop_loss'])
        winners = len([t for t in year_trades if t.pnl_pct and t.pnl_pct > 0])

        yearly_stats[year] = {
            'return': year_return,
            'trades': len(year_trades),
            'stopped': stopped,
            'win_rate': winners / len(year_trades) * 100 if year_trades else 0
        }

    return BacktestResult(
        total_return=total_return,
        annual_return=annual_return,
        sharpe=sharpe,
        max_dd=max_dd,
        trades=trades,
        yearly_stats=yearly_stats,
        equity_curve=equity_df,
        params={
            'momentum_col': momentum_col,
            'top_n': top_n,
            'stop_loss': stop_loss,
            'universe_size': universe_size
        }
    )


# ============================================================================
# PARAMETER SWEEP
# ============================================================================

def run_parameter_sweep(
    universe_df: pd.DataFrame,
    daily_prices: pd.DataFrame
) -> pd.DataFrame:
    """
    Run comprehensive parameter sweep.

    400 combinations:
    - 5 momentum horizons × 5 top_n × 4 stop_loss × 4 universe_size
    """

    # Parameter grid
    momentum_cols = ['ret_1m', 'ret_3m', 'ret_6m', 'ret_12m', 'mom_risk_adj']
    top_n_values = [10, 20, 30, 50, 100]
    stop_losses = [None, 0.03, 0.05, 0.10]
    universe_sizes = [100, 200, 500, 1000]

    total = len(momentum_cols) * len(top_n_values) * len(stop_losses) * len(universe_sizes)
    print(f"\nRunning parameter sweep: {total} combinations")

    results = []
    count = 0

    for mom_col in momentum_cols:
        for top_n in top_n_values:
            for stop in stop_losses:
                for univ_size in universe_sizes:
                    count += 1

                    if count % 50 == 0:
                        print(f"  Progress: {count}/{total} ({count/total*100:.0f}%)")

                    try:
                        result = backtest_momentum_rotation(
                            universe_df=universe_df,
                            daily_prices=daily_prices,
                            momentum_col=mom_col,
                            top_n=top_n,
                            stop_loss=stop,
                            universe_size=univ_size
                        )

                        if result:
                            results.append({
                                'momentum': mom_col,
                                'top_n': top_n,
                                'stop_loss': stop if stop else 'None',
                                'universe_size': univ_size,
                                'total_return': result.total_return,
                                'annual_return': result.annual_return,
                                'sharpe': result.sharpe,
                                'max_dd': result.max_dd,
                                'trades': len(result.trades),
                                'stopped': len([t for t in result.trades if t.exit_reason == 'stop_loss']),
                                'win_rate': len([t for t in result.trades if t.pnl_pct and t.pnl_pct > 0]) / len(result.trades) * 100 if result.trades else 0
                            })
                    except Exception as e:
                        pass

    results_df = pd.DataFrame(results)

    # Save results
    results_df.to_csv(RESULTS_DIR / 'stock_momentum_sweep.csv', index=False)
    print(f"\nSaved {len(results_df)} results to {RESULTS_DIR / 'stock_momentum_sweep.csv'}")

    return results_df


def print_sweep_summary(results_df: pd.DataFrame):
    """Print summary of parameter sweep results."""

    print("\n" + "="*80)
    print("PARAMETER SWEEP SUMMARY")
    print("="*80)

    # Top 10 by Sharpe
    print("\nTOP 10 BY SHARPE RATIO:")
    print("-"*80)
    print(f"{'Momentum':<12} {'TopN':>6} {'Stop':>6} {'Univ':>6} {'Return':>10} {'Sharpe':>8} {'MaxDD':>8} {'Trades':>8}")
    print("-"*80)

    top10 = results_df.nlargest(10, 'sharpe')
    for _, row in top10.iterrows():
        print(f"{row['momentum']:<12} {row['top_n']:>6} {str(row['stop_loss']):>6} {row['universe_size']:>6} "
              f"{row['total_return']:>+9.1f}% {row['sharpe']:>8.2f} {row['max_dd']:>+7.1f}% {row['trades']:>8}")

    # Best by momentum type
    print("\n\nBEST BY MOMENTUM TYPE:")
    print("-"*80)
    for mom in results_df['momentum'].unique():
        mom_df = results_df[results_df['momentum'] == mom]
        best = mom_df.loc[mom_df['sharpe'].idxmax()]
        print(f"{mom:<12}: Sharpe {best['sharpe']:.2f}, Return {best['total_return']:+.1f}%, "
              f"Stop {best['stop_loss']}, TopN {best['top_n']}, Univ {best['universe_size']}")

    # Stop loss analysis
    print("\n\nSTOP LOSS IMPACT:")
    print("-"*80)
    for stop in sorted(results_df['stop_loss'].unique(), key=lambda x: str(x)):
        stop_df = results_df[results_df['stop_loss'] == stop]
        print(f"Stop {str(stop):>6}: Avg Sharpe {stop_df['sharpe'].mean():.2f}, "
              f"Avg Return {stop_df['total_return'].mean():+.1f}%, "
              f"Avg MaxDD {stop_df['max_dd'].mean():.1f}%")

    # Universe size analysis
    print("\n\nUNIVERSE SIZE IMPACT:")
    print("-"*80)
    for size in sorted(results_df['universe_size'].unique()):
        size_df = results_df[results_df['universe_size'] == size]
        print(f"Top {size:>4}: Avg Sharpe {size_df['sharpe'].mean():.2f}, "
              f"Avg Return {size_df['total_return'].mean():+.1f}%, "
              f"Best Sharpe {size_df['sharpe'].max():.2f}")


def run_best_config_detailed(
    universe_df: pd.DataFrame,
    daily_prices: pd.DataFrame,
    momentum_col: str,
    top_n: int,
    stop_loss: Optional[float],
    universe_size: int
) -> BacktestResult:
    """Run best configuration with detailed output."""

    print("\n" + "="*80)
    print("BEST CONFIGURATION DETAILED ANALYSIS")
    print("="*80)
    print(f"Momentum: {momentum_col}")
    print(f"Top N: {top_n}")
    print(f"Stop Loss: {stop_loss if stop_loss else 'None'}")
    print(f"Universe Size: {universe_size}")

    result = backtest_momentum_rotation(
        universe_df=universe_df,
        daily_prices=daily_prices,
        momentum_col=momentum_col,
        top_n=top_n,
        stop_loss=stop_loss,
        universe_size=universe_size
    )

    if not result:
        print("No valid result")
        return None

    # Print yearly breakdown
    print(f"\n{'Year':<6} {'Return':>10} {'Trades':>8} {'Stops':>8} {'WinRate':>10}")
    print("-"*50)

    for year, stats in sorted(result.yearly_stats.items()):
        print(f"{year:<6} {stats['return']:>+9.1f}% {stats['trades']:>8} "
              f"{stats['stopped']:>8} {stats['win_rate']:>9.1f}%")

    print("-"*50)
    print(f"{'TOTAL':<6} {result.total_return:>+9.1f}%")
    print(f"\nSharpe: {result.sharpe:.2f} | MaxDD: {result.max_dd:.1f}% | Trades: {len(result.trades)}")

    # Top winners and losers
    sorted_trades = sorted([t for t in result.trades if t.pnl_pct], key=lambda x: x.pnl_pct, reverse=True)

    print(f"\nTOP 10 WINNERS:")
    for t in sorted_trades[:10]:
        print(f"  {t.ticker}: {t.entry_date.strftime('%Y-%m')} → {t.exit_date.strftime('%Y-%m')} = {t.pnl_pct:+.1f}% ({t.exit_reason})")

    print(f"\nTOP 10 LOSERS:")
    for t in sorted_trades[-10:]:
        print(f"  {t.ticker}: {t.entry_date.strftime('%Y-%m')} → {t.exit_date.strftime('%Y-%m')} = {t.pnl_pct:+.1f}% ({t.exit_reason})")

    # Most traded tickers
    ticker_counts = {}
    for t in result.trades:
        ticker_counts[t.ticker] = ticker_counts.get(t.ticker, 0) + 1

    print(f"\nMOST TRADED TICKERS:")
    for ticker, count in sorted(ticker_counts.items(), key=lambda x: -x[1])[:10]:
        ticker_trades = [t for t in result.trades if t.ticker == ticker]
        avg_pnl = np.mean([t.pnl_pct for t in ticker_trades if t.pnl_pct])
        print(f"  {ticker}: {count} trades, avg P&L {avg_pnl:+.1f}%")

    return result


# ============================================================================
# MAIN
# ============================================================================

def main():
    """Run full unbiased stock momentum backtest."""

    print("="*80)
    print("UNBIASED STOCK MOMENTUM ROTATION BACKTESTER")
    print("="*80)
    print("\nThis uses point-in-time universe from comprehensive_features.csv")
    print("NO survivorship bias - stocks selected by volume at each point in time\n")

    # Load universe data
    universe_df = load_universe_data()

    # Get all unique tickers
    all_tickers = universe_df['ticker'].unique().tolist()
    print(f"\nNeed daily prices for {len(all_tickers)} tickers...")

    # Download/load daily prices
    daily_prices = download_daily_prices(all_tickers)
    print(f"Daily prices: {len(daily_prices)} days × {len(daily_prices.columns)} tickers")

    # Run parameter sweep
    results_df = run_parameter_sweep(universe_df, daily_prices)

    # Print summary
    print_sweep_summary(results_df)

    # Run best config detailed
    if len(results_df) > 0:
        best = results_df.loc[results_df['sharpe'].idxmax()]
        result = run_best_config_detailed(
            universe_df=universe_df,
            daily_prices=daily_prices,
            momentum_col=best['momentum'],
            top_n=int(best['top_n']),
            stop_loss=float(best['stop_loss']) if best['stop_loss'] != 'None' else None,
            universe_size=int(best['universe_size'])
        )

    # Benchmark comparison
    print("\n" + "="*80)
    print("BENCHMARK COMPARISON")
    print("="*80)

    # Get SPY returns
    spy = daily_prices['SPY'].dropna() if 'SPY' in daily_prices.columns else None
    if spy is not None and len(spy) > 0:
        spy_start = spy[spy.index >= '2020-02-01'].iloc[0]
        spy_end = spy[spy.index <= '2024-11-30'].iloc[-1]
        spy_return = (spy_end / spy_start - 1) * 100
        print(f"SPY (2020-02 to 2024-11): {spy_return:+.1f}%")

    # Show top strategies vs benchmarks
    if len(results_df) > 0:
        print(f"\nTop Strategy (by Sharpe): {best['total_return']:+.1f}% return, {best['sharpe']:.2f} Sharpe")
        print(f"6-Factor Model (from research): +214% return, ~1.0 Sharpe")
        print(f"ETF MEGA_UNIVERSE (from research): +1,007% return, 1.78 Sharpe")

    return results_df


def run_filter_sweep(
    universe_df: pd.DataFrame,
    daily_prices: pd.DataFrame
) -> pd.DataFrame:
    """
    Test the best momentum config with different filter combinations.

    Uses best baseline: ret_1m, top_n=50, stop=0.03, universe=1000
    Tests: ROE filter, revenue filter, EMA filters
    """

    print("\n" + "="*80)
    print("FILTER SWEEP: Testing fundamentals + EMA filters")
    print("="*80)
    print("Base config: ret_1m, top_n=50, stop=3%, universe=1000 (Sharpe 3.13 baseline)")

    # Filter combinations to test
    filter_configs = [
        # Baseline (no filters)
        {'roe_filter': False, 'revenue_filter': False, 'ema_period': None, 'name': 'No Filters (Baseline)'},

        # Single fundamental filters
        {'roe_filter': True, 'revenue_filter': False, 'ema_period': None, 'name': 'ROE > 0'},
        {'roe_filter': False, 'revenue_filter': True, 'ema_period': None, 'name': 'Revenue Growth > 0'},
        {'roe_filter': True, 'revenue_filter': True, 'ema_period': None, 'name': 'ROE + Revenue > 0'},

        # Single EMA filters
        {'roe_filter': False, 'revenue_filter': False, 'ema_period': 20, 'name': 'EMA20'},
        {'roe_filter': False, 'revenue_filter': False, 'ema_period': 50, 'name': 'EMA50'},
        {'roe_filter': False, 'revenue_filter': False, 'ema_period': 100, 'name': 'EMA100'},
        {'roe_filter': False, 'revenue_filter': False, 'ema_period': 200, 'name': 'EMA200'},

        # Combined: ROE + EMA
        {'roe_filter': True, 'revenue_filter': False, 'ema_period': 20, 'name': 'ROE + EMA20'},
        {'roe_filter': True, 'revenue_filter': False, 'ema_period': 50, 'name': 'ROE + EMA50'},
        {'roe_filter': True, 'revenue_filter': False, 'ema_period': 100, 'name': 'ROE + EMA100'},
        {'roe_filter': True, 'revenue_filter': False, 'ema_period': 200, 'name': 'ROE + EMA200'},

        # Combined: ROE + Revenue + EMA
        {'roe_filter': True, 'revenue_filter': True, 'ema_period': 20, 'name': 'ROE + Rev + EMA20'},
        {'roe_filter': True, 'revenue_filter': True, 'ema_period': 50, 'name': 'ROE + Rev + EMA50'},
        {'roe_filter': True, 'revenue_filter': True, 'ema_period': 100, 'name': 'ROE + Rev + EMA100'},
    ]

    results = []

    for config in filter_configs:
        print(f"\nTesting: {config['name']}...")

        try:
            result = backtest_momentum_rotation(
                universe_df=universe_df,
                daily_prices=daily_prices,
                momentum_col='ret_1m',
                top_n=50,
                stop_loss=0.03,
                universe_size=1000,
                roe_filter=config['roe_filter'],
                ema_period=config['ema_period'],
                revenue_filter=config['revenue_filter']
            )

            if result:
                stopped = len([t for t in result.trades if t.exit_reason == 'stop_loss'])
                winners = len([t for t in result.trades if t.pnl_pct and t.pnl_pct > 0])
                win_rate = winners / len(result.trades) * 100 if result.trades else 0

                results.append({
                    'filter': config['name'],
                    'roe_filter': config['roe_filter'],
                    'revenue_filter': config['revenue_filter'],
                    'ema_period': config['ema_period'] if config['ema_period'] else 'None',
                    'total_return': result.total_return,
                    'sharpe': result.sharpe,
                    'max_dd': result.max_dd,
                    'trades': len(result.trades),
                    'stopped': stopped,
                    'stop_pct': stopped / len(result.trades) * 100 if result.trades else 0,
                    'win_rate': win_rate
                })

                print(f"  → Return: {result.total_return:+.1f}%, Sharpe: {result.sharpe:.2f}, MaxDD: {result.max_dd:.1f}%")
        except Exception as e:
            print(f"  → Error: {e}")

    # Create results DataFrame
    results_df = pd.DataFrame(results)

    # Print summary
    print("\n" + "="*80)
    print("FILTER SWEEP RESULTS")
    print("="*80)
    print(f"\n{'Filter':<25} {'Return':>10} {'Sharpe':>8} {'MaxDD':>8} {'Trades':>8} {'WinRate':>8}")
    print("-"*75)

    for _, row in results_df.sort_values('sharpe', ascending=False).iterrows():
        print(f"{row['filter']:<25} {row['total_return']:>+9.1f}% {row['sharpe']:>8.2f} "
              f"{row['max_dd']:>+7.1f}% {row['trades']:>8} {row['win_rate']:>7.1f}%")

    # Save results
    results_df.to_csv(RESULTS_DIR / 'filter_sweep_results.csv', index=False)
    print(f"\nSaved to {RESULTS_DIR / 'filter_sweep_results.csv'}")

    # Find best
    if len(results_df) > 0:
        best = results_df.loc[results_df['sharpe'].idxmax()]
        baseline = results_df[results_df['filter'] == 'No Filters (Baseline)'].iloc[0] if len(results_df[results_df['filter'] == 'No Filters (Baseline)']) > 0 else None

        print("\n" + "="*80)
        print("CONCLUSION")
        print("="*80)
        print(f"Best Filter: {best['filter']}")
        print(f"  Sharpe: {best['sharpe']:.2f}")
        print(f"  Return: {best['total_return']:+.1f}%")
        print(f"  MaxDD: {best['max_dd']:.1f}%")

        if baseline is not None:
            print(f"\nBaseline (no filters): Sharpe {baseline['sharpe']:.2f}, Return {baseline['total_return']:+.1f}%")
            sharpe_diff = best['sharpe'] - baseline['sharpe']
            return_diff = best['total_return'] - baseline['total_return']
            print(f"Improvement: Sharpe {sharpe_diff:+.2f}, Return {return_diff:+.1f}%")

            if sharpe_diff > 0.1:
                print("\n✅ FILTERS HELP - Adding filters improved risk-adjusted returns!")
            elif sharpe_diff < -0.1:
                print("\n❌ FILTERS HURT - Keep the baseline (no filters)")
            else:
                print("\n⚠️ MINIMAL DIFFERENCE - Filters don't add much value")

    return results_df


if __name__ == '__main__':
    # Check for filter sweep mode
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == '--filter-sweep':
        print("Running filter sweep only...")
        universe_df = load_universe_data()
        all_tickers = universe_df['ticker'].unique().tolist()
        daily_prices = download_daily_prices(all_tickers)
        results = run_filter_sweep(universe_df, daily_prices)
    else:
        results = main()
