"""
SUPREME Universe Parameter Sweep - Test optimal settings on 220 ETFs
"""

import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime
from itertools import product
from expanded_universe_test import UNIVERSES, backtest_rotation
import warnings
warnings.filterwarnings('ignore')


def backtest_with_ema(prices, tickers, lookback, top_n, rebalance, stop_loss,
                       trailing_stop, ema_period=None):
    """Extended backtest with EMA filter option."""

    available = [t for t in tickers if t in prices.columns]
    if len(available) < top_n:
        return None, [], {}

    price_df = prices[available].copy()
    momentum = price_df.pct_change(lookback)

    # Apply EMA filter if specified
    if ema_period:
        ema = price_df.ewm(span=ema_period, adjust=False).mean()
        # Only consider stocks above their EMA
        above_ema = price_df > ema
    else:
        above_ema = pd.DataFrame(True, index=price_df.index, columns=price_df.columns)

    # Get rebalance dates
    if rebalance == 'W':
        rebal_dates = price_df.resample('W').last().index.tolist()
    elif rebalance == '2W':
        rebal_dates = price_df.resample('2W').last().index.tolist()
    else:  # 'M'
        rebal_dates = price_df.resample('M').last().index.tolist()

    rebal_dates = [d for d in rebal_dates if d in price_df.index]
    min_date = price_df.index[max(lookback, ema_period or 0)]
    rebal_dates = [d for d in rebal_dates if d >= min_date]

    if len(rebal_dates) < 2:
        return None, [], {}

    portfolio_value = 100.0
    positions = {}
    trades = []
    daily_values = []

    for i, date in enumerate(rebal_dates[:-1]):
        next_date = rebal_dates[i + 1]

        try:
            mom = momentum.loc[date].dropna()
            ema_filter = above_ema.loc[date] if ema_period else pd.Series(True, index=mom.index)
        except KeyError:
            continue

        # Filter by EMA if enabled
        if ema_period:
            mom = mom[ema_filter[mom.index] == True]

        if len(mom) < 1:
            daily_values.append({'date': date, 'value': portfolio_value})
            continue

        # Select top N by momentum
        top_tickers = mom.nlargest(min(top_n, len(mom))).index.tolist()

        # Close positions not in new selection
        for ticker in list(positions.keys()):
            if ticker not in top_tickers:
                pos = positions[ticker]
                exit_price = price_df.loc[date, ticker]
                pnl_pct = (exit_price / pos['entry_price'] - 1) * 100
                trades.append({
                    'entry_date': pos['entry_date'],
                    'exit_date': date,
                    'ticker': ticker,
                    'entry_price': pos['entry_price'],
                    'exit_price': exit_price,
                    'pnl_pct': pnl_pct,
                    'exit_reason': 'rebalance'
                })
                del positions[ticker]

        # Open new positions
        for ticker in top_tickers:
            if ticker not in positions:
                entry_price = price_df.loc[date, ticker]
                positions[ticker] = {
                    'entry_date': date,
                    'entry_price': entry_price,
                    'high_price': entry_price,
                    'stop_price': entry_price * (1 - stop_loss) if stop_loss else 0,
                }

        # Simulate period with stop checks
        period_dates = price_df.loc[date:next_date].index[1:]

        # Track position weights (set at rebalance)
        n_positions = len(positions)
        position_weight = 1.0 / n_positions if n_positions > 0 else 0
        for pos in positions.values():
            pos['weight'] = position_weight

        for day in period_dates:
            # PHASE 1: Check stops and collect stopped positions
            stopped_today = []
            if stop_loss:
                for ticker in list(positions.keys()):
                    pos = positions[ticker]
                    current_price = price_df.loc[day, ticker]

                    # Update trailing stop if enabled
                    if trailing_stop and current_price > pos['high_price']:
                        pos['high_price'] = current_price
                        pos['stop_price'] = current_price * (1 - stop_loss)

                    if current_price <= pos['stop_price']:
                        stopped_today.append((ticker, pos, current_price))

            # PHASE 2: Apply stop losses to portfolio (BUG FIX)
            for ticker, pos, exit_price in stopped_today:
                pnl_decimal = (exit_price / pos['entry_price'] - 1)
                portfolio_value *= (1 + pnl_decimal * pos['weight'])

                trades.append({
                    'entry_date': pos['entry_date'],
                    'exit_date': day,
                    'ticker': ticker,
                    'entry_price': pos['entry_price'],
                    'exit_price': exit_price,
                    'pnl_pct': pnl_decimal * 100,
                    'exit_reason': 'stop_loss'
                })
                del positions[ticker]

            # PHASE 3: Calculate daily return for remaining positions
            if positions:
                daily_return = 0
                for ticker, pos in positions.items():
                    prev_idx = price_df.index.get_loc(day) - 1
                    if prev_idx >= 0:
                        prev_price = price_df.iloc[prev_idx][ticker]
                        curr_price = price_df.loc[day, ticker]
                        daily_return += (curr_price / prev_price - 1) * pos['weight']

                portfolio_value *= (1 + daily_return)

            daily_values.append({'date': day, 'value': portfolio_value})

    # Create equity curve
    equity_df = pd.DataFrame(daily_values)
    if len(equity_df) == 0:
        return None, trades, {}

    equity_df['date'] = pd.to_datetime(equity_df['date'])
    equity_df = equity_df.set_index('date')

    return equity_df, trades, {}


def run_parameter_sweep():
    """Run parameter sweep on SUPREME universe."""

    print("=" * 80)
    print("SUPREME UNIVERSE PARAMETER SWEEP (220 ETFs)")
    print("=" * 80)

    # Get SUPREME universe
    supreme = UNIVERSES['SUPREME_UNIVERSE']
    print(f"\nUniverse: {len(supreme)} ETFs")

    # Download data
    print("\nDownloading price data (this may take a minute)...")
    all_tickers = list(set(supreme + ['SPY']))
    df = yf.download(all_tickers, start='2020-01-01',
                     end=datetime.now().strftime('%Y-%m-%d'),
                     auto_adjust=True, progress=False)

    if isinstance(df.columns, pd.MultiIndex):
        prices = df['Close']
    else:
        prices = df

    prices = prices.dropna(how='all').ffill()

    available = [t for t in supreme if t in prices.columns]
    print(f"Available: {len(available)}/{len(supreme)} tickers")

    # SPY benchmark
    spy_return = (prices['SPY'].iloc[-1] / prices['SPY'].iloc[0] - 1) * 100
    print(f"SPY Total Return: {spy_return:+.1f}%")

    # Parameter grid (focused)
    params = {
        'stop_loss': [None, 0.02, 0.03, 0.05, 0.07, 0.10],
        'trailing_stop': [True, False],
        'rebalance': ['W', 'M'],
        'top_n': [3, 5, 7, 10],
        'lookback': [21, 42, 63],
        'ema_period': [None, 20, 50, 100],
    }

    total_configs = 1
    for v in params.values():
        total_configs *= len(v)
    print(f"\nTotal configurations to test: {total_configs}")

    results = []
    tested = 0

    # Run sweep
    for stop_loss, trailing, rebal, top_n, lookback, ema in product(
        params['stop_loss'], params['trailing_stop'], params['rebalance'],
        params['top_n'], params['lookback'], params['ema_period']
    ):
        # Skip invalid combos
        if stop_loss is None and trailing:
            continue  # Can't trail without stop

        tested += 1
        if tested % 50 == 0:
            print(f"  Tested {tested}/{total_configs} configurations...")

        equity, trades, _ = backtest_with_ema(
            prices, available, lookback, top_n, rebal,
            stop_loss, trailing, ema
        )

        if equity is None or len(equity) < 50:
            continue

        # Calculate metrics
        total_return = (equity['value'].iloc[-1] / equity['value'].iloc[0] - 1) * 100
        returns = equity['value'].pct_change().dropna()
        sharpe = returns.mean() / returns.std() * np.sqrt(252) if returns.std() > 0 else 0
        max_dd = ((equity['value'] - equity['value'].cummax()) / equity['value'].cummax()).min() * 100

        stopped = len([t for t in trades if t['exit_reason'] == 'stop_loss'])

        results.append({
            'stop_loss': f"{stop_loss*100:.0f}%" if stop_loss else "None",
            'trailing': trailing,
            'rebalance': rebal,
            'top_n': top_n,
            'lookback': lookback,
            'ema': ema if ema else "None",
            'return': total_return,
            'sharpe': sharpe,
            'max_dd': max_dd,
            'trades': len(trades),
            'stopped': stopped,
            'alpha': total_return - spy_return
        })

    # Convert to DataFrame
    results_df = pd.DataFrame(results)

    # Save to CSV
    results_df.to_csv('results/supreme_parameter_sweep.csv', index=False)
    print(f"\nResults saved to results/supreme_parameter_sweep.csv")

    # Print summary
    print("\n" + "=" * 80)
    print("TOP 20 CONFIGURATIONS BY SHARPE RATIO")
    print("=" * 80)

    top20 = results_df.nlargest(20, 'sharpe')
    print(f"\n{'Stop':<6} {'Trail':<6} {'Rebal':<6} {'TopN':<5} {'Look':<5} {'EMA':<5} {'Return':>10} {'Sharpe':>8} {'MaxDD':>8} {'Alpha':>10}")
    print("-" * 80)

    for _, row in top20.iterrows():
        print(f"{row['stop_loss']:<6} {str(row['trailing']):<6} {row['rebalance']:<6} "
              f"{row['top_n']:<5} {row['lookback']:<5} {str(row['ema']):<5} "
              f"{row['return']:>+9.1f}% {row['sharpe']:>8.2f} {row['max_dd']:>+7.1f}% {row['alpha']:>+9.1f}%")

    # Stop loss analysis
    print("\n" + "=" * 80)
    print("STOP LOSS LEVEL ANALYSIS")
    print("=" * 80)

    stop_summary = results_df.groupby('stop_loss').agg({
        'sharpe': 'mean',
        'return': 'mean',
        'max_dd': 'mean'
    }).round(2)
    print(stop_summary)

    # Trailing vs Fixed
    print("\n" + "=" * 80)
    print("TRAILING VS FIXED STOP ANALYSIS")
    print("=" * 80)

    # Only compare where stop_loss is not None
    stop_configs = results_df[results_df['stop_loss'] != 'None']
    trail_summary = stop_configs.groupby('trailing').agg({
        'sharpe': 'mean',
        'return': 'mean',
        'max_dd': 'mean'
    }).round(2)
    print(trail_summary)

    # Rebalance frequency
    print("\n" + "=" * 80)
    print("REBALANCING FREQUENCY ANALYSIS")
    print("=" * 80)

    rebal_summary = results_df.groupby('rebalance').agg({
        'sharpe': 'mean',
        'return': 'mean',
        'max_dd': 'mean'
    }).round(2)
    print(rebal_summary)

    # EMA filter
    print("\n" + "=" * 80)
    print("EMA FILTER ANALYSIS")
    print("=" * 80)

    ema_summary = results_df.groupby('ema').agg({
        'sharpe': 'mean',
        'return': 'mean',
        'max_dd': 'mean'
    }).round(2)
    print(ema_summary)

    # Best single configuration
    best = results_df.loc[results_df['sharpe'].idxmax()]
    print("\n" + "=" * 80)
    print("BEST CONFIGURATION")
    print("=" * 80)
    print(f"""
Stop Loss:    {best['stop_loss']}
Trailing:     {best['trailing']}
Rebalance:    {best['rebalance']}
Top N:        {best['top_n']}
Lookback:     {best['lookback']} days
EMA Filter:   {best['ema']}

Performance:
  Return:     {best['return']:+.1f}%
  Sharpe:     {best['sharpe']:.2f}
  Max DD:     {best['max_dd']:.1f}%
  Alpha:      {best['alpha']:+.1f}% vs SPY
  Trades:     {best['trades']}
  Stopped:    {best['stopped']}
""")

    return results_df


if __name__ == '__main__':
    results = run_parameter_sweep()
