"""
Backtest Pine Script Strategies on SPY Data
============================================
1. Pivot Point SuperTrend Strategy [ES Optimized]
2. Hull Suite Strategy

Using real SPY data from the trading database.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from dataclasses import dataclass
from typing import Literal, Optional
import warnings
warnings.filterwarnings('ignore')


# =============================================================================
# DATA LOADING
# =============================================================================

def load_spy_daily() -> pd.DataFrame:
    """Load SPY daily data from leveraged ETF dataset (2006-2025)"""
    path = Path('/Users/nish_macbook/development/trading/0dte-intraday/strategies/leveraged_etf/data/etf_data/SPY.parquet')
    df = pd.read_parquet(path)

    # Flatten multi-level columns
    df.columns = [col[0].lower() if isinstance(col, tuple) else col.lower() for col in df.columns]
    df = df.reset_index()
    df.columns = ['date', 'close', 'high', 'low', 'open', 'volume']
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values('date').reset_index(drop=True)

    return df


# =============================================================================
# PIVOT POINT SUPERTREND STRATEGY
# =============================================================================

def pivot_supertrend_strategy(
    df: pd.DataFrame,
    prd: int = 2,           # Pivot Point Period
    factor: float = 3.0,    # ATR Factor
    atr_period: int = 23,   # ATR Period
    use_volume_filter: bool = True,
    volume_ma_length: int = 20,
) -> pd.DataFrame:
    """
    Pivot Point SuperTrend Strategy [ES Optimized]

    Pine Script logic translated to Python:
    - Detects pivot highs and lows
    - Creates center line from pivot points
    - ATR-based bands around center
    - Trend detection with trailing stop
    """
    df = df.copy()

    # Pivot Point Detection
    df['pivot_high'] = df['high'].rolling(window=2*prd+1, center=True).apply(
        lambda x: x[prd] if x[prd] == x.max() else np.nan, raw=True
    )
    df['pivot_low'] = df['low'].rolling(window=2*prd+1, center=True).apply(
        lambda x: x[prd] if x[prd] == x.min() else np.nan, raw=True
    )

    # Center Line Calculation (EMA-like smoothing of pivot points)
    center = np.full(len(df), np.nan)
    for i in range(len(df)):
        ph = df['pivot_high'].iloc[i]
        pl = df['pivot_low'].iloc[i]
        lastpp = ph if not np.isnan(ph) else (pl if not np.isnan(pl) else np.nan)

        if not np.isnan(lastpp):
            if np.isnan(center[i-1]) if i > 0 else True:
                center[i] = lastpp
            else:
                center[i] = (center[i-1] * 2 + lastpp) / 3
        elif i > 0:
            center[i] = center[i-1]

    df['center'] = center

    # ATR Calculation
    df['tr'] = np.maximum(
        df['high'] - df['low'],
        np.maximum(
            abs(df['high'] - df['close'].shift(1)),
            abs(df['low'] - df['close'].shift(1))
        )
    )
    df['atr'] = df['tr'].rolling(window=atr_period).mean()

    # ATR Bands
    df['up'] = df['center'] - (factor * df['atr'])
    df['dn'] = df['center'] + (factor * df['atr'])

    # Trend Detection with Trailing Stop Logic
    trend = np.ones(len(df))
    tup = np.full(len(df), np.nan)
    tdown = np.full(len(df), np.nan)

    for i in range(1, len(df)):
        up_val = df['up'].iloc[i]
        dn_val = df['dn'].iloc[i]
        close_prev = df['close'].iloc[i-1]
        close_curr = df['close'].iloc[i]

        # TUp calculation
        if np.isnan(tup[i-1]):
            tup[i] = up_val
        elif close_prev > tup[i-1]:
            tup[i] = max(up_val, tup[i-1]) if not np.isnan(up_val) else tup[i-1]
        else:
            tup[i] = up_val if not np.isnan(up_val) else tup[i-1]

        # TDown calculation
        if np.isnan(tdown[i-1]):
            tdown[i] = dn_val
        elif close_prev < tdown[i-1]:
            tdown[i] = min(dn_val, tdown[i-1]) if not np.isnan(dn_val) else tdown[i-1]
        else:
            tdown[i] = dn_val if not np.isnan(dn_val) else tdown[i-1]

        # Trend determination
        if close_curr > tdown[i-1] if not np.isnan(tdown[i-1]) else False:
            trend[i] = 1
        elif close_curr < tup[i-1] if not np.isnan(tup[i-1]) else False:
            trend[i] = -1
        else:
            trend[i] = trend[i-1]

    df['trend'] = trend
    df['tup'] = tup
    df['tdown'] = tdown
    df['trailing_stop'] = np.where(df['trend'] == 1, df['tup'], df['tdown'])

    # Volume Filter
    if use_volume_filter:
        df['vol_ma'] = df['volume'].rolling(window=volume_ma_length).mean()
        df['volume_ok'] = df['volume'] > df['vol_ma']
    else:
        df['volume_ok'] = True

    # Signals (trend change)
    df['buy_signal'] = (df['trend'] == 1) & (df['trend'].shift(1) == -1) & df['volume_ok']
    df['sell_signal'] = (df['trend'] == -1) & (df['trend'].shift(1) == 1) & df['volume_ok']

    return df


# =============================================================================
# HULL SUITE STRATEGY
# =============================================================================

def wma(series: pd.Series, period: int) -> pd.Series:
    """Weighted Moving Average"""
    weights = np.arange(1, period + 1)
    return series.rolling(period).apply(lambda x: np.dot(x, weights) / weights.sum(), raw=True)


def ema(series: pd.Series, period: int) -> pd.Series:
    """Exponential Moving Average"""
    return series.ewm(span=period, adjust=False).mean()


def hma(series: pd.Series, length: int) -> pd.Series:
    """Hull Moving Average"""
    half_length = int(length / 2)
    sqrt_length = int(round(np.sqrt(length)))
    return wma(2 * wma(series, half_length) - wma(series, length), sqrt_length)


def ehma(series: pd.Series, length: int) -> pd.Series:
    """Exponential Hull Moving Average"""
    half_length = int(length / 2)
    sqrt_length = int(round(np.sqrt(length)))
    return ema(2 * ema(series, half_length) - ema(series, length), sqrt_length)


def thma(series: pd.Series, length: int) -> pd.Series:
    """Triple Hull Moving Average"""
    third_length = int(length / 3)
    half_length = int(length / 2)
    return wma(wma(series, third_length) * 3 - wma(series, half_length) - wma(series, length), length)


def hull_suite_strategy(
    df: pd.DataFrame,
    length: int = 55,
    mode: Literal['Hma', 'Ehma', 'Thma'] = 'Hma',
    src: str = 'close',
) -> pd.DataFrame:
    """
    Hull Suite Strategy

    Pine Script logic translated to Python:
    - Calculates Hull MA variation (HMA, EHMA, or THMA)
    - Buy when HULL[0] > HULL[2] (uptrend)
    - Sell when HULL[0] < HULL[2] (downtrend)
    """
    df = df.copy()

    # Select Hull variation
    if mode == 'Hma':
        df['hull'] = hma(df[src], length)
    elif mode == 'Ehma':
        df['hull'] = ehma(df[src], length)
    elif mode == 'Thma':
        df['hull'] = thma(df[src], length // 2)

    # Trend detection: compare current hull to 2 bars ago
    df['hull_trend'] = np.where(df['hull'] > df['hull'].shift(2), 1, -1)

    # Signals (trend change)
    df['buy_signal'] = (df['hull_trend'] == 1) & (df['hull_trend'].shift(1) == -1)
    df['sell_signal'] = (df['hull_trend'] == -1) & (df['hull_trend'].shift(1) == 1)

    return df


# =============================================================================
# BACKTESTING ENGINE
# =============================================================================

@dataclass
class BacktestResult:
    """Backtest results container"""
    strategy_name: str
    total_return: float
    cagr: float
    sharpe_ratio: float
    max_drawdown: float
    win_rate: float
    num_trades: int
    avg_trade_return: float
    profit_factor: float
    start_date: str
    end_date: str
    trades: pd.DataFrame


def run_backtest(
    df: pd.DataFrame,
    strategy_name: str,
    initial_capital: float = 100000,
    commission_pct: float = 0.0,  # 0 for futures/index comparison
    allow_short: bool = True,
) -> BacktestResult:
    """
    Run backtest with buy/sell signals

    Assumes:
    - buy_signal and sell_signal columns exist
    - Long on buy_signal, Short on sell_signal (if allowed)
    - Always in position (reversal system)
    """
    df = df.copy()
    df = df.dropna(subset=['close']).reset_index(drop=True)

    # Initialize
    position = 0  # 1 = long, -1 = short, 0 = flat
    entry_price = 0
    equity = initial_capital
    equity_curve = [initial_capital]
    trades = []

    for i in range(1, len(df)):
        current_price = df['close'].iloc[i]
        buy = df['buy_signal'].iloc[i] if 'buy_signal' in df.columns else False
        sell = df['sell_signal'].iloc[i] if 'sell_signal' in df.columns else False

        # Close existing position and open new one on signal
        if buy and position != 1:
            # Close short if exists
            if position == -1:
                pnl = (entry_price - current_price) / entry_price
                pnl_after_comm = pnl - commission_pct * 2
                equity *= (1 + pnl_after_comm)
                trades.append({
                    'entry_date': entry_date,
                    'exit_date': df['date'].iloc[i],
                    'direction': 'short',
                    'entry_price': entry_price,
                    'exit_price': current_price,
                    'return': pnl_after_comm,
                })
            # Go long
            position = 1
            entry_price = current_price
            entry_date = df['date'].iloc[i]

        elif sell and position != -1:
            # Close long if exists
            if position == 1:
                pnl = (current_price - entry_price) / entry_price
                pnl_after_comm = pnl - commission_pct * 2
                equity *= (1 + pnl_after_comm)
                trades.append({
                    'entry_date': entry_date,
                    'exit_date': df['date'].iloc[i],
                    'direction': 'long',
                    'entry_price': entry_price,
                    'exit_price': current_price,
                    'return': pnl_after_comm,
                })
            # Go short (if allowed)
            if allow_short:
                position = -1
                entry_price = current_price
                entry_date = df['date'].iloc[i]
            else:
                position = 0

        equity_curve.append(equity)

    # Close final position
    if position != 0:
        current_price = df['close'].iloc[-1]
        if position == 1:
            pnl = (current_price - entry_price) / entry_price
        else:
            pnl = (entry_price - current_price) / entry_price
        pnl_after_comm = pnl - commission_pct * 2
        equity *= (1 + pnl_after_comm)
        trades.append({
            'entry_date': entry_date,
            'exit_date': df['date'].iloc[-1],
            'direction': 'long' if position == 1 else 'short',
            'entry_price': entry_price,
            'exit_price': current_price,
            'return': pnl_after_comm,
        })

    # Calculate metrics
    trades_df = pd.DataFrame(trades) if trades else pd.DataFrame()
    equity_series = pd.Series(equity_curve)

    # Total return
    total_return = (equity - initial_capital) / initial_capital

    # CAGR
    years = (df['date'].iloc[-1] - df['date'].iloc[0]).days / 365.25
    cagr = (equity / initial_capital) ** (1 / years) - 1 if years > 0 else 0

    # Sharpe Ratio (daily returns)
    equity_series_full = pd.Series(equity_curve)
    daily_returns = equity_series_full.pct_change().dropna()
    sharpe = daily_returns.mean() / daily_returns.std() * np.sqrt(252) if daily_returns.std() > 0 else 0

    # Max Drawdown
    rolling_max = equity_series_full.cummax()
    drawdown = (equity_series_full - rolling_max) / rolling_max
    max_dd = drawdown.min()

    # Trade statistics
    if len(trades_df) > 0:
        win_rate = (trades_df['return'] > 0).mean()
        avg_trade = trades_df['return'].mean()
        winners = trades_df[trades_df['return'] > 0]['return'].sum()
        losers = abs(trades_df[trades_df['return'] < 0]['return'].sum())
        profit_factor = winners / losers if losers > 0 else float('inf')
    else:
        win_rate = 0
        avg_trade = 0
        profit_factor = 0

    return BacktestResult(
        strategy_name=strategy_name,
        total_return=total_return,
        cagr=cagr,
        sharpe_ratio=sharpe,
        max_drawdown=max_dd,
        win_rate=win_rate,
        num_trades=len(trades_df),
        avg_trade_return=avg_trade,
        profit_factor=profit_factor,
        start_date=str(df['date'].iloc[0].date()),
        end_date=str(df['date'].iloc[-1].date()),
        trades=trades_df,
    )


def print_results(result: BacktestResult):
    """Print backtest results in a formatted way"""
    print(f"\n{'='*60}")
    print(f"  {result.strategy_name}")
    print(f"{'='*60}")
    print(f"  Period: {result.start_date} to {result.end_date}")
    print(f"  {'─'*56}")
    print(f"  Total Return:     {result.total_return:>10.1%}")
    print(f"  CAGR:             {result.cagr:>10.1%}")
    print(f"  Sharpe Ratio:     {result.sharpe_ratio:>10.2f}")
    print(f"  Max Drawdown:     {result.max_drawdown:>10.1%}")
    print(f"  {'─'*56}")
    print(f"  Number of Trades: {result.num_trades:>10}")
    print(f"  Win Rate:         {result.win_rate:>10.1%}")
    print(f"  Avg Trade Return: {result.avg_trade_return:>10.2%}")
    print(f"  Profit Factor:    {result.profit_factor:>10.2f}")
    print(f"{'='*60}\n")


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    print("\n" + "="*60)
    print("  PINE SCRIPT STRATEGIES BACKTEST")
    print("  Using Real SPY Data (2006-2025)")
    print("="*60)

    # Load data
    print("\nLoading SPY daily data...")
    df = load_spy_daily()
    print(f"Loaded {len(df):,} bars from {df['date'].min().date()} to {df['date'].max().date()}")

    # Filter to recent period for comparison (2016-2025 like Pine Script default)
    df_recent = df[df['date'] >= '2016-01-01'].copy().reset_index(drop=True)
    print(f"Filtered to {len(df_recent):,} bars (2016-2025)")

    results = []

    # ==========================================================================
    # Strategy 1: Pivot Point SuperTrend
    # ==========================================================================
    print("\n" + "-"*60)
    print("Running Pivot Point SuperTrend Strategy...")
    print("Parameters: prd=2, factor=3.0, atr_period=23, volume_filter=True")
    print("-"*60)

    df_pivot = pivot_supertrend_strategy(
        df_recent,
        prd=2,
        factor=3.0,
        atr_period=23,
        use_volume_filter=True,
        volume_ma_length=20,
    )

    result_pivot = run_backtest(
        df_pivot,
        strategy_name="Pivot Point SuperTrend [ES Optimized]",
        initial_capital=100000,
        commission_pct=0.001,  # 0.1% round trip
        allow_short=True,
    )
    results.append(result_pivot)
    print_results(result_pivot)

    # Also test long-only
    result_pivot_long = run_backtest(
        df_pivot,
        strategy_name="Pivot Point SuperTrend [Long Only]",
        initial_capital=100000,
        commission_pct=0.001,
        allow_short=False,
    )
    results.append(result_pivot_long)
    print_results(result_pivot_long)

    # ==========================================================================
    # Strategy 2: Hull Suite
    # ==========================================================================
    print("\n" + "-"*60)
    print("Running Hull Suite Strategy...")
    print("Parameters: length=55, mode=HMA")
    print("-"*60)

    df_hull = hull_suite_strategy(
        df_recent,
        length=55,
        mode='Hma',
    )

    result_hull = run_backtest(
        df_hull,
        strategy_name="Hull Suite Strategy [HMA-55]",
        initial_capital=100000,
        commission_pct=0.001,
        allow_short=True,
    )
    results.append(result_hull)
    print_results(result_hull)

    # Long only variant
    result_hull_long = run_backtest(
        df_hull,
        strategy_name="Hull Suite Strategy [Long Only]",
        initial_capital=100000,
        commission_pct=0.001,
        allow_short=False,
    )
    results.append(result_hull_long)
    print_results(result_hull_long)

    # ==========================================================================
    # Hull Suite - Longer Length (180 for S/R)
    # ==========================================================================
    print("\n" + "-"*60)
    print("Running Hull Suite Strategy (Length=180 for S/R)...")
    print("-"*60)

    df_hull_180 = hull_suite_strategy(
        df_recent,
        length=180,
        mode='Hma',
    )

    result_hull_180 = run_backtest(
        df_hull_180,
        strategy_name="Hull Suite Strategy [HMA-180]",
        initial_capital=100000,
        commission_pct=0.001,
        allow_short=True,
    )
    results.append(result_hull_180)
    print_results(result_hull_180)

    # ==========================================================================
    # Buy & Hold Benchmark
    # ==========================================================================
    print("\n" + "-"*60)
    print("Running Buy & Hold Benchmark...")
    print("-"*60)

    df_bh = df_recent.copy()
    df_bh['buy_signal'] = False
    df_bh.loc[0, 'buy_signal'] = True
    df_bh['sell_signal'] = False

    result_bh = run_backtest(
        df_bh,
        strategy_name="Buy & Hold SPY (Benchmark)",
        initial_capital=100000,
        commission_pct=0,
        allow_short=False,
    )
    results.append(result_bh)
    print_results(result_bh)

    # ==========================================================================
    # Summary Comparison
    # ==========================================================================
    print("\n" + "="*80)
    print("  STRATEGY COMPARISON SUMMARY")
    print("="*80)
    print(f"{'Strategy':<40} {'Return':>10} {'CAGR':>8} {'Sharpe':>8} {'MaxDD':>8} {'Trades':>8}")
    print("-"*80)

    for r in results:
        name = r.strategy_name[:38]
        print(f"{name:<40} {r.total_return:>9.1%} {r.cagr:>7.1%} {r.sharpe_ratio:>7.2f} {r.max_drawdown:>7.1%} {r.num_trades:>8}")

    print("="*80)

    # Save detailed results
    output_dir = Path('/Users/nish_macbook/development/trading/strategy_backtests')
    output_dir.mkdir(exist_ok=True)

    # Save trade logs
    for r in results:
        if len(r.trades) > 0:
            safe_name = r.strategy_name.replace(' ', '_').replace('[', '').replace(']', '').replace('&', 'and')
            r.trades.to_csv(output_dir / f"{safe_name}_trades.csv", index=False)

    print(f"\nTrade logs saved to: {output_dir}")
