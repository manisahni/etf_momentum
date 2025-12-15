"""
Strategy: Kalman Filter Trend Following on ES Futures (15-min)

Inspired by successful Gaussian Filter (204% return, Sharpe 1.68)
Hypothesis: Kalman Filter provides optimal noise filtering for trend detection

Kalman Filter advantages:
1. Adapts to changing volatility (unlike fixed MA)
2. Predicts next value, not just smooths past
3. Optimal in statistical sense for noisy observations

Data: ES Futures 15-min bars from IB Gateway (same as Gaussian Filter)
"""

import pandas as pd
import numpy as np
import sqlite3
from dataclasses import dataclass
from typing import List, Optional
import warnings
warnings.filterwarnings('ignore')

# Database path
DB_PATH = '/Users/nish_macbook/development/trading/emini-futures/data/emini_futures_15min.db'
TABLE_NAME = 'es_15min_bars'

# Kalman Filter Parameters
PROCESS_NOISE = 0.01  # Q: How much the true signal changes
MEASUREMENT_NOISE = 1.0  # R: How noisy our observations are
LOOKBACK_FOR_VOL = 20  # For adaptive noise estimation


@dataclass
class Trade:
    entry_time: object
    entry_price: float
    direction: str
    exit_time: Optional[object] = None
    exit_price: Optional[float] = None
    exit_reason: Optional[str] = None
    pnl: Optional[float] = None
    pnl_pct: Optional[float] = None
    bars_held: int = 0


def load_es_data(db_path: str) -> pd.DataFrame:
    """Load ES futures 15-min data."""
    conn = sqlite3.connect(db_path)
    query = f"""
    SELECT timestamp, open, high, low, close, volume
    FROM {TABLE_NAME}
    ORDER BY timestamp
    """
    df = pd.read_sql_query(query, conn)
    conn.close()

    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df = df.set_index('timestamp')

    print(f"Loaded {len(df)} bars from {df.index[0]} to {df.index[-1]}")
    return df


def kalman_filter(prices: pd.Series, process_noise: float = 0.01,
                  measurement_noise: float = 1.0) -> pd.DataFrame:
    """
    1D Kalman Filter for price smoothing.

    State: [price]
    Observation: price

    Returns DataFrame with:
    - kalman_estimate: Filtered price
    - kalman_velocity: Estimated velocity (trend direction)
    - kalman_error: Estimation error covariance
    """
    n = len(prices)
    prices_array = prices.values

    # State estimates
    x_est = np.zeros(n)  # Price estimate
    p_est = np.zeros(n)  # Error covariance
    velocity = np.zeros(n)  # Price velocity

    # Initialize
    x_est[0] = prices_array[0]
    p_est[0] = 1.0

    # Kalman Filter loop
    for i in range(1, n):
        # Prediction step
        x_pred = x_est[i-1]  # State transition (price stays same)
        p_pred = p_est[i-1] + process_noise  # Add process noise

        # Update step
        K = p_pred / (p_pred + measurement_noise)  # Kalman gain
        x_est[i] = x_pred + K * (prices_array[i] - x_pred)
        p_est[i] = (1 - K) * p_pred

        # Calculate velocity (change in estimate)
        velocity[i] = x_est[i] - x_est[i-1]

    # Create result DataFrame
    result = pd.DataFrame({
        'kalman_estimate': x_est,
        'kalman_velocity': velocity,
        'kalman_error': p_est
    }, index=prices.index)

    return result


def adaptive_kalman_filter(prices: pd.Series, lookback: int = 20) -> pd.DataFrame:
    """
    Adaptive Kalman Filter that adjusts noise parameters based on recent volatility.
    """
    n = len(prices)
    prices_array = prices.values

    x_est = np.zeros(n)
    p_est = np.zeros(n)
    velocity = np.zeros(n)

    x_est[0] = prices_array[0]
    p_est[0] = 1.0

    for i in range(1, n):
        # Adaptive measurement noise based on recent volatility
        start_idx = max(0, i - lookback)
        recent_returns = np.diff(prices_array[start_idx:i+1])
        if len(recent_returns) > 1:
            r_adaptive = np.var(recent_returns) * 100  # Scale volatility
        else:
            r_adaptive = 1.0

        r_adaptive = max(0.1, min(10.0, r_adaptive))  # Clamp

        # Process noise proportional to volatility
        q_adaptive = r_adaptive * 0.01

        # Prediction
        x_pred = x_est[i-1]
        p_pred = p_est[i-1] + q_adaptive

        # Update
        K = p_pred / (p_pred + r_adaptive)
        x_est[i] = x_pred + K * (prices_array[i] - x_pred)
        p_est[i] = (1 - K) * p_pred

        velocity[i] = x_est[i] - x_est[i-1]

    return pd.DataFrame({
        'kalman_estimate': x_est,
        'kalman_velocity': velocity,
        'kalman_error': p_est
    }, index=prices.index)


def generate_signals(df: pd.DataFrame, kalman_df: pd.DataFrame,
                     velocity_threshold: float = 0.5,
                     use_volume_filter: bool = True) -> pd.DataFrame:
    """
    Generate trading signals based on Kalman velocity.

    Long: velocity crosses above threshold
    Short: velocity crosses below -threshold
    """
    df = df.copy()
    df['kalman'] = kalman_df['kalman_estimate']
    df['velocity'] = kalman_df['kalman_velocity']

    # Volume filter (similar to Gaussian Filter strategy)
    if use_volume_filter:
        df['volume_ma'] = df['volume'].rolling(20).mean()
        df['volume_ok'] = df['volume'] > df['volume_ma']
    else:
        df['volume_ok'] = True

    # Velocity-based signals
    df['velocity_prev'] = df['velocity'].shift(1)

    # Long when velocity crosses above threshold
    df['long_signal'] = (
        (df['velocity'] > velocity_threshold) &
        (df['velocity_prev'] <= velocity_threshold) &
        df['volume_ok']
    )

    # Short when velocity crosses below -threshold
    df['short_signal'] = (
        (df['velocity'] < -velocity_threshold) &
        (df['velocity_prev'] >= -velocity_threshold) &
        df['volume_ok']
    )

    return df


def backtest(df: pd.DataFrame, atr_mult_stop: float = 2.0,
             atr_mult_target: float = 3.0, atr_period: int = 14) -> List[Trade]:
    """
    Backtest the Kalman Filter strategy.

    Exit conditions:
    1. Opposite signal
    2. ATR-based stop loss
    3. ATR-based profit target
    """
    # Calculate ATR for stops
    high_low = df['high'] - df['low']
    high_close = abs(df['high'] - df['close'].shift())
    low_close = abs(df['low'] - df['close'].shift())
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    df['atr'] = tr.rolling(atr_period).mean()

    trades = []
    position = None  # 'long' or 'short' or None

    for i in range(atr_period, len(df)):
        bar = df.iloc[i]
        atr = bar['atr']

        # Check for exit if in position
        if position is not None:
            current_trade = trades[-1]
            bars_held = i - df.index.get_loc(current_trade.entry_time)

            if position == 'long':
                # Stop loss
                stop_price = current_trade.entry_price - atr_mult_stop * atr
                if bar['low'] <= stop_price:
                    current_trade.exit_time = bar.name
                    current_trade.exit_price = stop_price
                    current_trade.exit_reason = 'stop'
                    current_trade.pnl = current_trade.exit_price - current_trade.entry_price
                    current_trade.pnl_pct = current_trade.pnl / current_trade.entry_price * 100
                    current_trade.bars_held = bars_held
                    position = None
                    continue

                # Target
                target_price = current_trade.entry_price + atr_mult_target * atr
                if bar['high'] >= target_price:
                    current_trade.exit_time = bar.name
                    current_trade.exit_price = target_price
                    current_trade.exit_reason = 'target'
                    current_trade.pnl = current_trade.exit_price - current_trade.entry_price
                    current_trade.pnl_pct = current_trade.pnl / current_trade.entry_price * 100
                    current_trade.bars_held = bars_held
                    position = None
                    continue

                # Opposite signal
                if bar['short_signal']:
                    current_trade.exit_time = bar.name
                    current_trade.exit_price = bar['close']
                    current_trade.exit_reason = 'signal'
                    current_trade.pnl = current_trade.exit_price - current_trade.entry_price
                    current_trade.pnl_pct = current_trade.pnl / current_trade.entry_price * 100
                    current_trade.bars_held = bars_held
                    # Enter short immediately
                    trades.append(Trade(
                        entry_time=bar.name,
                        entry_price=bar['close'],
                        direction='short'
                    ))
                    position = 'short'
                    continue

            else:  # short position
                # Stop loss
                stop_price = current_trade.entry_price + atr_mult_stop * atr
                if bar['high'] >= stop_price:
                    current_trade.exit_time = bar.name
                    current_trade.exit_price = stop_price
                    current_trade.exit_reason = 'stop'
                    current_trade.pnl = current_trade.entry_price - current_trade.exit_price
                    current_trade.pnl_pct = current_trade.pnl / current_trade.entry_price * 100
                    current_trade.bars_held = bars_held
                    position = None
                    continue

                # Target
                target_price = current_trade.entry_price - atr_mult_target * atr
                if bar['low'] <= target_price:
                    current_trade.exit_time = bar.name
                    current_trade.exit_price = target_price
                    current_trade.exit_reason = 'target'
                    current_trade.pnl = current_trade.entry_price - current_trade.exit_price
                    current_trade.pnl_pct = current_trade.pnl / current_trade.entry_price * 100
                    current_trade.bars_held = bars_held
                    position = None
                    continue

                # Opposite signal
                if bar['long_signal']:
                    current_trade.exit_time = bar.name
                    current_trade.exit_price = bar['close']
                    current_trade.exit_reason = 'signal'
                    current_trade.pnl = current_trade.entry_price - current_trade.exit_price
                    current_trade.pnl_pct = current_trade.pnl / current_trade.entry_price * 100
                    current_trade.bars_held = bars_held
                    # Enter long immediately
                    trades.append(Trade(
                        entry_time=bar.name,
                        entry_price=bar['close'],
                        direction='long'
                    ))
                    position = 'long'
                    continue

        # Check for new entry if flat
        if position is None:
            if bar['long_signal']:
                trades.append(Trade(
                    entry_time=bar.name,
                    entry_price=bar['close'],
                    direction='long'
                ))
                position = 'long'
            elif bar['short_signal']:
                trades.append(Trade(
                    entry_time=bar.name,
                    entry_price=bar['close'],
                    direction='short'
                ))
                position = 'short'

    # Close any open position at end
    if position is not None and trades:
        current_trade = trades[-1]
        if current_trade.exit_time is None:
            current_trade.exit_time = df.index[-1]
            current_trade.exit_price = df.iloc[-1]['close']
            current_trade.exit_reason = 'eod'
            if current_trade.direction == 'long':
                current_trade.pnl = current_trade.exit_price - current_trade.entry_price
            else:
                current_trade.pnl = current_trade.entry_price - current_trade.exit_price
            current_trade.pnl_pct = current_trade.pnl / current_trade.entry_price * 100

    return trades


def calculate_metrics(trades: List[Trade], df: pd.DataFrame) -> dict:
    """Calculate performance metrics."""
    if not trades:
        return {}

    completed_trades = [t for t in trades if t.pnl_pct is not None]
    if not completed_trades:
        return {}

    pnl_pcts = np.array([t.pnl_pct for t in completed_trades])
    wins = pnl_pcts[pnl_pcts > 0]
    losses = pnl_pcts[pnl_pcts < 0]

    win_rate = len(wins) / len(pnl_pcts) * 100
    total_return = np.sum(pnl_pcts)

    # Approximate annual return (assume ~1700 15-min bars per year of RTH)
    bars = len(df)
    years = bars / 1700  # Rough estimate
    annual_return = total_return / years if years > 0 else 0

    # Sharpe ratio
    sharpe = (np.mean(pnl_pcts) / np.std(pnl_pcts)) * np.sqrt(252) if np.std(pnl_pcts) > 0 else 0

    # Max drawdown
    cumsum = np.cumsum(pnl_pcts)
    running_max = np.maximum.accumulate(cumsum)
    drawdown = cumsum - running_max
    max_dd = np.min(drawdown)

    # Profit factor
    gross_profit = np.sum(wins) if len(wins) > 0 else 0
    gross_loss = abs(np.sum(losses)) if len(losses) > 0 else 1
    pf = gross_profit / gross_loss if gross_loss > 0 else float('inf')

    # ES benchmark
    es_return = (df.iloc[-1]['close'] / df.iloc[0]['close'] - 1) * 100

    return {
        'trades': len(completed_trades),
        'win_rate': win_rate,
        'total_return': total_return,
        'annual_return': annual_return,
        'sharpe': sharpe,
        'max_dd': max_dd,
        'pf': pf,
        'avg_win': np.mean(wins) if len(wins) > 0 else 0,
        'avg_loss': np.mean(losses) if len(losses) > 0 else 0,
        'es_return': es_return,
        'years': years
    }


def print_results(metrics: dict, name: str = ""):
    """Print formatted results."""
    print(f"\n{'='*60}")
    print(f"RESULTS: {name}")
    print(f"{'='*60}")
    print(f"Trades: {metrics['trades']}")
    print(f"Win Rate: {metrics['win_rate']:.1f}%")
    print(f"Avg Win: +{metrics['avg_win']:.3f}%")
    print(f"Avg Loss: {metrics['avg_loss']:.3f}%")
    print(f"\nTotal Return: {metrics['total_return']:.1f}%")
    print(f"Annual Return: {metrics['annual_return']:.1f}%")
    print(f"Sharpe Ratio: {metrics['sharpe']:.2f}")
    print(f"Max Drawdown: {metrics['max_dd']:.1f}%")
    print(f"Profit Factor: {metrics['pf']:.2f}")
    print(f"\nES Buy & Hold: {metrics['es_return']:.1f}% ({metrics['years']:.1f} years)")
    print(f"Alpha vs ES: {metrics['total_return'] - metrics['es_return']:.1f}%")


def parameter_sweep(df: pd.DataFrame) -> List[dict]:
    """Test different parameter combinations."""
    print("\n" + "="*60)
    print("PARAMETER SWEEP")
    print("="*60)

    results = []

    for velocity_thresh in [0.25, 0.5, 0.75, 1.0, 1.5]:
        for atr_stop in [1.5, 2.0, 2.5, 3.0]:
            for atr_target in [2.0, 3.0, 4.0]:
                for use_adaptive in [False, True]:
                    for use_vol_filter in [False, True]:
                        # Calculate Kalman
                        if use_adaptive:
                            kalman_df = adaptive_kalman_filter(df['close'])
                        else:
                            kalman_df = kalman_filter(df['close'])

                        # Generate signals
                        signal_df = generate_signals(
                            df, kalman_df,
                            velocity_threshold=velocity_thresh,
                            use_volume_filter=use_vol_filter
                        )

                        # Backtest
                        trades = backtest(
                            signal_df,
                            atr_mult_stop=atr_stop,
                            atr_mult_target=atr_target
                        )

                        metrics = calculate_metrics(trades, df)
                        if not metrics or metrics['trades'] < 30:
                            continue

                        results.append({
                            'velocity': velocity_thresh,
                            'atr_stop': atr_stop,
                            'atr_target': atr_target,
                            'adaptive': use_adaptive,
                            'vol_filter': use_vol_filter,
                            **metrics
                        })

    # Sort by Sharpe
    results.sort(key=lambda x: -x['sharpe'])

    print(f"\n{'Vel':>5} {'Stop':>5} {'Tgt':>5} {'Adpt':>5} {'Vol':>5} {'Trades':>7} {'WR':>6} {'Return':>8} {'Sharpe':>7}")
    print("-"*65)

    for r in results[:15]:
        print(f"{r['velocity']:5.2f} {r['atr_stop']:5.1f} {r['atr_target']:5.1f} "
              f"{'Y' if r['adaptive'] else 'N':>5} {'Y' if r['vol_filter'] else 'N':>5} "
              f"{r['trades']:7d} {r['win_rate']:5.1f}% {r['total_return']:7.1f}% {r['sharpe']:6.2f}")

    return results


if __name__ == '__main__':
    print("="*60)
    print("KALMAN FILTER TREND FOLLOWING - ES FUTURES 15-MIN")
    print("="*60)

    # Load data
    df = load_es_data(DB_PATH)

    # Calculate Kalman Filter
    print("\nCalculating Kalman Filter...")
    kalman_df = kalman_filter(df['close'])

    # Generate signals with default parameters
    signal_df = generate_signals(df, kalman_df, velocity_threshold=0.5, use_volume_filter=True)

    # Count signals
    n_long = signal_df['long_signal'].sum()
    n_short = signal_df['short_signal'].sum()
    print(f"Long signals: {n_long}, Short signals: {n_short}")

    # Run backtest
    trades = backtest(signal_df)
    metrics = calculate_metrics(trades, df)

    print_results(metrics, "Kalman Filter + Volume Filter")

    # Test adaptive version
    print("\n" + "-"*60)
    print("Testing Adaptive Kalman Filter...")
    kalman_adaptive = adaptive_kalman_filter(df['close'])
    signal_adaptive = generate_signals(df, kalman_adaptive, velocity_threshold=0.5, use_volume_filter=True)
    trades_adaptive = backtest(signal_adaptive)
    metrics_adaptive = calculate_metrics(trades_adaptive, df)
    print_results(metrics_adaptive, "Adaptive Kalman + Volume Filter")

    # Parameter sweep
    sweep_results = parameter_sweep(df)

    if sweep_results:
        best = sweep_results[0]
        print(f"\nBest Configuration:")
        print(f"  Velocity threshold: {best['velocity']}")
        print(f"  ATR Stop: {best['atr_stop']}, ATR Target: {best['atr_target']}")
        print(f"  Adaptive: {best['adaptive']}, Volume Filter: {best['vol_filter']}")
        print(f"  Sharpe: {best['sharpe']:.2f}, Return: {best['total_return']:.1f}%")
