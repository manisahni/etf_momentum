# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Strategy Backtesting Laboratory - self-contained scripts for testing trading strategy hypotheses before live deployment. Part of the larger `/Users/nish_macbook/development/trading/` ecosystem.

## Running Backtests

```bash
# SPY Intraday (uses Alpaca parquet data)
python spy_15min_mean_reversion.py
python orb_fade_backtest.py
python vwap_band_backtest.py

# MES/ES Futures (requires IB SQLite database)
python mes_15min_mean_reversion.py      # Best: Sharpe 5.02, 28% annual
python mes_15min_enhanced.py            # Filter optimization testing
python kalman_filter_es_backtest.py
python hull_donchian_backtest.py
python es_timeframe_test.py

# Daily strategies (uses yfinance)
python sector_rotation_backtest.py
python spy_trend_vol_backtest.py
```

## Data Sources

| Source | Path | Format | Used By |
|--------|------|--------|---------|
| Alpaca SPY 1-min | `/Users/nish_macbook/development/trading/central_data/market_data/alpaca_spy_1min/` | Parquet by year | orb_fade, gap_orb, vwap_band, spy_15min_mean_reversion |
| IB ES 15-min | `/Users/nish_macbook/development/trading/emini-futures/data/emini_futures_15min.db` | SQLite | mes_15min_mean_reversion, kalman_filter, hull_donchian, es_timeframe |
| VIX Daily | `/Users/nish_macbook/development/trading/daily-optionslab/data/vix_daily_2020_2025.csv` | CSV | vwap_band (regime filter) |
| yfinance | Real-time API | Direct | sector_rotation, spy_trend_vol |

## Code Patterns

### Data Loading (Alpaca)
```python
df = pd.read_parquet(path, engine='fastparquet')
df['timestamp'] = pd.to_datetime(df['timestamp']).dt.tz_convert('America/New_York')
df = df.set_index('timestamp').between_time('09:30', '15:59')
```

### Trade Tracking
```python
@dataclass
class Trade:
    entry_time: object
    entry_price: float
    direction: str
    exit_time: Optional[object] = None
    pnl_pct: Optional[float] = None
```

### Risk Management (Required)
- Profit targets: 0.75-1.0% for intraday mean reversion
- Stop losses: 0.25-0.30% (tight, fixed - NOT trailing for mean reversion)
- Max holding: 8-10 bars to prevent overnight risk
- Regime filters: VIX thresholds critical for mean reversion

## Validation Requirements

Before declaring strategy "validated":
- 2+ years real data without errors
- Year-by-year consistency (works in each year separately)
- Sharpe > 1.2
- Max drawdown < 30%
- Trade count > 50

## Key Learnings

**Mean reversion**: Use fixed stops, not trailing (trailing kills performance by -35%)

**Regime filtering**: VIX < 22 filter improved VWAP strategy Sharpe from -0.12 to 0.85

**Asymmetric risk/reward**: Tight stop (0.25%) + larger target (1.0%) transforms weak edges into strong ones

**Futures vs ETFs**: MES/ES mean-revert faster than SPY - shorter hold (6 bars) and tighter stop (0.15%) work better

**Indicator comparison**: RSI beats Williams %R, Keltner, Stochastic, CCI, and Bollinger for mean reversion on both Sharpe AND $P&L

**Filter optimization**: Sometimes no filters is optimal - if baseline Sharpe is already high (>5), adding complexity reduces performance

**Dollar P&L calculation**: For futures, use `entry_price * (pnl_pct / 100) * point_value` to properly cap at stop/target levels

## Parent System Integration

- Strategy registry: `/Users/nish_macbook/development/trading/STRATEGY_REGISTRY.md`
- Data sources: `/Users/nish_macbook/development/trading/DATA_SOURCES.md`
- Master index: `/Users/nish_macbook/development/trading/README.md`

Update STRATEGY_REGISTRY.md when strategies are validated or rejected.

## Real Data Only

Per trading system policy - NO synthetic data generation:
```python
# FORBIDDEN
prices = np.random.normal(100, 2, 1000)

# REQUIRED
df = pd.read_parquet('/path/to/real_data.parquet')
```
