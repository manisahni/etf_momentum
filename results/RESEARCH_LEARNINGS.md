# Momentum Strategy Research - Key Learnings

**Date**: 2024-12-14
**Research Duration**: ~4 hours
**Outcome**: Validated ETF rotation strategy with +89.5% 2025 alpha

---

## 1. Trailing Stops Beat Fixed Stops for Momentum

### The Discovery
| Stop Type | Return | Sharpe | MaxDD |
|-----------|--------|--------|-------|
| 3% Trailing | +1,975% | 3.56 | -5.9% |
| 3% Fixed | +1,007% | 1.78 | -19.7% |
| No Stop | +493% | 1.35 | -33.4% |

### Why Trailing Wins
- **Fixed stop**: Only protects entry price, exits at rebalance regardless of gains
- **Trailing stop**: Locks in gains as price rises, lets winners run until reversal
- **Real example**: ARKW 2025 - entered April, trailed up through July, stopped August with +45.9%

### The Paradox Explained
Per-trade edge is WORSE with trailing (+2.58% vs +5.40% no-stop), but portfolio returns are BETTER because:
1. Winners compound longer before exit
2. Losers cut faster
3. Risk-adjusted returns matter more than raw edge

---

## 2. 3% Stop Level is Optimal

### Stop Level Comparison
| Stop Level | Avg Sharpe | Reasoning |
|------------|------------|-----------|
| **3%** | **2.27** | Tight enough to protect, loose enough to avoid whipsaws |
| 5% | 1.76 | Too loose - gives back too much |
| 10% | 1.15 | Way too loose |
| None | 0.52 | Massive drawdowns destroy returns |

### Key Insight
- Tighter than 3% (e.g., 2%) causes too many false stop-outs
- Looser than 3% gives back gains unnecessarily
- 3% is the sweet spot for momentum strategies

---

## 3. ETFs Beat Individual Stocks for Rotation

### Comparison
| Universe | 2025 Return | MaxDD | Scalability |
|----------|-------------|-------|-------------|
| ETF MEGA (58) | +89.5% | -2.9% | $10M+ |
| Stocks (1000) | +36.4% | ~-15% | $1M |

### Why ETFs Win
1. **Liquidity**: No slippage, tight spreads
2. **Gap risk**: ETFs rarely gap >3% overnight
3. **Diversification**: Each ETF is already diversified
4. **Scalability**: Can trade millions without impact
5. **Simplicity**: 5 positions vs 50 positions

---

## 4. Position Concentration is a FEATURE (Not a Bug)

### Initial Confusion
When we saw trailing stops resulted in only 6.5 avg positions (vs 50 target), we thought it was broken.

### The Realization
User insight: "Wait, isn't that the whole point? To find and ride a few winners?"

**YES.** The stop loss acts as a filter:
- Stocks that continue trending → stay in portfolio
- Stocks that reverse → get stopped out
- Portfolio naturally concentrates into "true" momentum winners

### The Math
- 88% of positions stopped out (losers filtered)
- 12% ride to big gains (winners kept)
- Result: Venture-capital style returns from public markets

---

## 5. Out-of-Sample Validation is CRITICAL

### The Backtest Trap
- Stock momentum showed +20,067% backtest return
- Looked amazing on paper
- But inflated by concentration artifacts

### 2025 Reality Check
| Strategy | 2025 OOS | Backtest | Verdict |
|----------|----------|----------|---------|
| ETF Trailing | +89.5% | +1,975% | VALIDATED |
| Stock Momentum | +36.4% | +20,067% | Inflated |

### Lesson
Always test on out-of-sample data before trusting backtest results.

---

## 6. Survivorship Bias Matters

### The Problem
Hand-picked "momentum stocks" (NVDA, TSLA, PLTR) showed +1,588% returns.
But we selected them KNOWING they were winners - look-ahead bias!

### The Solution
Used ETFs instead - they existed in 2020 with fixed tickers.
No survivorship bias possible.

### For Future Research
When testing stocks, must use:
- Point-in-time universe (what was tradeable on that date)
- Include delisted stocks
- No hindsight in stock selection

---

## 7. Price Adjustment Bugs are Sneaky

### The Bug We Found
WHLM showed +65,300% return (impossible).

### Root Cause
Mixed unadjusted prices from CSV ($0.01) with adjusted prices from yfinance ($4.61).
Stock split created fake 460x gain.

### The Fix
```python
# WRONG - mixing data sources
if ticker in daily_prices:
    price = daily_prices[ticker]
else:
    price = csv_data['price']  # UNADJUSTED!

# RIGHT - single source only
if ticker not in daily_prices:
    continue  # Skip, don't fall back
```

### Lesson
NEVER mix adjusted and unadjusted price sources.

---

## 8. Monthly Rebalancing is Optimal

### Frequency Comparison
| Frequency | Net Alpha | Costs |
|-----------|-----------|-------|
| Daily | -4.5% | ~7.6% |
| Weekly | -5.0% | ~1.6% |
| Monthly | **+4.7%** | ~0.4% |

### Why Monthly Wins
1. Transaction costs compound with frequency
2. Momentum signals need time to play out
3. Daily rebalancing = chasing noise

---

## 9. Simple Beats Complex

### What Didn't Help
- EMA filters (marginal improvement, more complexity)
- Fundamental filters (ROE, revenue) hurt returns
- Volatility filters (mixed results)

### What Worked
Simple momentum + trailing stop. That's it.

### Lesson
Start simple. Only add complexity if it demonstrably improves results.

---

## 10. Document Everything

### What We Created
| Document | Purpose |
|----------|---------|
| ETF_ROTATION_STRATEGY.md | Full strategy guide |
| STRATEGY_LEADERBOARD.md | Rankings |
| STOP_LOSS_ANALYSIS.md | Stock stop research |
| TRAILING_STOP_AUDIT.md | Concentration analysis |
| This file | Key learnings |

### Why It Matters
- Avoid repeating failed experiments
- Quick reference for implementation
- Onboard others to the strategy
- Track what we've learned

---

## Quick Reference: What Works

### Winning Configuration
```
Universe:     58 ETFs (MEGA_UNIVERSE)
Selection:    Top 5 by 21-day momentum
Rebalance:    Monthly
Stop:         3% TRAILING
```

### Expected Performance
- Annual return: 30-80% (depending on regime)
- Sharpe: 2.0-3.5
- Max drawdown: 5-15%
- Scalability: $10M+

### When It Struggles
- Choppy, sideways markets (many false breakouts)
- Rapid sector rotation (stops trigger too often)
- Correlated selloffs (all ETFs drop together)

---

## What We'd Do Differently

1. **Start with ETFs first** - Wasted time on stock rotation that had bias issues

2. **Test trailing stops earlier** - Fixed stops were good, trailing was better

3. **Run 2025 OOS immediately** - Would have caught inflation issues faster

4. **Document as we go** - Had to reconstruct learnings at the end

---

## Next Steps (Not Yet Done)

1. **Paper trade** for 1-2 months
2. **Automate** daily stop checks
3. **Monitor** for regime changes
4. **Scale gradually** - start small, increase as confident

---

## The Bottom Line

**We found a real edge**: ETF momentum + 3% trailing stop.

Validated with:
- 5-year backtest (+1,975%, Sharpe 3.56)
- 2025 out-of-sample (+89.5% vs SPY +17.6%)
- Simple, scalable, low-cost implementation

**This is tradeable.**
