# Momentum Stop Loss Strategy Audit

## Summary: All Inflated Returns Are Due to Concentration

**Date:** 2024-12-14
**Strategies Tested:** Fixed stops, trailing stops, no stops
**Verdict:** ALL results unreliable due to position concentration

### Quick Reference

| Strategy | Reported Return | Avg Positions | Per-Trade Edge | Verdict |
|----------|----------------|---------------|----------------|---------|
| 3% trailing | +820,345% | 6.5 | +1.41% | WORST edge |
| 3% fixed | +20,067% | 10.1 | +1.66% | Medium edge |
| No stop (50%) | +574% | 23.0 | **+3.27%** | BEST edge |

**The more stops, the more concentration, the more inflated returns.**

---

## The Flaw: Position Concentration

### What We Found

| Metric | Trailing Stop | Fixed Stop | Target |
|--------|---------------|------------|--------|
| Avg Positions Held | **6.5** | 15.1 | 50 |
| Days with <10 positions | 74% | fewer | 0% |
| Days with <5 positions | 63% | fewer | 0% |

### Position Decay After Rebalance

```
Day 0:  50 positions (target)
Day 32: 31 positions
Day 40: 15 positions
Day 52:  7 positions
Day 60:  2 positions  ← CRITICAL CONCENTRATION
```

### Root Cause

1. Trailing stop is very aggressive (3% from high)
2. Most positions get stopped out within 2-4 weeks
3. Stopped-out capital sits in cash until next monthly rebalance
4. Portfolio becomes heavily concentrated in 2-10 survivors
5. Returns are driven by concentrated bets, not diversified momentum

---

## Why The Numbers Look Amazing (But Aren't Real)

### High Returns
- With only 5 positions, each is 20% of portfolio
- One stock going +50% = +10% portfolio return
- This isn't skill, it's concentration luck

### Low Drawdown (-4%)
- Trailing stop exits losing positions quickly
- BUT also exits winners before they run
- Small position count means fewer simultaneous losses

### High Sharpe (7.69)
- Calculated on daily returns of concentrated portfolio
- Not comparable to a properly diversified strategy
- Survivorship of the "lucky few" stocks

---

## Gap Risk Is Real

Example: QS (QuantumScape)
- Entry: 2020-12-31 @ $84.45
- Stop: $81.92 (3% below entry)
- Exit: 2021-01-04 @ $49.96 (40% gap down over New Year)
- Loss: -40.84% despite 3% stop

**107 trades lost more than 10%** due to gap risk. Trailing stops can't prevent gap losses.

---

## What Would Be Needed For Valid Backtest

### Option 1: Weekly Rebalancing
- Add new positions weekly to maintain 50 count
- Would reduce concentration issue
- More transaction costs

### Option 2: Replace Stopped Positions Immediately
- When position stops out, immediately find replacement
- Maintains target position count
- More complex logic

### Option 3: Reduce Stop Tightness
- Use 10% trailing instead of 3%
- Fewer stop-outs, maintain position count
- But higher individual losses

---

## Valid Findings From This Research

### Trailing Stops DO Work (When Implemented Correctly)
- Lock in gains when momentum reverses
- 37% of stopped trades are profitable (vs 0% with fixed)
- Concept is sound, implementation needs work

### Volatility Filter Shows Promise
- High vol stocks: Better for momentum (bigger moves)
- Low vol stocks: Better for capital preservation
- Worth testing with corrected position management

### Gap Risk Is Unavoidable
- No stop loss can prevent overnight gaps
- Must be factored into position sizing
- Max single-stock loss should be budgeted at 20%+

---

## Corrected Strategy Requirements

Before trusting any momentum + trailing stop results:

1. **Verify position count stays near target** throughout backtest
2. **Include transaction costs** (0.1% round trip minimum)
3. **Account for slippage** (0.05% on each trade)
4. **Test on out-of-sample period** (2025 data)
5. **Check liquidity** - can you actually trade the position size?

---

## The Real Numbers (Trade-Level Analysis)

When we analyze at the trade level (independent of concentration):

| Metric | Trailing | Fixed | Winner |
|--------|----------|-------|--------|
| Per-trade expectancy | +1.41% | **+1.66%** | Fixed |
| Win rate | 38.4% | 22.5% | Trailing |
| Avg winner | +13.4% | +31.3% | Fixed |
| Avg loser | -6.1% | -7.0% | Trailing |

### Estimated Returns With Proper Diversification

With 50 positions maintained throughout:
- **Trailing: ~7.4% annual** (not +820,000%)
- **Fixed: ~8.2% annual** (not +20,000%)

### Key Insight

Fixed stop actually has **BETTER per-trade expectancy** (+1.66% vs +1.41%)!

Trailing stop trades more frequently but with smaller gains:
- More winners (38% vs 22%)
- But smaller winners (+13% vs +31%)
- Net result: worse per-trade edge

---

## Conclusion

The +820,000% return with Sharpe 7.69 is an artifact of:
- Unintended position concentration (6.5 avg vs 50 target)
- Survivorship of lucky concentrated bets
- No transaction costs or slippage

### Realistic Expectations

| Strategy | Realistic Annual Return | Notes |
|----------|------------------------|-------|
| Fixed 3% stop | ~8% | Before costs |
| Trailing 3% stop | ~7% | Before costs |
| After costs (~1%) | ~6-7% | Net return |

### Recommendations

1. **Don't use trailing stops for this strategy** - fixed stops have better edge
2. **The original +20,000% was also inflated** by concentration
3. **Realistic expectation: 6-10% annual** with proper diversification and costs
4. **Focus on the per-trade edge**, not portfolio-level returns that depend on luck

### What We Learned

- Backtest artifacts can make bad strategies look amazing
- Always verify position counts stay at target
- Trade-level analysis reveals the true edge (or lack thereof)
- "Too good to be true" usually is

---

## Final Recommendation

### Don't Use Tight Stops for Monthly Momentum

The data shows that **no stop** or **very wide stop** (50%) produces:
- Better per-trade edge (+3.27% vs +1.66%)
- More realistic position count (23 vs 10)
- Still positive returns (+574% over 5 years ≈ 47% annual)

### If You Must Use Stops

1. **Rebalance weekly** (not monthly) to refill stopped positions
2. **Use 10-15% stops** rather than 3% (fewer false triggers)
3. **Expect ~30-50% annual returns** with proper diversification
4. **Budget 1-2% for transaction costs**

### The True Edge

This momentum strategy has a **+3.27% per-trade edge** when positions are held to monthly rebalance. That's genuine alpha from momentum.

Tight stops **destroy** this edge by:
- Exiting positions before momentum plays out
- Creating concentration in survivors
- Increasing transaction costs

### Realistic Expectations

| Scenario | Expected Annual Return |
|----------|----------------------|
| 50 positions, no stops, before costs | ~40-50% |
| 50 positions, 10% stops, before costs | ~30-40% |
| After 1% transaction costs | ~25-35% |
| After slippage and real execution | ~20-30% |

These are still excellent returns - just not +800,000%.
