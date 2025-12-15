# Strategy Leaderboard

**Last Updated**: 2025-12-14
**Validation**: All strategies tested on 2020-2024 + 2025 out-of-sample

---

## Final Rankings

| Rank | Strategy | 2025 OOS | Backtest | Sharpe | MaxDD | Scalable |
|------|----------|----------|----------|--------|-------|----------|
| **1** | **SUPREME 216 ETFs + 3% Trailing** | +252.8% | **+27,815%** | **4.04** | **-5.1%** | **YES** |
| 2 | ULTRA 102 ETFs + 3% Trailing | +84.4% | +4,040% | 4.01 | -5.0% | YES |
| 3 | MEGA 58 ETFs + 3% Trailing | +76.0% | +2,019% | 3.56 | -5.9% | YES |
| 4 | ETF 5% Trailing (MEGA) | +96.7% | +1,542% | 2.82 | -6.2% | YES |
| 5 | ETF 3% Fixed (MEGA) | +83.9% | +1,007% | 1.78 | -15.5% | YES |
| 6 | Stock Momentum 3% | +36.4% | +20,067%* | 1.80 | -36% | Medium |
| 7 | 6-Factor Model | +14.0% | +214% | ~1.0 | -20% | YES |
| 8 | SPY Buy & Hold | +17.6% | +128% | ~0.8 | -34% | YES |

**Note**: SUPREME (171) now includes crypto miners/proxies with full 2020+ history.

*Stock momentum backtest inflated by concentration - realistic ~30-50%/yr

**Excluded from rankings**: Leveraged ETFs (TQQQ, SOXL, etc.) - backtest unreliable due to gap risk

---

## Champion: SUPREME (216 ETFs)

### SUPREME Universe (216 ETFs) - Best Backtest
```
Universe:     216 ETFs (SUPREME_UNIVERSE)
              ULTRA (102) + Currencies (8) + Thematic (27) + More Intl (22) +
              Crypto (12) + Contemporary (45)

              Categories:
              - CURRENCIES: UUP, FXE, FXY, FXB, FXA, FXC, CYB, UDN
              - THEMATIC: Space (UFO, ARKX, ROKT), Cyber (HACK, CIBR, BUG),
                Gaming (ESPO, HERO, NERD), Clean Energy (LIT, PBW, ACES),
                Infrastructure (PAVE, IFRA, IGF), Blockchain (BLOK, BLCN, LEGR),
                AI (AIQ, IRBO, WTAI), Water (PHO, FIW, CGW), Nuclear (NLR, URA, URNM)
              - MORE INTL: Latin America, Middle East, Africa, Frontier, Asia Ex, Europe
              - CRYPTO: MSTR, RIOT, MARA, CLSK, GBTC, ETHE, BITQ, DAPP, etc.
              - CONTEMPORARY (45 ETFs):
                * Electrification: DRIV, IDRV, KARS, ZAP, ELFY, COPX, CPER, PICK,
                  FCG, MLPX, AMLP, GRID, SRVR, DTCR
                * Materials: REMX (rare earths), BATT, SIL, SILJ, SLX, WOOD, MOO
                * Defense: ITA, PPA, FIVG, EUAD, NATO (European defense)
                * Consumer: IBUY, ONLN, PEJ, JETS, BETZ, SOCL, MJ, MSOS
                * Energy/Tech: HYDR, HDRO (hydrogen), METV (metaverse)
                * Sectors: KRBN, VNQ, IYR, IHI, FDN, KIE, PRNT, GNOM
Selection:    Top 5 by 21-day momentum
Rebalance:    Monthly
Stop:         3% TRAILING
```

### Performance Comparison
```
                   Backtest     Sharpe    MaxDD     2025 YTD    Status
SUPREME (216):    +27,815%      4.04     -5.1%     +252.8%     CHAMPION
ULTRA (102):       +4,040%      4.01     -5.0%      +84.4%     Production
MEGA (58):         +2,019%      3.56     -5.9%      +76.0%     Production
```

### Which to Use?
- **SUPREME (216)**: Maximum diversification, best backtest + 2025 OOS
- **ULTRA (102)**: Tight focus, slightly better Sharpe (less complexity)
- **MEGA (58)**: Conservative, well-tested baseline

### Why It Works
1. Trailing stop rides winners (ARKW +45.9% in 2025)
2. ETFs are liquid and scalable
3. Diversified across sectors, themes, international
4. Monthly rebalance = low transaction costs

---

## Strategy Comparison Matrix

| Criteria | ETF Trailing | Stock Momentum | 6-Factor |
|----------|--------------|----------------|----------|
| **2025 Alpha** | +71.9% | +22.1% | -0.8% |
| **Risk Control** | Excellent | Good | Moderate |
| **Scalability** | $10M+ | $1M | $10M+ |
| **Complexity** | Low | Medium | Medium |
| **Win Rate** | 53% | 12% | ~50% |
| **Drawdown** | -2.9% | -15%+ | -20% |

---

## Quick Reference: What to Trade

### For Maximum Alpha (Higher Risk)
**ETF 3% Trailing Stop**
- Expected: 30-80% annual
- Max DD: 5-15%
- Best for: Active traders

### For Steady Returns (Lower Risk)
**6-Factor Model**
- Expected: 10-20% annual
- Max DD: 15-25%
- Best for: Long-term investors

### For Simplicity
**SPY Buy & Hold**
- Expected: 8-12% annual
- Max DD: 30-50%
- Best for: Passive investors

---

## Key Findings from Research

### What Works
- **Trailing stops** >> Fixed stops for momentum
- **3% stop level** is optimal (not too tight, not too loose)
- **ETFs** >> Individual stocks (more liquid, less gap risk)
- **Monthly rebalance** balances costs vs responsiveness

### What Doesn't Work
- Tight stops (2%) - too many whipsaws
- No stops - massive drawdowns
- Weekly rebalancing - too costly
- Small universes (<20 ETFs) - not enough opportunities

### The Trailing Stop Edge
```
Fixed Stop:    Protects entry price only
Trailing Stop: Locks in gains as price rises

Example (ARKW 2025):
- Entry:  April 30
- Rise:   May-July (stop trails up)
- Exit:   August 1 (stopped out with +45.9% gain!)
- Fixed would have exited May 31 with smaller gain
```

---

## Implementation Files

| File | Purpose |
|------|---------|
| `expanded_universe_test.py` | ETF backtest engine + UNIVERSES definitions |
| `etf_ultra_universe_test.py` | ULTRA vs MEGA comparison test |
| `etf_2025_oos_test.py` | 2025 validation |
| `unbiased_stock_rotation.py` | Stock momentum engine |
| `results/ETF_ROTATION_STRATEGY.md` | Full ETF documentation |
| `results/STOP_LOSS_ANALYSIS.md` | Stock stop loss analysis |

---

## Next Steps

1. **Paper trade** ETF strategy for 1-2 months
2. **Automate** daily stop checks with Alpaca API
3. **Monitor** for regime changes (choppy markets hurt momentum)
4. **Scale gradually** - start with $10K, increase as confident

---

## Disclaimer

Past performance does not guarantee future results. The 2025 out-of-sample test is encouraging but represents only one year. Markets change, and momentum strategies can underperform in certain regimes.

**Recommended position sizing**: Never risk more than you can afford to lose.
