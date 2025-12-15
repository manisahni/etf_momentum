"""
Expanded Universe Testing - International + High-Flying Stocks

Test larger universes with momentum + 3% stop loss approach
"""

import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime
from dataclasses import dataclass
from typing import List, Dict, Optional, Tuple
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# EXPANDED UNIVERSES
# ============================================================================

# Core US Sectors (11)
US_SECTORS = ['XLK', 'XLV', 'XLF', 'XLY', 'XLP', 'XLE', 'XLI', 'XLB', 'XLU', 'XLRE', 'XLC']

# High-Flying Growth/Tech (20)
HIGH_FLYERS = [
    # Mega-cap tech
    'QQQ', 'ARKK', 'ARKG', 'ARKW', 'ARKF',
    # Semiconductors
    'SOXX', 'SMH', 'XSD',
    # Software/Cloud
    'IGV', 'WCLD', 'CLOU',
    # Biotech
    'XBI', 'IBB',
    # Clean Energy
    'TAN', 'ICLN', 'QCLN',
    # Robotics/AI
    'BOTZ', 'ROBO',
    # Fintech
    'FINX', 'IPAY'
]

# International Developed (15)
INTL_DEVELOPED = [
    'EFA',   # All developed ex-US
    'VEA',   # Vanguard developed
    'EWJ',   # Japan
    'EWG',   # Germany
    'EWU',   # UK
    'EWC',   # Canada
    'EWA',   # Australia
    'EWS',   # Singapore
    'EWH',   # Hong Kong
    'EWI',   # Italy
    'EWP',   # Spain
    'EWQ',   # France
    'EWL',   # Switzerland
    'EWN',   # Netherlands
    'EWD',   # Sweden
]

# Emerging Markets (12)
EMERGING = [
    'VWO',   # All emerging
    'EEM',   # iShares emerging
    'INDA',  # India
    'EWT',   # Taiwan
    'EWY',   # South Korea
    'EWZ',   # Brazil
    'EWW',   # Mexico
    'EIDO',  # Indonesia
    'THD',   # Thailand
    'VNM',   # Vietnam
    'EPOL',  # Poland
    'TUR',   # Turkey
]

# China (careful - has been a disaster)
CHINA = ['KWEB', 'FXI', 'MCHI', 'CQQQ', 'GXC']

# Factor ETFs (8)
FACTOR = ['MTUM', 'QUAL', 'VLUE', 'SIZE', 'USMV', 'VFMO', 'VFQY', 'VFVA']

# Commodities (8)
COMMODITIES = ['GLD', 'SLV', 'GDX', 'GDXJ', 'USO', 'UNG', 'DBA', 'DBC']

# Industries (8)
INDUSTRIES = ['XHB', 'XRT', 'XME', 'XOP', 'KRE', 'KBE', 'ITB', 'XAR']

# Income/Dividend (7)
INCOME = ['VYM', 'SCHD', 'DVY', 'HDV', 'SPHD', 'SPYD', 'VIG']

# Small/Mid Cap (8)
SIZE_ETFS = ['IWM', 'IWO', 'IWN', 'IJR', 'IJH', 'VB', 'VO', 'MDY']

# ============================================================================
# NEW CATEGORIES (Added 2025-12-14)
# ============================================================================

# Currencies (12)
CURRENCIES = ['UUP', 'FXE', 'FXY', 'FXB', 'FXA', 'FXC', 'CYB', 'UDN', 'FXF', 'FXS', 'BZF', 'CEW']
# FXF=Swiss Franc, FXS=Swedish Krona, BZF=Brazilian Real, CEW=Emerging Markets Currency

# Thematic - Space (3)
SPACE = ['UFO', 'ARKX', 'ROKT']

# Thematic - Cybersecurity (3)
CYBER = ['HACK', 'CIBR', 'BUG']

# Thematic - Gaming/Esports (3)
GAMING = ['ESPO', 'HERO', 'NERD']

# Thematic - Clean Energy/Lithium (3)
CLEAN_ENERGY = ['LIT', 'PBW', 'ACES']

# Thematic - Infrastructure (3)
INFRASTRUCTURE = ['PAVE', 'IFRA', 'IGF']

# Thematic - Blockchain (3)
BLOCKCHAIN = ['BLOK', 'BLCN', 'LEGR']

# Thematic - AI (3)
AI_THEME = ['AIQ', 'IRBO', 'WTAI']

# Thematic - Water (3)
WATER = ['PHO', 'FIW', 'CGW']

# Thematic - Nuclear/Uranium (3)
NUCLEAR = ['NLR', 'URA', 'URNM']

# More International - Latin America (5)
LATIN_AMERICA = ['ILF', 'ARGT', 'ECH', 'EPU', 'FLBR']

# More International - Middle East (3)
MIDDLE_EAST = ['KSA', 'UAE', 'GULF']

# More International - Africa (3)
AFRICA = ['AFK', 'EZA', 'NGE']

# More International - Frontier Markets (2)
FRONTIER = ['FM', 'FRN']

# More International - Asia Ex-Japan (3)
ASIA_EX = ['AAXJ', 'ASEA', 'EEMA']

# More International - Europe Single Country (6)
EUROPE_SINGLE = ['EWO', 'EFNL', 'GREK', 'EIS', 'NORW', 'EWK']

# Crypto ETFs with full history (excludes 2024 Bitcoin/Ethereum spot ETFs) (12)
CRYPTO = [
    # Miners/Proxies
    'MSTR', 'RIOT', 'MARA', 'CLSK',
    # Grayscale Trusts
    'GBTC', 'ETHE',
    # Crypto Industry ETFs
    'BITQ', 'DAPP', 'CRPT', 'SATO',
    # Bitcoin Futures
    'BITO', 'BTF'
]
# Note: BLOK already in BLOCKCHAIN category

# Leveraged ETFs (10) - 3x daily reset
# ⚠️ RESEARCH ONLY - Backtest unreliable due to:
#    - Overnight gaps blow through 3% stops (10-20% gaps common)
#    - All 10 ETFs crash together (no escape in 2022-style selloff)
#    - 98% stop rate = massive hidden transaction costs
#    - 2022 result implausible (+50% when all 3x crashed 60-80%)
LEVERAGED = [
    'TQQQ', 'SOXL', 'UPRO', 'SPXL', 'FAS',   # 3x bull
    'TECL', 'LABU', 'WEBL', 'FNGU', 'NAIL'
]
# DO NOT add to production universes - paper trade 3-6 months first

# ============================================================================
# CONTEMPORARY THEMES (Added 2025-12-14) - User requested
# ============================================================================

# Electric Vehicles & Electrification (3)
# Note: ZAP (Dec 2024), ELFY (Apr 2025) excluded - not enough history
ELECTRIC_VEHICLES = ['DRIV', 'IDRV', 'KARS']
# Future additions when history available: ZAP, ELFY

# Copper & Strategic Metals (3) - Critical for AI/electrification
COPPER_METALS = ['COPX', 'CPER', 'PICK']

# Natural Gas & Energy Infrastructure (3)
NATGAS_INFRA = ['FCG', 'MLPX', 'AMLP']

# Grid Infrastructure (1)
# GRID tracks smart grid, electrification infrastructure
GRID_INFRA = ['GRID']

# Data Center Infrastructure (2)
# Note: DTCR may be newer, verify history
DATA_CENTER = ['SRVR', 'DTCR']

# Rare Earths & Strategic Metals (1)
RARE_EARTHS = ['REMX']

# Battery Technology (1)
BATTERY = ['BATT']

# 5G & Telecom (1)
TELECOM_5G = ['FIVG']

# Aerospace & Defense (2) - supplements XAR
DEFENSE = ['ITA', 'PPA']
# Note: EUAD, NATO (European defense) launched Oct 2024 - add when history available

# Cannabis (2) - HIGH RISK, volatile sector
CANNABIS = ['MJ', 'MSOS']

# Carbon Credits (1)
CARBON = ['KRBN']

# Online Retail & E-commerce (2)
ONLINE_RETAIL = ['IBUY', 'ONLN']

# Travel & Leisure (2)
TRAVEL = ['PEJ', 'JETS']

# Silver Miners (2)
SILVER_MINERS = ['SIL', 'SILJ']

# Real Estate (2)
REAL_ESTATE = ['VNQ', 'IYR']

# Medical Devices (1)
MEDICAL_DEVICES = ['IHI']

# Internet (1)
INTERNET = ['FDN']

# Agribusiness (1)
AGRIBUSINESS = ['MOO']

# Steel (1)
STEEL = ['SLX']

# Timber & Forestry (1)
TIMBER = ['WOOD']

# Insurance (1)
INSURANCE = ['KIE']

# Sports Betting & iGaming (1)
SPORTS_BETTING = ['BETZ']

# Social Media (1)
SOCIAL_MEDIA = ['SOCL']

# 3D Printing (1)
PRINTING_3D = ['PRNT']

# Genomics (1) - supplements ARKG
GENOMICS = ['GNOM']

# Hydrogen (2) - 4+ years history
HYDROGEN = ['HYDR', 'HDRO']

# Metaverse (1) - 4+ years history
METAVERSE = ['METV']

# European Defense (2) - ~1 year history, add for tracking
# Note: Limited backtest reliability but important contemporary theme
EUROPE_DEFENSE = ['EUAD', 'NATO']

# Electrification - Newer (2) - <1 year history
# Note: ZAP (Dec 2024), ELFY (Apr 2025) - watch list only
ELECTRIFICATION_NEW = ['ZAP', 'ELFY']

# Individual High-Flyers (Popular momentum stocks)
MOMENTUM_STOCKS = [
    'NVDA', 'TSLA', 'AMD', 'META', 'GOOGL', 'AMZN', 'MSFT', 'AAPL',
    'NFLX', 'CRM', 'ADBE', 'NOW', 'SNOW', 'PLTR', 'COIN', 'SQ',
    'SHOP', 'MELI', 'SE', 'DDOG', 'NET', 'CRWD', 'ZS', 'PANW'
]

# Combined Universes
UNIVERSES = {
    # Individual categories
    'US_SECTORS': US_SECTORS,
    'HIGH_FLYERS': HIGH_FLYERS,
    'INTL_DEVELOPED': INTL_DEVELOPED,
    'EMERGING': EMERGING,
    'CHINA': CHINA,
    'FACTOR': FACTOR,
    'COMMODITIES': COMMODITIES,
    'INDUSTRIES': INDUSTRIES,
    'INCOME': INCOME,
    'SIZE_ETFS': SIZE_ETFS,
    'MOMENTUM_STOCKS': MOMENTUM_STOCKS,

    # Combined ETF universes
    'US_PLUS_GROWTH': US_SECTORS + HIGH_FLYERS,  # 31 ETFs
    'GLOBAL_DEVELOPED': US_SECTORS + INTL_DEVELOPED,  # 26 ETFs
    'GLOBAL_ALL': US_SECTORS + INTL_DEVELOPED + EMERGING,  # 38 ETFs
    'GROWTH_GLOBAL': HIGH_FLYERS + INTL_DEVELOPED + EMERGING,  # 47 ETFs
    'MEGA_UNIVERSE': US_SECTORS + HIGH_FLYERS + INTL_DEVELOPED + EMERGING,  # 58 ETFs

    # ULTRA_UNIVERSE: All ETFs (no individual stocks)
    'ULTRA_UNIVERSE': (
        US_SECTORS + HIGH_FLYERS + INTL_DEVELOPED + EMERGING +
        CHINA + FACTOR + COMMODITIES + INDUSTRIES + INCOME + SIZE_ETFS
    ),  # 102 ETFs

    # New categories
    'CURRENCIES': CURRENCIES,
    'SPACE': SPACE,
    'CYBER': CYBER,
    'GAMING': GAMING,
    'CLEAN_ENERGY': CLEAN_ENERGY,
    'INFRASTRUCTURE': INFRASTRUCTURE,
    'BLOCKCHAIN': BLOCKCHAIN,
    'AI_THEME': AI_THEME,
    'WATER': WATER,
    'NUCLEAR': NUCLEAR,
    'LATIN_AMERICA': LATIN_AMERICA,
    'MIDDLE_EAST': MIDDLE_EAST,
    'AFRICA': AFRICA,
    'FRONTIER': FRONTIER,
    'ASIA_EX': ASIA_EX,
    'EUROPE_SINGLE': EUROPE_SINGLE,
    'CRYPTO': CRYPTO,
    'LEVERAGED': LEVERAGED,

    # Contemporary themes (added 2025-12-14)
    'ELECTRIC_VEHICLES': ELECTRIC_VEHICLES,
    'COPPER_METALS': COPPER_METALS,
    'NATGAS_INFRA': NATGAS_INFRA,
    'GRID_INFRA': GRID_INFRA,
    'DATA_CENTER': DATA_CENTER,
    'RARE_EARTHS': RARE_EARTHS,
    'BATTERY': BATTERY,
    'TELECOM_5G': TELECOM_5G,
    'DEFENSE': DEFENSE,
    'CANNABIS': CANNABIS,
    'CARBON': CARBON,
    'ONLINE_RETAIL': ONLINE_RETAIL,
    'TRAVEL': TRAVEL,
    'SILVER_MINERS': SILVER_MINERS,
    'REAL_ESTATE': REAL_ESTATE,
    'MEDICAL_DEVICES': MEDICAL_DEVICES,
    'INTERNET': INTERNET,
    'AGRIBUSINESS': AGRIBUSINESS,
    'STEEL': STEEL,
    'TIMBER': TIMBER,
    'INSURANCE': INSURANCE,
    'SPORTS_BETTING': SPORTS_BETTING,
    'SOCIAL_MEDIA': SOCIAL_MEDIA,
    'PRINTING_3D': PRINTING_3D,
    'GENOMICS': GENOMICS,
    'HYDROGEN': HYDROGEN,
    'METAVERSE': METAVERSE,
    'EUROPE_DEFENSE': EUROPE_DEFENSE,
    'ELECTRIFICATION_NEW': ELECTRIFICATION_NEW,

    # SUPREME_UNIVERSE: Everything (220 ETFs)
    # Note: LEVERAGED excluded from SUPREME due to daily reset risk
    # Note: Includes some ETFs with 1-4 years history (HYDR, HDRO, METV, EUAD, NATO)
    'SUPREME_UNIVERSE': (
        # Original ULTRA (102)
        US_SECTORS + HIGH_FLYERS + INTL_DEVELOPED + EMERGING +
        CHINA + FACTOR + COMMODITIES + INDUSTRIES + INCOME + SIZE_ETFS +
        # Currencies (8)
        CURRENCIES +
        # Thematic (27)
        SPACE + CYBER + GAMING + CLEAN_ENERGY + INFRASTRUCTURE +
        BLOCKCHAIN + AI_THEME + WATER + NUCLEAR +
        # More International (22)
        LATIN_AMERICA + MIDDLE_EAST + AFRICA + FRONTIER + ASIA_EX + EUROPE_SINGLE +
        # Crypto (12)
        CRYPTO +
        # Contemporary Themes - Electrification & Infrastructure (12)
        ELECTRIC_VEHICLES + COPPER_METALS + NATGAS_INFRA + GRID_INFRA + DATA_CENTER +
        # Contemporary Themes - Materials & Resources (7)
        RARE_EARTHS + BATTERY + SILVER_MINERS + STEEL + TIMBER + AGRIBUSINESS +
        # Contemporary Themes - Defense & Telecom (5) - includes European defense
        DEFENSE + TELECOM_5G + EUROPE_DEFENSE +
        # Contemporary Themes - Consumer & Lifestyle (8)
        ONLINE_RETAIL + TRAVEL + SPORTS_BETTING + SOCIAL_MEDIA + CANNABIS +
        # Contemporary Themes - Sectors (7)
        CARBON + REAL_ESTATE + MEDICAL_DEVICES + INTERNET + INSURANCE + PRINTING_3D + GENOMICS +
        # Contemporary Themes - Energy & Tech (5)
        HYDROGEN + METAVERSE + ELECTRIFICATION_NEW
    ),  # 220 ETFs
    # Note: ELECTRIFICATION_NEW (ZAP, ELFY) have <1 year history - backtest uses available data

    # Stock universes (not used for ETF rotation)
    'STOCKS_ONLY': MOMENTUM_STOCKS,  # 24 individual stocks
    'STOCKS_PLUS_ETFS': MOMENTUM_STOCKS + HIGH_FLYERS,  # 44 mixed
}


def download_universe(tickers: List[str], start: str = '2020-01-01') -> Tuple[pd.DataFrame, List[str]]:
    """Download price data for universe."""
    all_tickers = list(set(tickers + ['SPY']))

    print(f"Downloading {len(all_tickers)} tickers...")
    df = yf.download(all_tickers, start=start, end=datetime.now().strftime('%Y-%m-%d'),
                     auto_adjust=True, progress=False)

    if isinstance(df.columns, pd.MultiIndex):
        prices = df['Close']
    else:
        prices = df[['Close']].rename(columns={'Close': all_tickers[0]})

    prices = prices.dropna(how='all').ffill()

    available = [t for t in tickers if t in prices.columns]
    print(f"Available: {len(available)}/{len(tickers)} tickers")
    print(f"Date range: {prices.index[0].date()} to {prices.index[-1].date()}")

    return prices, available


@dataclass
class Trade:
    entry_date: datetime
    exit_date: Optional[datetime]
    ticker: str
    entry_price: float
    exit_price: Optional[float]
    momentum: float
    stop_price: Optional[float]
    exit_reason: str
    pnl_pct: Optional[float]


def backtest_rotation(
    prices: pd.DataFrame,
    tickers: List[str],
    lookback: int = 21,
    top_n: int = 5,
    rebalance: str = 'M',
    stop_loss: float = 0.03,
    trailing_stop: bool = False  # NEW: trailing stop option
) -> Tuple[pd.DataFrame, List[Trade], Dict]:
    """Backtest momentum rotation with stop loss."""

    available = [t for t in tickers if t in prices.columns]
    if len(available) < top_n:
        return None, [], {}

    price_df = prices[available].copy()
    momentum = price_df.pct_change(lookback)

    # Get rebalance dates
    if rebalance == 'W':
        rebal_dates = price_df.resample('W').last().index.tolist()
    else:
        rebal_dates = price_df.resample('M').last().index.tolist()

    rebal_dates = [d for d in rebal_dates if d in price_df.index]
    min_date = price_df.index[lookback]
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
        except KeyError:
            continue

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

                trades.append(Trade(
                    entry_date=pos['entry_date'],
                    exit_date=date,
                    ticker=ticker,
                    entry_price=pos['entry_price'],
                    exit_price=exit_price,
                    momentum=pos['momentum'],
                    stop_price=pos.get('stop_price'),
                    exit_reason='rebalance',
                    pnl_pct=pnl_pct
                ))
                del positions[ticker]

        # Open new positions
        for ticker in top_tickers:
            if ticker not in positions:
                entry_price = price_df.loc[date, ticker]
                positions[ticker] = {
                    'entry_date': date,
                    'entry_price': entry_price,
                    'high_price': entry_price,  # Track high for trailing
                    'stop_price': entry_price * (1 - stop_loss),
                    'momentum': mom.get(ticker, 0),
                    'shares': 1.0 / len(top_tickers)
                }

        # Simulate period with stop checks
        period_dates = price_df.loc[date:next_date].index[1:]

        for day in period_dates:
            # Check stops
            for ticker in list(positions.keys()):
                pos = positions[ticker]
                current_price = price_df.loc[day, ticker]

                # Update trailing stop if enabled
                if trailing_stop and current_price > pos['high_price']:
                    pos['high_price'] = current_price
                    pos['stop_price'] = current_price * (1 - stop_loss)

                if current_price <= pos['stop_price']:
                    pnl_pct = (current_price / pos['entry_price'] - 1) * 100

                    trades.append(Trade(
                        entry_date=pos['entry_date'],
                        exit_date=day,
                        ticker=ticker,
                        entry_price=pos['entry_price'],
                        exit_price=current_price,
                        momentum=pos['momentum'],
                        stop_price=pos['stop_price'],
                        exit_reason='stop_loss',
                        pnl_pct=pnl_pct
                    ))
                    del positions[ticker]

            # Calculate portfolio value
            if positions:
                daily_return = 0
                weight = 1.0 / len(positions)
                for ticker, pos in positions.items():
                    prev_idx = price_df.index.get_loc(day) - 1
                    if prev_idx >= 0:
                        prev_price = price_df.iloc[prev_idx][ticker]
                        curr_price = price_df.loc[day, ticker]
                        daily_return += (curr_price / prev_price - 1) * weight

                portfolio_value *= (1 + daily_return)

            daily_values.append({'date': day, 'value': portfolio_value})

    # Close remaining positions
    if positions:
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
    yearly_stats = {}
    for year in range(2020, 2026):
        year_eq = equity_df[equity_df.index.year == year]
        year_trades = [t for t in trades if t.entry_date.year == year]

        if len(year_eq) < 2:
            continue

        year_return = (year_eq['value'].iloc[-1] / year_eq['value'].iloc[0] - 1) * 100

        spy_year = prices['SPY'][prices['SPY'].index.year == year]
        spy_return = (spy_year.iloc[-1] / spy_year.iloc[0] - 1) * 100 if len(spy_year) > 1 else 0

        cummax = year_eq['value'].cummax()
        max_dd = ((year_eq['value'] - cummax) / cummax).min() * 100

        stopped = [t for t in year_trades if t.exit_reason == 'stop_loss']

        yearly_stats[year] = {
            'return': year_return,
            'spy_return': spy_return,
            'alpha': year_return - spy_return,
            'max_dd': max_dd,
            'trades': len(year_trades),
            'stopped': len(stopped)
        }

    return equity_df, trades, yearly_stats


def run_universe_test(universe_name: str, tickers: List[str], params: dict):
    """Run test for a single universe."""
    print(f"\n{'='*70}")
    print(f"Testing: {universe_name} ({len(tickers)} tickers)")
    print(f"{'='*70}")

    prices, available = download_universe(tickers)

    if len(available) < params['top_n']:
        print(f"Skipping - only {len(available)} available")
        return None

    equity, trades, yearly = backtest_rotation(
        prices, available,
        lookback=params['lookback'],
        top_n=min(params['top_n'], len(available)),
        rebalance=params['rebalance'],
        stop_loss=params['stop_loss']
    )

    if equity is None or len(equity) == 0:
        print("No valid trades")
        return None

    # Calculate metrics
    total_return = (equity['value'].iloc[-1] / 100 - 1) * 100
    returns = equity['value'].pct_change().dropna()
    sharpe = returns.mean() / returns.std() * np.sqrt(252) if returns.std() > 0 else 0
    max_dd = ((equity['value'] - equity['value'].cummax()) / equity['value'].cummax()).min() * 100

    stopped = len([t for t in trades if t.exit_reason == 'stop_loss'])

    # Print yearly breakdown
    print(f"\n{'Year':<6} {'Return':>10} {'SPY':>10} {'Alpha':>10} {'MaxDD':>10} {'Trades':>8} {'Stops':>8}")
    print("-" * 70)

    for year, s in sorted(yearly.items()):
        print(f"{year:<6} {s['return']:>+10.1f}% {s['spy_return']:>+9.1f}% {s['alpha']:>+9.1f}% "
              f"{s['max_dd']:>+9.1f}% {s['trades']:>8} {s['stopped']:>8}")

    print("-" * 70)
    print(f"{'TOTAL':<6} {total_return:>+10.1f}%")
    print(f"\nSharpe: {sharpe:.2f} | Max DD: {max_dd:.1f}% | Trades: {len(trades)} | Stopped: {stopped}")

    # Top winners
    sorted_trades = sorted([t for t in trades if t.pnl_pct], key=lambda x: x.pnl_pct, reverse=True)
    print(f"\nTop 5 Winners:")
    for t in sorted_trades[:5]:
        print(f"  {t.ticker}: {t.entry_date.strftime('%Y-%m')} → {t.exit_date.strftime('%Y-%m')} = {t.pnl_pct:+.1f}%")

    return {
        'universe': universe_name,
        'tickers': len(available),
        'total_return': total_return,
        'sharpe': sharpe,
        'max_dd': max_dd,
        'trades': len(trades),
        'stopped': stopped,
        'stop_pct': stopped / len(trades) * 100 if trades else 0
    }


def main():
    """Test all expanded universes."""

    # Best params from previous research
    params = {
        'lookback': 21,
        'top_n': 5,
        'rebalance': 'M',
        'stop_loss': 0.03
    }

    print("="*70)
    print("EXPANDED UNIVERSE TESTING")
    print("="*70)
    print(f"Parameters: Lookback={params['lookback']}d, Top={params['top_n']}, "
          f"Rebalance={params['rebalance']}, Stop={params['stop_loss']*100}%")

    results = []

    # Test each universe
    for name, tickers in UNIVERSES.items():
        result = run_universe_test(name, tickers, params)
        if result:
            results.append(result)

    # Summary table
    print("\n" + "="*70)
    print("UNIVERSE COMPARISON SUMMARY")
    print("="*70)
    print(f"{'Universe':<20} {'Tickers':>8} {'Return':>12} {'Sharpe':>8} {'MaxDD':>10} {'Stop%':>8}")
    print("-"*70)

    for r in sorted(results, key=lambda x: -x['sharpe']):
        print(f"{r['universe']:<20} {r['tickers']:>8} {r['total_return']:>+11.1f}% "
              f"{r['sharpe']:>8.2f} {r['max_dd']:>+9.1f}% {r['stop_pct']:>7.0f}%")

    # Top 5 by Sharpe
    print("\n" + "="*70)
    print("TOP 5 BY RISK-ADJUSTED RETURN (SHARPE)")
    print("="*70)
    for i, r in enumerate(sorted(results, key=lambda x: -x['sharpe'])[:5], 1):
        print(f"{i}. {r['universe']}: Sharpe {r['sharpe']:.2f}, Return {r['total_return']:+.1f}%, MaxDD {r['max_dd']:.1f}%")

    # Top 5 by Total Return
    print("\n" + "="*70)
    print("TOP 5 BY TOTAL RETURN")
    print("="*70)
    for i, r in enumerate(sorted(results, key=lambda x: -x['total_return'])[:5], 1):
        print(f"{i}. {r['universe']}: Return {r['total_return']:+.1f}%, Sharpe {r['sharpe']:.2f}, MaxDD {r['max_dd']:.1f}%")

    return results


if __name__ == '__main__':
    results = main()
