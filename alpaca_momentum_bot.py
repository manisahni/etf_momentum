"""
ETF Momentum Rotation Bot - Alpaca Paper Trading

Strategy: SUPREME 220 ETFs + 3% Trailing Stop
- Monthly rebalance: Top 10 by 21-day momentum
- Daily: Check trailing stops

Usage:
    # Set credentials
    export APCA_API_KEY_ID="your_key"
    export APCA_API_SECRET_KEY="your_secret"

    # Run daily (cron or manually)
    python alpaca_momentum_bot.py
"""

import os
import json
import logging
from datetime import datetime, timedelta
from typing import List, Dict, Optional
import pandas as pd
import numpy as np

# Alpaca
from alpaca_trade_api import REST
from alpaca_trade_api.rest import APIError

# Local universe
from expanded_universe_test import UNIVERSES

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/momentum_bot.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# =============================================================================
# CONFIGURATION
# =============================================================================

CONFIG = {
    'universe': 'SUPREME_UNIVERSE',  # 220 ETFs
    'lookback': 21,                  # 21-day momentum
    'top_n': 10,                     # Hold top 10
    'stop_loss': 0.03,               # 3% trailing stop
    'rebalance_day': 1,              # First trading day of month
    'paper': True,                   # Paper trading mode
    'redistribute': True,            # Redistribute capital to survivors after stop-out
                                     # True: Higher returns (~517k%), higher DD (-13.5%)
                                     # False: Lower returns (~4.5k%), lower DD (-8.1%)
}

# State file to track positions and stops
STATE_FILE = 'data/bot_state.json'


class MomentumBot:
    """ETF Momentum Rotation Bot with Trailing Stops."""

    def __init__(self, paper: bool = True):
        """Initialize bot with Alpaca connection."""
        self.paper = paper
        base_url = 'https://paper-api.alpaca.markets' if paper else 'https://api.alpaca.markets'

        self.api = REST(
            key_id=os.environ.get('APCA_API_KEY_ID'),
            secret_key=os.environ.get('APCA_API_SECRET_KEY'),
            base_url=base_url
        )

        # Get universe
        self.universe = UNIVERSES[CONFIG['universe']]
        logger.info(f"Initialized bot with {len(self.universe)} ETFs (paper={paper})")

        # Load state
        self.state = self._load_state()

    def _load_state(self) -> Dict:
        """Load bot state from file."""
        if os.path.exists(STATE_FILE):
            with open(STATE_FILE, 'r') as f:
                return json.load(f)
        return {'positions': {}, 'last_rebalance': None}

    def _save_state(self):
        """Save bot state to file."""
        os.makedirs(os.path.dirname(STATE_FILE), exist_ok=True)
        with open(STATE_FILE, 'w') as f:
            json.dump(self.state, f, indent=2, default=str)

    def get_account(self) -> Dict:
        """Get account info."""
        account = self.api.get_account()
        return {
            'equity': float(account.equity),
            'cash': float(account.cash),
            'buying_power': float(account.buying_power),
            'status': account.status
        }

    def get_positions(self) -> Dict[str, Dict]:
        """Get current positions."""
        positions = {}
        for pos in self.api.list_positions():
            positions[pos.symbol] = {
                'qty': float(pos.qty),
                'avg_price': float(pos.avg_entry_price),
                'current_price': float(pos.current_price),
                'market_value': float(pos.market_value),
                'pnl': float(pos.unrealized_pl),
                'pnl_pct': float(pos.unrealized_plpc) * 100
            }
        return positions

    def get_prices(self, symbols: List[str], lookback_days: int = 30) -> pd.DataFrame:
        """Get historical prices from Alpaca."""
        end = datetime.now()
        start = end - timedelta(days=lookback_days + 10)  # Extra days for weekends

        bars = self.api.get_bars(
            symbols,
            '1Day',
            start=start.strftime('%Y-%m-%d'),
            end=end.strftime('%Y-%m-%d'),
            adjustment='split'
        ).df

        # Pivot to get prices by symbol
        if len(bars) > 0:
            bars = bars.reset_index()
            prices = bars.pivot(index='timestamp', columns='symbol', values='close')
            return prices.ffill()
        return pd.DataFrame()

    def calculate_momentum(self, prices: pd.DataFrame) -> pd.Series:
        """Calculate 21-day momentum for all symbols."""
        if len(prices) < CONFIG['lookback']:
            logger.warning(f"Not enough data for momentum calculation")
            return pd.Series()

        momentum = prices.iloc[-1] / prices.iloc[-CONFIG['lookback']] - 1
        return momentum.dropna().sort_values(ascending=False)

    def is_rebalance_day(self) -> bool:
        """Check if today is rebalance day (first trading day of month)."""
        today = datetime.now()

        # Get market calendar
        calendar = self.api.get_calendar(
            start=today.replace(day=1).strftime('%Y-%m-%d'),
            end=today.strftime('%Y-%m-%d')
        )

        if calendar:
            first_trading_day = calendar[0].date
            is_rebal = today.date() == first_trading_day
            logger.info(f"First trading day of month: {first_trading_day}, Today: {today.date()}, Rebalance: {is_rebal}")
            return is_rebal
        return False

    def check_trailing_stops(self) -> List[str]:
        """Check and execute trailing stops. Returns list of stopped symbols."""
        stopped = []
        positions = self.get_positions()

        for symbol, pos in positions.items():
            if symbol not in self.state['positions']:
                # New position not in state, initialize
                self.state['positions'][symbol] = {
                    'entry_price': pos['avg_price'],
                    'high_price': pos['current_price'],
                    'stop_price': pos['avg_price'] * (1 - CONFIG['stop_loss'])
                }
                logger.info(f"Initialized stop for {symbol}: entry=${pos['avg_price']:.2f}, stop=${self.state['positions'][symbol]['stop_price']:.2f}")

            state = self.state['positions'][symbol]
            current_price = pos['current_price']

            # Update trailing stop if price made new high
            if current_price > state['high_price']:
                state['high_price'] = current_price
                state['stop_price'] = current_price * (1 - CONFIG['stop_loss'])
                logger.info(f"{symbol}: New high ${current_price:.2f}, stop raised to ${state['stop_price']:.2f}")

            # Check if stopped out
            if current_price <= state['stop_price']:
                pnl_pct = (current_price / state['entry_price'] - 1) * 100
                logger.warning(f"STOP TRIGGERED: {symbol} at ${current_price:.2f} (stop=${state['stop_price']:.2f}), P&L: {pnl_pct:+.1f}%")
                stopped.append(symbol)

        self._save_state()
        return stopped

    def close_position(self, symbol: str) -> bool:
        """Close a position."""
        try:
            self.api.close_position(symbol)
            logger.info(f"Closed position: {symbol}")

            # Remove from state
            if symbol in self.state['positions']:
                del self.state['positions'][symbol]
            self._save_state()
            return True
        except APIError as e:
            logger.error(f"Error closing {symbol}: {e}")
            return False

    def open_position(self, symbol: str, weight: float) -> bool:
        """Open a new position with given portfolio weight."""
        try:
            account = self.get_account()
            target_value = account['equity'] * weight

            # Get current price
            quote = self.api.get_latest_quote(symbol)
            price = float(quote.ask_price) if quote.ask_price else float(quote.bid_price)

            if price <= 0:
                logger.error(f"Invalid price for {symbol}: {price}")
                return False

            qty = int(target_value / price)
            if qty <= 0:
                logger.warning(f"Quantity too small for {symbol}: {qty}")
                return False

            # Submit order
            order = self.api.submit_order(
                symbol=symbol,
                qty=qty,
                side='buy',
                type='market',
                time_in_force='day'
            )

            logger.info(f"Opened position: {symbol}, qty={qty}, ~${target_value:.0f}")

            # Initialize state
            self.state['positions'][symbol] = {
                'entry_price': price,
                'high_price': price,
                'stop_price': price * (1 - CONFIG['stop_loss'])
            }
            self._save_state()
            return True

        except APIError as e:
            logger.error(f"Error opening {symbol}: {e}")
            return False

    def rebalance(self):
        """Execute monthly rebalance."""
        logger.info("=" * 60)
        logger.info("STARTING MONTHLY REBALANCE")
        logger.info("=" * 60)

        # Get prices and calculate momentum
        logger.info("Fetching prices...")
        prices = self.get_prices(self.universe, lookback_days=CONFIG['lookback'] + 5)

        if prices.empty:
            logger.error("No price data available")
            return

        available = [s for s in self.universe if s in prices.columns]
        logger.info(f"Available symbols: {len(available)}/{len(self.universe)}")

        # Calculate momentum
        momentum = self.calculate_momentum(prices[available])
        top_n = momentum.head(CONFIG['top_n'])

        logger.info(f"\nTop {CONFIG['top_n']} by momentum:")
        for symbol, mom in top_n.items():
            logger.info(f"  {symbol}: {mom*100:+.1f}%")

        target_symbols = set(top_n.index)

        # Get current positions
        current_positions = self.get_positions()
        current_symbols = set(current_positions.keys())

        # Close positions not in target
        to_close = current_symbols - target_symbols
        for symbol in to_close:
            logger.info(f"Closing {symbol} (not in top {CONFIG['top_n']})")
            self.close_position(symbol)

        # Open new positions
        to_open = target_symbols - current_symbols
        weight = 1.0 / CONFIG['top_n']

        for symbol in to_open:
            logger.info(f"Opening {symbol} (new top {CONFIG['top_n']})")
            self.open_position(symbol, weight)

        # Update rebalance date
        self.state['last_rebalance'] = datetime.now().isoformat()
        self._save_state()

        logger.info("=" * 60)
        logger.info("REBALANCE COMPLETE")
        logger.info("=" * 60)

    def redistribute_capital(self):
        """Redistribute capital to surviving positions after stop-outs."""
        positions = self.get_positions()

        if not positions:
            logger.info("No positions to redistribute to")
            return

        account = self.get_account()
        target_weight = 1.0 / len(positions)
        target_value_each = account['equity'] * target_weight

        logger.info(f"\nREDISTRIBUTING to {len(positions)} positions ({target_weight*100:.1f}% each)")

        for symbol, pos in positions.items():
            current_value = pos['market_value']
            diff = target_value_each - current_value

            if abs(diff) < 50:  # Skip if difference < $50
                continue

            try:
                if diff > 0:
                    # Need to buy more
                    qty = int(diff / pos['current_price'])
                    if qty > 0:
                        self.api.submit_order(
                            symbol=symbol,
                            qty=qty,
                            side='buy',
                            type='market',
                            time_in_force='day'
                        )
                        logger.info(f"  {symbol}: Bought {qty} shares (adding ${diff:.0f})")
                else:
                    # Need to sell some
                    qty = int(abs(diff) / pos['current_price'])
                    if qty > 0:
                        self.api.submit_order(
                            symbol=symbol,
                            qty=qty,
                            side='sell',
                            type='market',
                            time_in_force='day'
                        )
                        logger.info(f"  {symbol}: Sold {qty} shares (reducing ${abs(diff):.0f})")
            except APIError as e:
                logger.error(f"  {symbol}: Redistribution failed: {e}")

    def run_daily(self):
        """Run daily bot logic."""
        logger.info("=" * 60)
        logger.info(f"MOMENTUM BOT - {datetime.now().strftime('%Y-%m-%d %H:%M')}")
        logger.info("=" * 60)

        # Check account
        account = self.get_account()
        logger.info(f"Account: ${account['equity']:,.0f} equity, ${account['cash']:,.0f} cash")

        # Check market is open
        clock = self.api.get_clock()
        if not clock.is_open:
            logger.info("Market is closed")
            return

        # Check trailing stops
        logger.info("\nChecking trailing stops...")
        stopped = self.check_trailing_stops()

        for symbol in stopped:
            self.close_position(symbol)

        # Redistribute capital to survivors if enabled
        if stopped and CONFIG.get('redistribute', False):
            self.redistribute_capital()

        # Check if rebalance day
        if self.is_rebalance_day():
            self.rebalance()
        else:
            logger.info("Not rebalance day - stops checked only")

        # Show current positions
        positions = self.get_positions()
        if positions:
            logger.info(f"\nCurrent positions ({len(positions)}):")
            for symbol, pos in positions.items():
                state = self.state['positions'].get(symbol, {})
                stop = state.get('stop_price', 0)
                logger.info(f"  {symbol}: ${pos['current_price']:.2f} (stop=${stop:.2f}), P&L: {pos['pnl_pct']:+.1f}%")


def main():
    """Main entry point."""
    # Create logs directory
    os.makedirs('logs', exist_ok=True)

    # Check credentials
    if not os.environ.get('APCA_API_KEY_ID'):
        print("ERROR: Set APCA_API_KEY_ID environment variable")
        print("  export APCA_API_KEY_ID='your_key'")
        print("  export APCA_API_SECRET_KEY='your_secret'")
        return

    # Run bot
    bot = MomentumBot(paper=CONFIG['paper'])
    bot.run_daily()


if __name__ == '__main__':
    main()
