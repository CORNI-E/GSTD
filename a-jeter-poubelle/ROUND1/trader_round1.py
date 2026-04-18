import json
import math
import numpy as np
from typing import Any, List
from datamodel import Listing, Observation, Order, OrderDepth, ProsperityEncoder, Symbol, Trade, TradingState


class Logger:
    def __init__(self) -> None:
        self.logs = ""
        self.max_log_length = 3750

    def print(self, *objects: Any, sep: str = " ", end: str = "\n") -> None:
        self.logs += sep.join(map(str, objects)) + end

    def flush(self, state: TradingState, orders: dict[Symbol, list[Order]], conversions: int, trader_data: str) -> None:
        base_length = len(
            self.to_json(
                [
                    self.compress_state(state, ""),
                    self.compress_orders(orders),
                    conversions,
                    "",
                    "",
                ]
            )
        )
        max_item_length = (self.max_log_length - base_length) // 3
        print(
            self.to_json(
                [
                    self.compress_state(state, self.truncate(state.traderData, max_item_length)),
                    self.compress_orders(orders),
                    conversions,
                    self.truncate(trader_data, max_item_length),
                    self.truncate(self.logs, max_item_length),
                ]
            )
        )
        self.logs = ""

    def compress_state(self, state: TradingState, trader_data: str) -> list[Any]:
        return [
            state.timestamp,
            trader_data,
            self.compress_listings(state.listings),
            self.compress_order_depths(state.order_depths),
            self.compress_trades(state.own_trades),
            self.compress_trades(state.market_trades),
            state.position,
            self.compress_observations(state.observations),
        ]

    def compress_listings(self, listings: dict[Symbol, Listing]) -> list[list[Any]]:
        compressed = []
        for listing in listings.values():
            compressed.append([listing.symbol, listing.product, listing.denomination])
        return compressed

    def compress_order_depths(self, order_depths: dict[Symbol, OrderDepth]) -> dict[Symbol, list[Any]]:
        compressed = {}
        for symbol, order_depth in order_depths.items():
            compressed[symbol] = [order_depth.buy_orders, order_depth.sell_orders]
        return compressed

    def compress_trades(self, trades: dict[Symbol, list[Trade]]) -> list[list[Any]]:
        compressed = []
        for arr in trades.values():
            for trade in arr:
                compressed.append(
                    [trade.symbol, trade.price, trade.quantity, trade.buyer, trade.seller, trade.timestamp]
                )
        return compressed

    def compress_observations(self, observations: Observation) -> list[Any]:
        conversion_observations = {}
        for product, observation in observations.conversionObservations.items():
            conversion_observations[product] = [
                observation.bidPrice, observation.askPrice,
                observation.transportFees, observation.exportTariff,
                observation.importTariff, observation.sugarPrice,
                observation.sunlightIndex,
            ]
        return [observations.plainValueObservations, conversion_observations]

    def compress_orders(self, orders: dict[Symbol, list[Order]]) -> list[list[Any]]:
        compressed = []
        for arr in orders.values():
            for order in arr:
                compressed.append([order.symbol, order.price, order.quantity])
        return compressed

    def to_json(self, value: Any) -> str:
        return json.dumps(value, cls=ProsperityEncoder, separators=(",", ":"))

    def truncate(self, value: str, max_length: int) -> str:
        if len(value) <= max_length:
            return value
        return value[: max_length - 3] + "..."


logger = Logger()


class Trader:
    def __init__(self):
        # =========================================================
        # Position limits for Round 1 products
        # =========================================================
        self.limits = {
            'INTARIAN_PEPPER_ROOT': 80,
            'ASH_COATED_OSMIUM': 80,
        }

        self.orders = {}
        self.conversions = 0
        self.traderData = ""

        # --- Pepper Root (stable product, like RAINFOREST_RESIN) ---
        self.pepper_position = 0
        self.pepper_buy_orders = 0
        self.pepper_sell_orders = 0

        # --- Ash Coated Osmium (volatile, like SQUID_INK) ---
        self.osmium_position = 0
        self.osmium_buy_orders = 0
        self.osmium_sell_orders = 0

        # Moving average windows for osmium
        self.osmium_short_window_prices = []
        self.osmium_long_window_prices = []
        self.osmium_volatility_diffs = []

        self.osmium_prev_price = None
        self.osmium_prev_vol = None

        # Osmium hyperparams (tuned from SQUID_INK analysis)
        self.osmium_volatility_window = 50
        self.osmium_short_window = 50
        self.osmium_long_window = 200

    # =================================================================
    #  HELPER FUNCTIONS
    # =================================================================
    def get_pos(self, state, product):
        return state.position.get(product, 0)

    def send_buy(self, product, price, amount, msg=None):
        self.orders[product].append(Order(product, int(price), amount))
        if msg:
            logger.print(msg)

    def send_sell(self, product, price, amount, msg=None):
        self.orders[product].append(Order(product, int(price), amount))
        if msg:
            logger.print(msg)

    def get_buy_budget(self, product):
        """How many more units we can buy."""
        limit = self.limits[product]
        if product == 'INTARIAN_PEPPER_ROOT':
            return limit - self.pepper_position - self.pepper_buy_orders
        elif product == 'ASH_COATED_OSMIUM':
            return limit - self.osmium_position - self.osmium_buy_orders
        return 0

    def get_sell_budget(self, product):
        """How many more units we can sell (positive number)."""
        limit = self.limits[product]
        if product == 'INTARIAN_PEPPER_ROOT':
            return self.pepper_position + limit - self.pepper_sell_orders
        elif product == 'ASH_COATED_OSMIUM':
            return self.osmium_position + limit - self.osmium_sell_orders
        return 0

    def record_buy(self, product, size):
        if product == 'INTARIAN_PEPPER_ROOT':
            self.pepper_buy_orders += size
        elif product == 'ASH_COATED_OSMIUM':
            self.osmium_buy_orders += size

    def record_sell(self, product, size):
        if product == 'INTARIAN_PEPPER_ROOT':
            self.pepper_sell_orders += size
        elif product == 'ASH_COATED_OSMIUM':
            self.osmium_sell_orders += size

    # ----- Take existing orders from the book -----
    def take_asks(self, state, product, max_price, depth=3):
        """Buy from asks that are <= max_price."""
        order_depth = state.order_depths[product]
        if not order_depth.sell_orders:
            return
        asks = sorted(order_depth.sell_orders.items())  # lowest ask first
        for ask, neg_amount in asks[:depth]:
            amount = -neg_amount  # sell_orders have negative quantities
            if ask <= max_price:
                budget = self.get_buy_budget(product)
                size = min(budget, amount)
                if size > 0:
                    self.record_buy(product, size)
                    self.send_buy(product, ask, size,
                                  msg=f"TAKE BUY {product} {size}x @ {ask}")

    def take_bids(self, state, product, min_price, depth=3):
        """Sell into bids that are >= min_price."""
        order_depth = state.order_depths[product]
        if not order_depth.buy_orders:
            return
        bids = sorted(order_depth.buy_orders.items(), reverse=True)  # highest bid first
        for bid, amount in bids[:depth]:
            if bid >= min_price:
                budget = self.get_sell_budget(product)
                size = min(budget, amount)
                if size > 0:
                    self.record_sell(product, size)
                    self.send_sell(product, bid, -size,
                                   msg=f"TAKE SELL {product} {size}x @ {bid}")

    # ----- Find best bid/ask outside fair price -----
    def best_bid_below(self, state, product, price):
        """Best bid strictly below price."""
        order_depth = state.order_depths[product]
        for bid, _ in sorted(order_depth.buy_orders.items(), reverse=True):
            if bid < price:
                return bid
        return None

    def best_ask_above(self, state, product, price):
        """Best ask strictly above price."""
        order_depth = state.order_depths[product]
        for ask, _ in sorted(order_depth.sell_orders.items()):
            if ask > price:
                return ask
        return None

    # =================================================================
    #  INTARIAN_PEPPER_ROOT  –  Stable market making around fair value
    #  (Analogous to RAINFOREST_RESIN at 10,000)
    # =================================================================
    def trade_pepper_root(self, state):
        product = 'INTARIAN_PEPPER_ROOT'
        if product not in state.order_depths:
            return

        order_depth = state.order_depths[product]
        if not order_depth.sell_orders or not order_depth.buy_orders:
            return

        # Estimate fair value from the order book
        # Pepper root is described as "steady" — we compute mid-price
        best_bid = max(order_depth.buy_orders.keys())
        best_ask = min(order_depth.sell_orders.keys())
        mid = (best_bid + best_ask) / 2
        fair = round(mid)

        logger.print(f"PEPPER_ROOT: mid={mid:.1f}, fair={fair}")

        # 1) Take any mispriced orders
        self.take_asks(state, product, mid, depth=3)
        self.take_bids(state, product, mid, depth=3)

        # 2) Place market-making quotes
        # Try to undercut other market makers
        other_bid = self.best_bid_below(state, product, fair)
        other_ask = self.best_ask_above(state, product, fair)

        buy_price = math.floor(mid) - 2
        sell_price = math.ceil(mid) + 2

        if other_bid is not None and other_ask is not None:
            if other_bid + 1 < mid:
                buy_price = other_bid + 1
            if other_ask - 1 > mid:
                sell_price = other_ask - 1

        max_buy = max(0, self.get_buy_budget(product))
        max_sell = max(0, self.get_sell_budget(product))

        if max_buy > 0:
            self.send_buy(product, buy_price, max_buy,
                          msg=f"PEPPER MM BUY {max_buy}x @ {buy_price}")
        if max_sell > 0:
            self.send_sell(product, sell_price, -max_sell,
                           msg=f"PEPPER MM SELL {max_sell}x @ {sell_price}")

    # =================================================================
    #  ASH_COATED_OSMIUM  –  Volatile with hidden pattern
    #  Strategy: Market making + mean reversion via SMA crossover
    #  (Evolved from SQUID_INK strategy)
    # =================================================================
    def trade_osmium(self, state):
        product = 'ASH_COATED_OSMIUM'
        if product not in state.order_depths:
            return

        order_depth = state.order_depths[product]
        if not order_depth.sell_orders or not order_depth.buy_orders:
            return

        best_bid = max(order_depth.buy_orders.keys())
        best_ask = min(order_depth.sell_orders.keys())

        # Also use worst levels for a wider fair price estimate
        worst_ask = max(order_depth.sell_orders.keys())
        worst_bid = min(order_depth.buy_orders.keys())

        mid = (best_bid + best_ask) / 2
        fair_int = int(math.ceil(mid))

        logger.print(f"OSMIUM: mid={mid:.1f}")

        # --- Update rolling windows ---
        self.osmium_long_window_prices.append(mid)
        self.osmium_long_window_prices = self.osmium_long_window_prices[-self.osmium_long_window:]
        self.osmium_short_window_prices.append(mid)
        self.osmium_short_window_prices = self.osmium_short_window_prices[-self.osmium_short_window:]

        if self.osmium_prev_price is not None:
            diff = mid - self.osmium_prev_price
            self.osmium_volatility_diffs.append(diff)
            self.osmium_volatility_diffs = self.osmium_volatility_diffs[-self.osmium_volatility_window:]

        # --- Decide which sides to quote ---
        buy_side = True
        sell_side = True

        # Volatility
        volatility = 0
        if len(self.osmium_volatility_diffs) >= self.osmium_volatility_window:
            volatility = float(np.std(self.osmium_volatility_diffs))
            logger.print(f"OSMIUM VOLATILITY: {volatility:.2f}")

        # SMA crossover trend detection
        if len(self.osmium_long_window_prices) >= self.osmium_long_window:
            short_mean = float(np.mean(self.osmium_short_window_prices))
            long_mean = float(np.mean(self.osmium_long_window_prices))

            if short_mean > long_mean:
                # Uptrend — don't sell into it, only buy
                sell_side = False
                logger.print("OSMIUM: UPTREND detected (short > long)")
            elif short_mean < long_mean:
                # Downtrend — don't buy into it, only sell
                buy_side = False
                logger.print("OSMIUM: DOWNTREND detected (short < long)")

        # If near position limit, enable both sides to unwind
        pos = self.get_pos(state, product)
        pos_pct = abs(pos) / self.limits[product]
        if pos_pct > 0.75:
            buy_side = True
            sell_side = True
            logger.print(f"OSMIUM: Near limit ({pos}), both sides ON")

        # --- Flash crash / spike detection ---
        if self.osmium_prev_vol is not None and volatility > 0:
            delta_vol = abs(volatility - self.osmium_prev_vol)
            if delta_vol > 2 and self.osmium_prev_price is not None:
                logger.print(f"OSMIUM: VOLATILITY SPIKE delta={delta_vol:.2f}")
                if self.osmium_prev_price > mid:
                    # Price crashed → buy the dip
                    self.take_asks(state, product, mid + 4, depth=3)
                else:
                    # Price spiked → sell the top
                    self.take_bids(state, product, mid - 4, depth=3)

        self.osmium_prev_vol = volatility

        # --- Take mispriced orders ---
        if buy_side:
            self.take_asks(state, product, mid, depth=3)
        if sell_side:
            self.take_bids(state, product, mid, depth=3)

        # --- Place market-making quotes ---
        other_bid = self.best_bid_below(state, product, fair_int)
        other_ask = self.best_ask_above(state, product, fair_int)

        buy_price = math.floor(mid) - 2
        sell_price = math.ceil(mid) + 2

        if other_bid is not None and other_ask is not None:
            if other_ask - 1 > mid:
                sell_price = other_ask - 1
            if other_bid + 1 < mid:
                buy_price = other_bid + 1

        max_buy = max(0, self.get_buy_budget(product))
        max_sell = max(0, self.get_sell_budget(product))

        if buy_side and max_buy > 0:
            self.send_buy(product, buy_price, max_buy,
                          msg=f"OSMIUM MM BUY {max_buy}x @ {buy_price}")
        if sell_side and max_sell > 0:
            self.send_sell(product, sell_price, -max_sell,
                           msg=f"OSMIUM MM SELL {max_sell}x @ {sell_price}")

        self.osmium_prev_price = mid

    # =================================================================
    #  MAIN RUN
    # =================================================================
    def reset_orders(self, state):
        self.orders = {}
        self.conversions = 0

        self.pepper_position = self.get_pos(state, 'INTARIAN_PEPPER_ROOT')
        self.pepper_buy_orders = 0
        self.pepper_sell_orders = 0

        self.osmium_position = self.get_pos(state, 'ASH_COATED_OSMIUM')
        self.osmium_buy_orders = 0
        self.osmium_sell_orders = 0

        for product in state.order_depths:
            self.orders[product] = []

    def run(self, state: TradingState):
        # Restore state from traderData if needed
        if state.traderData:
            try:
                saved = json.loads(state.traderData)
                self.osmium_short_window_prices = saved.get("osw_short", [])
                self.osmium_long_window_prices = saved.get("osw_long", [])
                self.osmium_volatility_diffs = saved.get("osw_vol", [])
                self.osmium_prev_price = saved.get("osw_prev_price", None)
                self.osmium_prev_vol = saved.get("osw_prev_vol", None)
            except Exception:
                pass

        self.reset_orders(state)

        # Trade each product
        self.trade_pepper_root(state)
        self.trade_osmium(state)

        # Save state for next tick
        trader_data = json.dumps({
            "osw_short": self.osmium_short_window_prices[-self.osmium_short_window:],
            "osw_long": self.osmium_long_window_prices[-self.osmium_long_window:],
            "osw_vol": self.osmium_volatility_diffs[-self.osmium_volatility_window:],
            "osw_prev_price": self.osmium_prev_price,
            "osw_prev_vol": self.osmium_prev_vol,
        })

        logger.flush(state, self.orders, self.conversions, trader_data)
        return self.orders, self.conversions, trader_data
