import json
import math
import numpy as np
from typing import Any, List, Optional
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


# ─────────────────────────────────────────────
#  Exponential Moving Average (faster than SMA)
# ─────────────────────────────────────────────
class EMA:
    """Online EMA calculator – no full window needed."""
    def __init__(self, period: int):
        self.period = period
        self.alpha = 2.0 / (period + 1)
        self.value: Optional[float] = None

    def update(self, price: float) -> float:
        if self.value is None:
            self.value = price
        else:
            self.value = self.alpha * price + (1 - self.alpha) * self.value
        return self.value

    def to_dict(self):
        return {"period": self.period, "value": self.value}

    @classmethod
    def from_dict(cls, d):
        obj = cls(d["period"])
        obj.value = d["value"]
        return obj


class Trader:
    def __init__(self):
        # =========================================================
        # Position limits
        # =========================================================
        self.limits = {
            'INTARIAN_PEPPER_ROOT': 80,
            'ASH_COATED_OSMIUM':    80,
        }

        self.orders: dict[Symbol, list[Order]] = {}
        self.conversions = 0

        # ── Pepper Root ───────────────────────────────────────────
        self.pepper_pos = 0
        self.pepper_buys = 0
        self.pepper_sells = 0
        # EMA for fair-value drift tracking
        self.pepper_ema_fast = EMA(20)
        self.pepper_ema_slow = EMA(100)

        # ── Ash Coated Osmium ─────────────────────────────────────
        self.osmium_pos = 0
        self.osmium_buys = 0
        self.osmium_sells = 0

        # EMAs for trend detection
        self.osmium_ema_fast = EMA(50)
        self.osmium_ema_slow = EMA(200)

        # Rolling window for volatility (std of diffs)
        self.osmium_vol_diffs: list[float] = []
        self.osmium_vol_window = 50

        # Bollinger Bands: rolling prices for mean + std
        self.osmium_bb_prices: list[float] = []
        self.osmium_bb_window = 50   # ~5 s of data at 100 ms ticks

        self.osmium_prev_price: Optional[float] = None
        self.osmium_prev_vol: Optional[float] = None

    # =================================================================
    #  UTILITY
    # =================================================================
    def _pos(self, state, product):
        return state.position.get(product, 0)

    # ── Budget helpers ─────────────────────────────────────────────
    def _buy_budget(self, product):
        lim = self.limits[product]
        if product == 'INTARIAN_PEPPER_ROOT':
            return lim - self.pepper_pos - self.pepper_buys
        return lim - self.osmium_pos - self.osmium_buys

    def _sell_budget(self, product):
        lim = self.limits[product]
        if product == 'INTARIAN_PEPPER_ROOT':
            return self.pepper_pos + lim - self.pepper_sells
        return self.osmium_pos + lim - self.osmium_sells

    def _record_buy(self, product, size):
        if product == 'INTARIAN_PEPPER_ROOT':
            self.pepper_buys += size
        else:
            self.osmium_buys += size

    def _record_sell(self, product, size):
        if product == 'INTARIAN_PEPPER_ROOT':
            self.pepper_sells += size
        else:
            self.osmium_sells += size

    # ── Order helpers ──────────────────────────────────────────────
    def _buy(self, product, price, qty, msg=None):
        self.orders[product].append(Order(product, int(price), qty))
        if msg:
            logger.print(msg)

    def _sell(self, product, price, qty, msg=None):
        self.orders[product].append(Order(product, int(price), qty))
        if msg:
            logger.print(msg)

    # ── Book helpers ───────────────────────────────────────────────
    def _weighted_mid(self, order_depth: OrderDepth) -> Optional[float]:
        """
        Imbalance-weighted mid-price.
        Fair = (bid * ask_vol + ask * bid_vol) / (bid_vol + ask_vol)
        Gives a better leading estimate of where price is heading.
        """
        if not order_depth.buy_orders or not order_depth.sell_orders:
            return None
        best_bid = max(order_depth.buy_orders)
        best_ask = min(order_depth.sell_orders)
        bid_vol = order_depth.buy_orders[best_bid]
        ask_vol = -order_depth.sell_orders[best_ask]  # stored negative
        denom = bid_vol + ask_vol
        if denom == 0:
            return (best_bid + best_ask) / 2
        return (best_bid * ask_vol + best_ask * bid_vol) / denom

    def _order_flow_signal(self, state, product) -> float:
        """
        Positive → recent aggressive buying (bullish).
        Negative → recent aggressive selling (bearish).
        Based on market trades from previous tick.
        """
        trades = state.market_trades.get(product, [])
        signal = 0.0
        for t in trades:
            # Aggressive buy: traded at ask → positive
            # Aggressive sell: traded at bid → negative
            # We approximate direction by sign of (price - mid)
            od = state.order_depths.get(product)
            if od and od.buy_orders and od.sell_orders:
                mid = (max(od.buy_orders) + min(od.sell_orders)) / 2
                signal += t.quantity * (1 if t.price >= mid else -1)
        return signal

    def _take_asks(self, state, product, max_price, depth=3):
        """Buy from asks ≤ max_price."""
        od = state.order_depths[product]
        for ask, neg_qty in sorted(od.sell_orders.items())[:depth]:
            if ask > max_price:
                break
            qty = min(self._buy_budget(product), -neg_qty)
            if qty > 0:
                self._record_buy(product, qty)
                self._buy(product, ask, qty, f"TAKE BUY {product} {qty}@{ask}")

    def _take_bids(self, state, product, min_price, depth=3):
        """Sell into bids ≥ min_price."""
        od = state.order_depths[product]
        for bid, qty in sorted(od.buy_orders.items(), reverse=True)[:depth]:
            if bid < min_price:
                break
            size = min(self._sell_budget(product), qty)
            if size > 0:
                self._record_sell(product, size)
                self._sell(product, bid, -size, f"TAKE SELL {product} {size}@{bid}")

    def _best_bid_below(self, state, product, price):
        od = state.order_depths[product]
        for bid in sorted(od.buy_orders, reverse=True):
            if bid < price:
                return bid
        return None

    def _best_ask_above(self, state, product, price):
        od = state.order_depths[product]
        for ask in sorted(od.sell_orders):
            if ask > price:
                return ask
        return None

    # =================================================================
    #  INTARIAN_PEPPER_ROOT
    #  – Trending upward slowly, high liquidity, tight spread
    #  – Strategy: market-making with EMA-adjusted fair value +
    #    penny-jump protection + inventory skew
    # =================================================================
    def trade_pepper_root(self, state):
        product = 'INTARIAN_PEPPER_ROOT'
        if product not in state.order_depths:
            return
        od = state.order_depths[product]
        if not od.sell_orders or not od.buy_orders:
            return

        wmid = self._weighted_mid(od)
        if wmid is None:
            return

        # Update EMAs
        fast = self.pepper_ema_fast.update(wmid)
        slow = self.pepper_ema_slow.update(wmid)

        # Fair value: favour the EMA fast (more responsive)
        fair = fast
        logger.print(f"PEPPER: wmid={wmid:.1f} fast={fast:.1f} slow={slow:.1f}")

        # Order-flow boost: slightly adjust fair in direction of flow
        of = self._order_flow_signal(state, product)
        if abs(of) > 5:
            fair += 0.3 * (1 if of > 0 else -1)

        # 1) Take clearly mispriced orders
        self._take_asks(state, product, fair - 1)
        self._take_bids(state, product, fair + 1)

        # 2) Inventory skew: shift quotes to unwind excess position
        pos = self._pos(state, product)
        lim = self.limits[product]
        skew = pos / lim  # ∈ [-1, +1]
        # skew > 0 → long → lower our ask, raise our bid threshold
        buy_skew  = -round(skew * 3)   # reduces buy aggressiveness when long
        sell_skew =  round(skew * 3)   # reduces sell aggressiveness when short

        # 3) Penny-jump logic (don't jump trivial size)
        other_bid = self._best_bid_below(state, product, round(fair))
        other_ask = self._best_ask_above(state, product, round(fair))

        # Minimum meaningful size to bother jumping
        MIN_JUMP_SIZE = 3

        spread = 3  # base half-spread
        buy_price  = math.floor(fair) - spread + buy_skew
        sell_price = math.ceil(fair)  + spread + sell_skew

        if other_bid is not None:
            ob_size = od.buy_orders.get(other_bid, 0)
            if ob_size >= MIN_JUMP_SIZE and other_bid + 1 < fair:
                buy_price = other_bid + 1 + buy_skew

        if other_ask is not None:
            oa_size = -od.sell_orders.get(other_ask, 0)
            if oa_size >= MIN_JUMP_SIZE and other_ask - 1 > fair:
                sell_price = other_ask - 1 + sell_skew

        # 4) Split orders across 2 levels for smoother fills
        max_buy  = max(0, self._buy_budget(product))
        max_sell = max(0, self._sell_budget(product))

        if max_buy > 0:
            # 60% at best price, 40% one tick lower
            q1 = max(1, round(max_buy * 0.6))
            q2 = max_buy - q1
            self._buy(product, buy_price, q1,
                      f"PEPPER MM BUY {q1}@{buy_price}")
            if q2 > 0:
                self._buy(product, buy_price - 1, q2,
                          f"PEPPER MM BUY2 {q2}@{buy_price-1}")

        if max_sell > 0:
            q1 = max(1, round(max_sell * 0.6))
            q2 = max_sell - q1
            self._sell(product, sell_price, -q1,
                       f"PEPPER MM SELL {q1}@{sell_price}")
            if q2 > 0:
                self._sell(product, sell_price + 1, -q2,
                           f"PEPPER MM SELL2 {q2}@{sell_price+1}")

    # =================================================================
    #  ASH_COATED_OSMIUM
    #  – Mean-reverting around ~9993-10002
    #  – Strategy: Bollinger Bands mean-reversion + EMA trend filter
    #    + dynamic spread + inventory skew + order-flow signal
    # =================================================================
    def trade_osmium(self, state):
        product = 'ASH_COATED_OSMIUM'
        if product not in state.order_depths:
            return
        od = state.order_depths[product]
        if not od.sell_orders or not od.buy_orders:
            return

        wmid = self._weighted_mid(od)
        if wmid is None:
            return

        # ── Update indicators ───────────────────────────────────────
        fast_ema = self.osmium_ema_fast.update(wmid)
        slow_ema = self.osmium_ema_slow.update(wmid)

        # Bollinger Bands
        self.osmium_bb_prices.append(wmid)
        self.osmium_bb_prices = self.osmium_bb_prices[-self.osmium_bb_window:]
        bb_mean = float(np.mean(self.osmium_bb_prices))
        bb_std  = float(np.std(self.osmium_bb_prices)) if len(self.osmium_bb_prices) > 5 else 5.0
        bb_upper = bb_mean + 2 * bb_std
        bb_lower = bb_mean - 2 * bb_std

        # Volatility (std of 1-tick differences)
        if self.osmium_prev_price is not None:
            self.osmium_vol_diffs.append(wmid - self.osmium_prev_price)
            self.osmium_vol_diffs = self.osmium_vol_diffs[-self.osmium_vol_window:]
        volatility = float(np.std(self.osmium_vol_diffs)) if len(self.osmium_vol_diffs) >= 10 else 5.0

        # Order-flow signal
        of = self._order_flow_signal(state, product)

        logger.print(
            f"OSMIUM: wmid={wmid:.1f} fast={fast_ema:.1f} slow={slow_ema:.1f} "
            f"BB=[{bb_lower:.1f},{bb_upper:.1f}] vol={volatility:.2f} of={of:.0f}"
        )

        # ── Decide directional bias ─────────────────────────────────
        # EMA crossover: slow trend filter
        trend_up   = fast_ema > slow_ema
        trend_down = fast_ema < slow_ema

        # Bollinger band breach: mean-reversion signal
        bb_buy  = wmid < bb_lower  # oversold → buy
        bb_sell = wmid > bb_upper  # overbought → sell

        # Order flow boosts conviction
        of_bullish = of > 10
        of_bearish = of < -10

        buy_side  = True
        sell_side = True

        # Only trade in direction of trend OR mean-reversion signal
        if self.osmium_ema_slow.value is not None:
            if trend_down and not bb_buy and not of_bullish:
                buy_side = False
            if trend_up and not bb_sell and not of_bearish:
                sell_side = False

        # Override: if near position limit, allow both sides (unwind)
        pos = self._pos(state, product)
        pos_frac = abs(pos) / self.limits[product]
        if pos_frac > 0.75:
            buy_side  = True
            sell_side = True
            logger.print(f"OSMIUM: Near limit pos={pos}, forcing both sides")

        # Aggressive mean-reversion: take immediately on BB breach
        if bb_buy:
            self._take_asks(state, product, wmid + 3, depth=3)
            logger.print(f"OSMIUM: BB oversold → aggressive buy")
        if bb_sell:
            self._take_bids(state, product, wmid - 3, depth=3)
            logger.print(f"OSMIUM: BB overbought → aggressive sell")

        # Volatility spike: flash crash / pump
        if self.osmium_prev_vol is not None:
            delta_vol = abs(volatility - self.osmium_prev_vol)
            if delta_vol > 2 and self.osmium_prev_price is not None:
                logger.print(f"OSMIUM: VOL SPIKE Δ={delta_vol:.2f}")
                if self.osmium_prev_price > wmid:
                    self._take_asks(state, product, wmid + 5, depth=3)
                else:
                    self._take_bids(state, product, wmid - 5, depth=3)
        self.osmium_prev_vol = volatility

        # Regular passive taking
        if buy_side:
            self._take_asks(state, product, wmid, depth=2)
        if sell_side:
            self._take_bids(state, product, wmid, depth=2)

        # ── Dynamic spread ──────────────────────────────────────────
        # Widen spread when volatile to protect against adverse selection
        base_spread = max(2, min(6, round(volatility * 0.5)))

        # Inventory skew: shift quotes toward mean
        lim = self.limits[product]
        skew = pos / lim
        buy_skew  = -round(skew * 4)
        sell_skew =  round(skew * 4)

        # Quote placement
        other_bid = self._best_bid_below(state, product, round(wmid))
        other_ask = self._best_ask_above(state, product, round(wmid))

        buy_price  = math.floor(wmid) - base_spread + buy_skew
        sell_price = math.ceil(wmid)  + base_spread + sell_skew

        if other_bid is not None and other_bid + 1 < wmid:
            buy_price = other_bid + 1 + buy_skew
        if other_ask is not None and other_ask - 1 > wmid:
            sell_price = other_ask - 1 + sell_skew

        max_buy  = max(0, self._buy_budget(product))
        max_sell = max(0, self._sell_budget(product))

        # Split orders: 60/40 across two price levels
        if buy_side and max_buy > 0:
            q1 = max(1, round(max_buy * 0.6))
            q2 = max_buy - q1
            self._buy(product, buy_price, q1,
                      f"OSMIUM MM BUY {q1}@{buy_price}")
            if q2 > 0:
                self._buy(product, buy_price - 1, q2,
                          f"OSMIUM MM BUY2 {q2}@{buy_price-1}")

        if sell_side and max_sell > 0:
            q1 = max(1, round(max_sell * 0.6))
            q2 = max_sell - q1
            self._sell(product, sell_price, -q1,
                       f"OSMIUM MM SELL {q1}@{sell_price}")
            if q2 > 0:
                self._sell(product, sell_price + 1, -q2,
                           f"OSMIUM MM SELL2 {q2}@{sell_price+1}")

        self.osmium_prev_price = wmid

    # =================================================================
    #  MAIN
    # =================================================================
    def _reset(self, state):
        self.orders = {p: [] for p in state.order_depths}
        self.conversions = 0

        self.pepper_pos   = self._pos(state, 'INTARIAN_PEPPER_ROOT')
        self.pepper_buys  = 0
        self.pepper_sells = 0

        self.osmium_pos   = self._pos(state, 'ASH_COATED_OSMIUM')
        self.osmium_buys  = 0
        self.osmium_sells = 0

    def run(self, state: TradingState):
        # ── Restore persisted state ─────────────────────────────────
        if state.traderData:
            try:
                s = json.loads(state.traderData)
                self.pepper_ema_fast  = EMA.from_dict(s["pef"])
                self.pepper_ema_slow  = EMA.from_dict(s["pes"])
                self.osmium_ema_fast  = EMA.from_dict(s["oef"])
                self.osmium_ema_slow  = EMA.from_dict(s["oes"])
                self.osmium_bb_prices = s.get("obb", [])
                self.osmium_vol_diffs = s.get("ovd", [])
                self.osmium_prev_price = s.get("opp", None)
                self.osmium_prev_vol   = s.get("opv", None)
            except Exception:
                pass  # Start fresh on decode error

        self._reset(state)

        self.trade_pepper_root(state)
        self.trade_osmium(state)

        # ── Persist state ───────────────────────────────────────────
        trader_data = json.dumps({
            "pef": self.pepper_ema_fast.to_dict(),
            "pes": self.pepper_ema_slow.to_dict(),
            "oef": self.osmium_ema_fast.to_dict(),
            "oes": self.osmium_ema_slow.to_dict(),
            "obb": self.osmium_bb_prices[-self.osmium_bb_window:],
            "ovd": self.osmium_vol_diffs[-self.osmium_vol_window:],
            "opp": self.osmium_prev_price,
            "opv": self.osmium_prev_vol,
        })

        logger.flush(state, self.orders, self.conversions, trader_data)
        return self.orders, self.conversions, trader_data