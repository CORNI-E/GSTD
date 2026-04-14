import json
from datamodel import OrderDepth, UserId, TradingState, Order
from typing import List, Dict

class Trader:
    POSITION_LIMITS = {
        'INTARIAN_PEPPER_ROOT': 80,
        'ASH_COATED_OSMIUM': 80
    }

    def run(self, state: TradingState) -> tuple[Dict[str, List[Order]], int, str]:
        result = {}
        
        # ---------------------------------------------------------
        # MÉMOIRE DYNAMIQUE : Récupération de l'état précédent
        # ---------------------------------------------------------
        try:
            if state.traderData:
                memory = json.loads(state.traderData)
            else:
                memory = {'EMA': {}}
        except Exception:
            memory = {'EMA': {}}
            
        ema = memory.get('EMA', {})
        # Alpha : la vitesse à laquelle notre robot s'adapte au nouveau prix.
        # 0.1 correspond environ à une mémoire lisse sur les 19 derniers ticks.
        alpha = 0.1 

        for product in state.order_depths:
            order_depth: OrderDepth = state.order_depths[product]
            orders: List[Order] = []
            
            # État de l'inventaire
            current_position = state.position.get(product, 0)
            limit = self.POSITION_LIMITS.get(product, 80)
            
            # Sécurité : vérifier que le carnet d'ordres est actif
            if len(order_depth.sell_orders) > 0 and len(order_depth.buy_orders) > 0:
                best_ask = min(order_depth.sell_orders.keys())
                best_bid = max(order_depth.buy_orders.keys())
                
                vol_ask = abs(order_depth.sell_orders[best_ask])
                vol_bid = order_depth.buy_orders[best_bid]
                
                mid_price = (best_ask + best_bid) / 2.0
                
                # ---------------------------------------------------------
                # 1. FAIR PRICE DYNAMIQUE (EMA - Anti Overfitting)
                # ---------------------------------------------------------
                # Si c'est le tout premier tick pour ce produit, on initialise
                if product not in ema:
                    ema[product] = mid_price
                else:
                    # Formule de l'EMA : (Prix Actuel * Poids) + (Ancien EMA * Reste du poids)
                    ema[product] = (mid_price * alpha) + (ema[product] * (1 - alpha))
                    
                fair_price = ema[product]

                # ---------------------------------------------------------
                # 2. SIGNAUX DE MARCHÉ (Pression et Inventaire)
                # ---------------------------------------------------------
                # Imbalance : De quel côté penche le carnet d'ordres actuel ?
                imbalance = (vol_bid - vol_ask) / (vol_bid + vol_ask)
                
                # Ratio d'inventaire : À quel point notre sac est-il plein (-1.0 à 1.0) ?
                inventory_ratio = current_position / limit
                
                # Paramètres de Market Making génériques et robustes
                imbalance_weight = 1.5 # On suit légèrement la pression du carnet
                skew_factor = 2.5      # On se protège agressivement si on a trop de stock
                base_spread = 1.5      # Marge bénéficiaire de base

                # Prix idéal = EMA + Poids du marché instantané - Risque de notre inventaire
                adjusted_fair_price = fair_price + (imbalance * imbalance_weight) - (inventory_ratio * skew_factor)
                
                # ---------------------------------------------------------
                # 3. CRÉATION DES ORDRES (Le Filet de Sécurité)
                # ---------------------------------------------------------
                our_bid_price = int(round(adjusted_fair_price - base_spread))
                our_ask_price = int(round(adjusted_fair_price + base_spread))
                
                # PENNYING : Règle absolue pour ne pas payer au-delà des prix réels du marché
                our_bid_price = min(our_bid_price, best_bid + 1)
                our_ask_price = max(our_ask_price, best_ask - 1)

                # Exécution ACHAT
                max_buy_qty = limit - current_position
                if max_buy_qty > 0:
                    orders.append(Order(product, our_bid_price, max_buy_qty))

                # Exécution VENTE
                max_sell_qty = -limit - current_position
                if max_sell_qty < 0:
                    orders.append(Order(product, our_ask_price, max_sell_qty))

            result[product] = orders

        # ---------------------------------------------------------
        # SAUVEGARDE : On enregistre notre EMA pour le prochain tick
        # ---------------------------------------------------------
        memory['EMA'] = ema
        traderData = json.dumps(memory)

        return result, 1, traderData