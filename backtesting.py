import os
import pandas as pd
import matplotlib.pyplot as plt
from typing import Dict, List
from datamodel import Order, TradingState, Symbol, OrderDepth, Listing, Observation
import random

# =================================================================
#  1. IMPORTE TON ALGORITHME ICI (OU COLLE LA CLASSE TRADER)
# =================================================================
# Si ton algorithme est dans un fichier `algo.py`, décommente la ligne suivante :
from trader_round1 import Trader 

# (Pour ce test, je pars du principe que ta classe Trader est déjà définie plus haut dans ton script)

## =================================================================
#  2. MOTEUR DE SIMULATION (MATCHING ENGINE) - VERSION AVANCÉE
# =================================================================


def simulate_matching(orders: Dict[Symbol, List[Order]], order_depths: Dict[Symbol, OrderDepth], 
                      positions: Dict[Symbol, int], cash: Dict[Symbol, float], limits: Dict[Symbol, int]):
    """
    Version RÉALISTE : Intègre les limites de volume du carnet et un taux d'exécution aléatoire.
    """
    # Paramètres de réalisme
    FILL_RATE_MAKER = 0.25  # Seulement 25% de chance d'être exécuté si on est "Maker" (file d'attente)
    
    for product, product_orders in orders.items():
        if product not in order_depths:
            continue
            
        depth = order_depths[product]
        current_pos = positions.get(product, 0)
        limit = limits[product]

        for order in product_orders:
            # --- ACHAT (BUY) ---
            if order.quantity > 0:
                # 1. Calcul de la quantité max achetable (respect des limites du challenge)
                qty_to_buy = min(order.quantity, limit - current_pos)
                if qty_to_buy <= 0: continue

                # 2. Logique TAKER (On tape dans le carnet adverse)
                # On regarde combien de volume est RÉELLEMENT dispo au prix de vente (Asks)
                available_volume = 0
                execution_price = 0
                for ask_price, ask_vol in sorted(depth.sell_orders.items()):
                    if order.price >= ask_price:
                        available_volume += abs(ask_vol)
                        execution_price = ask_price # On est exécuté au prix du vendeur
                        break # Simplification : on prend le meilleur niveau
                
                if available_volume > 0:
                    exec_qty = min(qty_to_buy, available_volume)
                    current_pos += exec_qty
                    cash[product] -= exec_qty * execution_price
                
                # 3. Logique MAKER (On attend dans le spread)
                elif order.price >= max(depth.buy_orders.keys()):
                    # Simulation de la file d'attente : on ne remplit qu'une fraction
                    if random.random() < FILL_RATE_MAKER:
                        # On limite aussi par un volume arbitraire "moyen" pour ne pas abuser
                        exec_qty = min(qty_to_buy, 5) 
                        current_pos += exec_qty
                        cash[product] -= exec_qty * order.price

            # --- VENTE (SELL) ---
            elif order.quantity < 0:
                qty_to_sell = min(abs(order.quantity), current_pos + limit)
                if qty_to_sell <= 0: continue

                # 1. Logique TAKER (On tape dans les Bids)
                available_volume = 0
                execution_price = 0
                for bid_price, bid_vol in sorted(depth.buy_orders.items(), reverse=True):
                    if order.price <= bid_price:
                        available_volume += abs(bid_vol)
                        execution_price = bid_price
                        break
                
                if available_volume > 0:
                    exec_qty = min(qty_to_sell, available_volume)
                    current_pos -= exec_qty
                    cash[product] += exec_qty * execution_price

                # 2. Logique MAKER
                elif order.price <= min(depth.sell_orders.keys()):
                    if random.random() < FILL_RATE_MAKER:
                        exec_qty = min(qty_to_sell, 5)
                        current_pos -= exec_qty
                        cash[product] += exec_qty * order.price
                    
        positions[product] = current_pos

    return positions, cash

# =================================================================
#  3. BOUCLE PRINCIPALE DU BACKTESTER
# =================================================================
def run_backtest():
    # Chemins de tes fichiers
    base_path = "/home/jean-marie/imc prosperity 4/GSTD/ROUND1/"
    files = [
        base_path + 'prices_round_1_day_-2.csv',
        base_path + 'prices_round_1_day_-1.csv',
        base_path + 'prices_round_1_day_0.csv'
    ]

    print("Chargement des données historiques...")
    dfs = []
    for i, f in enumerate(files):
        if os.path.exists(f):
            df = pd.read_csv(f, sep=';')
            df['global_timestamp'] = df['timestamp'] + (i * 1000000)
            dfs.append(df)
        else:
            print(f"Fichier introuvable : {f}")
            return

    df = pd.concat(dfs)
    df = df[df['mid_price'].notna() & (df['mid_price'] > 0)] # Nettoyage de base
    
    # On groupe par timestamp global pour avoir l'état du marché à chaque tick
    grouped = df.groupby('global_timestamp')
    
    # Initialisation du Trader et de l'état
    trader = Trader()
    limits = trader.limits
    
    positions = {'INTARIAN_PEPPER_ROOT': 0, 'ASH_COATED_OSMIUM': 0}
    cash = {'INTARIAN_PEPPER_ROOT': 0.0, 'ASH_COATED_OSMIUM': 0.0}
    
    history_pnl = []
    history_time = []
    trader_data = ""

    print("Début de la simulation...")
    
    for timestamp, group in grouped:
        order_depths = {}
        mid_prices = {}
        
        # 1. Construction du carnet d'ordres pour ce tick
        for _, row in group.iterrows():
            product = row['product']
            depth = OrderDepth()
            
            # Récupération des 3 niveaux de prix (Bids et Asks)
            for i in range(1, 4):
                if pd.notna(row.get(f'bid_price_{i}')):
                    depth.buy_orders[int(row[f'bid_price_{i}'])] = int(row[f'bid_volume_{i}'])
                if pd.notna(row.get(f'ask_price_{i}')):
                    depth.sell_orders[int(row[f'ask_price_{i}'])] = -int(row[f'ask_volume_{i}'])
            
            order_depths[product] = depth
            mid_prices[product] = row['mid_price']

       # 2. Création de l'état (TradingState)
        # Note : On passe trader_data directement dans le constructeur ici
        state = TradingState(
            traderData=trader_data,
            timestamp=timestamp,
            listings={},
            order_depths=order_depths,
            own_trades={},
            market_trades={},
            position=positions.copy(),
            observations=Observation({}, {})
        )
        
        # Plus besoin de faire state.traderData = trader_data après, 
        # puisque c'est déjà fait dans le constructeur au-dessus.
        state.traderData = trader_data

        # 3. Exécution de ton Algorithme
        orders, conversions, trader_data = trader.run(state)

        # 4. Simulation du Matching Engine (Croisement des ordres)
        positions, cash = simulate_matching(orders, order_depths, positions, cash, limits)

        # 5. Calcul du PnL (Mark-to-Market : Cash + Valeur des positions au mid_price)
        total_pnl = 0
        for prod in positions.keys():
            if prod in mid_prices:
                # PnL = (Argent en caisse) + (Quantité possédée * Prix actuel)
                prod_pnl = cash[prod] + (positions[prod] * mid_prices[prod])
                total_pnl += prod_pnl
                
        history_pnl.append(total_pnl)
        history_time.append(timestamp)

    print(f"Simulation terminée ! PnL Final : {history_pnl[-1]:.2f} XIRECs")

    # =================================================================
    #  4. VISUALISATION DU RÉSULTAT
    # =================================================================
    plt.figure(figsize=(12, 6))
    plt.plot(history_time, history_pnl, color='green', linewidth=1.5)
    plt.title(f'Progression du PnL - Stratégie (Final: {history_pnl[-1]:.0f})')
    plt.xlabel('Timestamp Global')
    plt.ylabel('Profit (XIRECs)')
    plt.grid(True, alpha=0.3)
    
    # Création du dossier si nécessaire
    os.makedirs("visualisation", exist_ok=True)
    out_file = "visualisation/backtest_pnl.png"
    plt.savefig(out_file)
    print(f"Graphique sauvegardé dans : {out_file}")

# Lancement du script
if __name__ == "__main__":
    run_backtest()