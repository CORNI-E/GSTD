import os
import pandas as pd
import matplotlib.pyplot as plt
from typing import Dict, List
from datamodel import Order, TradingState, Symbol, OrderDepth, Listing, Observation

# =================================================================
#  1. IMPORTE TON ALGORITHME ICI (OU COLLE LA CLASSE TRADER)
# =================================================================
# Si ton algorithme est dans un fichier `algo.py`, décommente la ligne suivante :
from trader_round1 import Trader 

# (Pour ce test, je pars du principe que ta classe Trader est déjà définie plus haut dans ton script)

# =================================================================
#  2. MOTEUR DE SIMULATION (MATCHING ENGINE)
# =================================================================
def simulate_matching(orders: Dict[Symbol, List[Order]], order_depths: Dict[Symbol, OrderDepth], 
                      positions: Dict[Symbol, int], cash: Dict[Symbol, float], limits: Dict[Symbol, int]):
    """
    Simule l'exécution des ordres en les croisant avec le carnet d'ordres actuel.
    """
    for product, product_orders in orders.items():
        if product not in order_depths:
            continue
            
        depth = order_depths[product]
        current_pos = positions.get(product, 0)
        
        for order in product_orders:
            # --- ORDRE D'ACHAT (BUY) ---
            if order.quantity > 0:
                # On cherche les vendeurs (asks)
                # Trier par prix croissant (on achète le moins cher d'abord)
                asks = sorted(depth.sell_orders.items())
                qty_to_buy = order.quantity
                
                # Vérification de la limite de position
                max_allowed = limits[product] - current_pos
                qty_to_buy = min(qty_to_buy, max_allowed)
                
                for ask_price, ask_vol in asks:
                    if qty_to_buy <= 0:
                        break
                    if order.price >= ask_price: # Notre prix d'achat croise le prix de vente
                        vol_available = abs(ask_vol) # Les volumes de vente sont négatifs dans le datamodel
                        executed_qty = min(qty_to_buy, vol_available)
                        
                        current_pos += executed_qty
                        cash[product] -= executed_qty * ask_price
                        qty_to_buy -= executed_qty
                        
                        # On retire le volume du carnet pour ne pas le réutiliser
                        depth.sell_orders[ask_price] -= -executed_qty
            
            # --- ORDRE DE VENTE (SELL) ---
            elif order.quantity < 0:
                # On cherche les acheteurs (bids)
                # Trier par prix décroissant (on vend au plus cher d'abord)
                bids = sorted(depth.buy_orders.items(), reverse=True)
                qty_to_sell = abs(order.quantity)
                
                # Vérification de la limite de position
                max_allowed = current_pos + limits[product]
                qty_to_sell = min(qty_to_sell, max_allowed)
                
                for bid_price, bid_vol in bids:
                    if qty_to_sell <= 0:
                        break
                    if order.price <= bid_price: # Notre prix de vente croise le prix d'achat
                        executed_qty = min(qty_to_sell, bid_vol)
                        
                        current_pos -= executed_qty
                        cash[product] += executed_qty * bid_price
                        qty_to_sell -= executed_qty
                        
                        depth.buy_orders[bid_price] -= executed_qty
                        
        # Mise à jour de la position finale après traitement de tous les ordres du produit
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