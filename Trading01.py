import os
import pandas as pd
import matplotlib.pyplot as plt
from typing import Dict, List
from datamodel import Order, TradingState, Symbol

# 1. Création du dossier de visualisation
output_dir = "visualisation"
if not os.path.exists(output_dir):
    os.makedirs(output_dir)
    print(f"Dossier '{output_dir}' créé.")

# 2. Liste des fichiers (réordonnés chronologiquement pour le graphique)
base_path = "/home/jean-marie/imc prosperity 4/ROUND1/"
files = [
    base_path + 'prices_round_1_day_-2.csv',
    base_path + 'prices_round_1_day_-1.csv',
    base_path + 'prices_round_1_day_0.csv'
]

# 3. Lecture et fusion des données
df_list = []
for i, f in enumerate(files):
    if os.path.exists(f):
        df = pd.read_csv(f, sep=';')
        # On ajuste le timestamp pour que les jours se suivent
        # Jour 0: 0-1M, Jour 1: 1M-2M, etc.
        df['global_timestamp'] = df['timestamp'] + (i * 1000000)
        df_list.append(df)
    else:
        print(f"Attention : le fichier {f} est introuvable.")

if df_list:
    full_df = pd.concat(df_list)

    # 4. Préparation du graphique
    osmium = full_df[full_df['product'] == 'ASH_COATED_OSMIUM']
    pepper = full_df[full_df['product'] == 'INTARIAN_PEPPER_ROOT']

    plt.figure(figsize=(15, 8))

    # Graphique pour l'Osmium
    plt.subplot(2, 1, 1)
    plt.plot(osmium['global_timestamp'], osmium['mid_price'], color='blue', label='Osmium Mid Price')
    plt.title('ASH_COATED_OSMIUM (Stable / Mean Reversion)')
    plt.ylabel('Prix')
    plt.grid(True, alpha=0.3)
    plt.legend()

    # Graphique pour le Pepper
    plt.subplot(2, 1, 2)
    plt.plot(pepper['global_timestamp'], pepper['mid_price'], color='orange', label='Pepper Mid Price')
    plt.title('INTARIAN_PEPPER_ROOT (Volatile / Momentum)')
    plt.xlabel('Timestamp Global (ms)')
    plt.ylabel('Prix')
    plt.grid(True, alpha=0.3)
    plt.legend()

    # 5. Sauvegarde au lieu de l'affichage interactif
    file_name = os.path.join(output_dir, "analyse_prix_round1.png")
    plt.tight_layout()
    plt.savefig(file_name)
    plt.close() # Libère la mémoire
    print(f"Graphique sauvegardé avec succès dans : {file_name}")

# --- Ta classe Trader pour IMC ---
class Trader:
    def run(self, state: TradingState) -> Dict[Symbol, List[Order]]:
        result = {}
        for product in state.order_depths:
            order_depth = state.order_depths[product]
            orders: List[Order] = []
            
            if product == "ASH_COATED_OSMIUM":
                acceptable_price = 10000 
                # Ta future logique ici
                
            result[product] = orders
        return result