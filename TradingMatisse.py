import os
import pandas as pd
import matplotlib.pyplot as plt
from typing import Dict, List
from datamodel import Order, TradingState, Symbol

# 1. Configuration et Dossiers
output_dir = "visualisation"
os.makedirs(output_dir, exist_ok=True)

# Vérifie bien que ce chemin est exact sur ton ordinateur !
base_path = "/home/jean-marie/imc prosperity 4/GSTD/ROUND1/"
files = [
    base_path + 'prices_round_1_day_-2.csv',
    base_path + 'prices_round_1_day_-1.csv',
    base_path + 'prices_round_1_day_0.csv'
]

# 2. Chargement des données brutes
df_list = []
for i, f in enumerate(files):
    if os.path.exists(f):
        print(f"Chargement du fichier : {f}")
        df = pd.read_csv(f, sep=';')
        # On ajuste le timestamp pour la continuité
        df['global_timestamp'] = df['timestamp'] + (i * 1000000)
        df_list.append(df)
    else:
        print(f"ERREUR : Fichier introuvable -> {f}")

if not df_list:
    print("CRITIQUE : Aucun fichier n'a été chargé. Vérifie le chemin 'base_path'.")
else:
    full_df = pd.concat(df_list)
    total_initial = len(full_df)

    # 3. Nettoyage et Statistiques
    # On identifie les NaNs et les 0
    mask_nan = full_df['mid_price'].isna()
    mask_zero = (full_df['mid_price'] == 0)

    count_nan = mask_nan.sum()
    count_zero = mask_zero.sum()

    # Calcul des pourcentages
    pct_nan = (count_nan / total_initial) * 100
    pct_zero = (count_zero / total_initial) * 100

    print(f"\n--- Rapport de Nettoyage ---")
    print(f"Total lignes brutes : {total_initial}")
    print(f"Lignes avec NaN supprimées : {count_nan} ({pct_nan:.2f}%)")
    print(f"Lignes avec 0 supprimées   : {count_zero} ({pct_zero:.2f}%)")

    # Création du DataFrame nettoyé
    # On ne garde que ce qui n'est ni NaN ni 0
    clean_df = full_df[~mask_nan & ~mask_zero].copy()
    
    print(f"Total lignes après nettoyage : {len(clean_df)}")
    print(f"----------------------------\n")

    # 4. Visualisation
    osmium = clean_df[clean_df['product'] == 'ASH_COATED_OSMIUM']
    pepper = clean_df[clean_df['product'] == 'INTARIAN_PEPPER_ROOT']

    # On crée une figure avec deux sous-graphiques
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 10))

    # Graphique Osmium (Zoomé sur les micro-oscillations)
    ax1.plot(osmium['global_timestamp'], osmium['mid_price'], color='blue', linewidth=0.5)
    ax1.set_ylim(9950, 10050) # Zoom très serré pour voir les "dents"
    ax1.set_title('ASH_COATED_OSMIUM : Micro-oscillations (Zoom 9995-10005)')
    ax1.grid(True, alpha=0.3)

    # Calcul de l'oscillation pour Pepper (Prix - Moyenne Mobile)
    # On utilise une fenêtre de 20 pour capturer les mouvements rapides
    pepper['rolling_mean'] = pepper['mid_price'].rolling(window=20).mean()
    pepper['oscillation'] = pepper['mid_price'] - pepper['rolling_mean']

    # Graphique Pepper (On affiche l'écart à la moyenne)
    ax2.plot(pepper['global_timestamp'], pepper['oscillation'], color='orange', linewidth=0.5)
    # On zoome sur les oscillations de ±20 seashells
    ax2.set_ylim(-20, 20) 
    ax2.set_title('INTARIAN_PEPPER_ROOT : Oscillations relatives (Écart à la Moyenne Mobile)')
    ax2.set_xlabel('Temps (ms)')
    ax2.set_ylabel('Écart (Seashells)')
    ax2.axhline(0, color='black', linestyle='--', linewidth=1) # Ligne du "zéro"
    ax2.grid(True, alpha=0.3)

    # 5. Sauvegarde forcée
    output_path = os.path.join(output_dir, "analyse_prix_clean.png")
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    
    # Vérification finale
    if os.path.exists(output_path):
        print(f"SUCCÈS : L'image a été générée ici -> {os.path.abspath(output_path)}")
    else:
        print("ERREUR : L'image n'a pas pu être sauvegardée.")
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