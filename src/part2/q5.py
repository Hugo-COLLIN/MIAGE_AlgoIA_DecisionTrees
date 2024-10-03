import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import KFold, cross_val_score
from sklearn.metrics import make_scorer, precision_score

# Chargement et préparation des données
df = pd.read_csv('../../data/music_mouv_data.csv')
df_filtered = df[df['emotion'].isin(['Joyful Activation', 'Tension'])]

# Listes de combinaisons de variables à tester
combinations = [
    ['EDA Phasic Number of Peaks', 'danceability', 'speechiness'],  # Combinaison initiale donnée
    ['EDA Phasic Number of Peaks', 'danceability', 'valence'],  # Essai avec "valence" au lieu de "speechiness"
    ['EDA Phasic Number of Peaks', 'acousticness', 'energy'],  # Essai avec d'autres variables musicales
    ['HRV', 'loudness', 'instrumentalness'],  # Combinaison avec des variables différentes
    ['EDA Tonic Mean', 'liveness', 'energy']  # Essai avec des variables moins corrélées
]

# Variable cible (émotion)
y = df_filtered['emotion']

# Création de l'objet KFold
kf = KFold(n_splits=5, shuffle=True, random_state=42)

# Définition du scorer pour la précision micro
micro_precision_scorer = make_scorer(precision_score, average='micro')

# Fonction pour tester une combinaison de variables
def test_combination(variables):
    print(f"Test de la combinaison : {variables}")

    # Filtrer les données pour ne conserver que les variables sélectionnées
    X_selected = df_filtered[variables]

    # Création du modèle Random Forest
    forest = RandomForestClassifier(n_estimators=100, criterion='entropy', random_state=42)

    # Validation croisée pour évaluer la précision micro
    scores = cross_val_score(forest, X_selected, y, cv=kf, scoring=micro_precision_scorer)

    # Affichage des résultats
    print(f"Précision micro moyenne : {scores.mean():.4f}")
    print(f"Écart-type de la précision micro : {scores.std():.4f}")
    print("\n")

# Tester chaque combinaison
for combination in combinations:
    test_combination(combination)
