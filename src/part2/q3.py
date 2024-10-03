import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import plot_tree
from sklearn.model_selection import KFold, cross_val_score
from sklearn.metrics import make_scorer, precision_score
import matplotlib.pyplot as plt

# Chargement et préparation des données
df = pd.read_csv('../../data/music_mouv_data.csv')
df_filtered = df[df['emotion'].isin(['Joyful Activation', 'Tension'])]
X = df_filtered.drop(['emotion', 'participant_id'], axis=1)

# Si certaines colonnes contiennent des chaînes, il faut les encoder
X_encoded = pd.get_dummies(X, drop_first=True)

y = df_filtered['emotion']

# Création de l'objet KFold
kf = KFold(n_splits=5, shuffle=True, random_state=42)

# Création du modèle Random Forest
forest = RandomForestClassifier(n_estimators=100, criterion='entropy', random_state=42)

# Définition du scorer pour la précision micro
micro_precision_scorer = make_scorer(precision_score, average='micro')

# Validation croisée et affichage d'un arbre de la forêt
fold = 1
for train_index, test_index in kf.split(X_encoded):
    # Séparation des données d'entraînement et de test pour ce pli
    X_train, X_test = X_encoded.iloc[train_index], X_encoded.iloc[test_index]
    y_train, y_test = y.iloc[train_index], y.iloc[test_index]

    # Entraînement du Random Forest
    forest.fit(X_train, y_train)

    if fold == 1:  # Visualisation d'un arbre du premier pli
        plt.figure(figsize=(20,10))
        # Visualisation du premier arbre de la forêt (l'indice [0])
        plot_tree(forest.estimators_[0], feature_names=X_encoded.columns, class_names=forest.classes_, filled=True, rounded=True)
        plt.title(f"Un arbre de la forêt pour le pli {fold}")
        plt.show()

    fold += 1

# Validation croisée pour évaluer la précision micro
scores = cross_val_score(forest, X_encoded, y, cv=kf, scoring=micro_precision_scorer)

# Affichage des résultats
print("Scores de précision micro pour chaque fold :")
for i, score in enumerate(scores):
    print(f"Fold {i+1}: {score:.4f}")
print(f"\nPrécision micro moyenne : {scores.mean():.4f}")
print(f"Écart-type de la précision micro : {scores.std():.4f}")
