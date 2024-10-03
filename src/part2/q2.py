import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import KFold, cross_val_score
from sklearn.metrics import make_scorer, precision_score

# Chargement et préparation des données
df = pd.read_csv('../../data/music_mouv_data.csv')
df_filtered = df[df['emotion'].isin(['Joyful Activation', 'Tension'])]
X = df_filtered.drop(['emotion', 'participant_id'], axis=1)

# Si certaines colonnes contiennent des chaînes, il faut les encoder
X_encoded = pd.get_dummies(X, drop_first=True)

y = df_filtered['emotion']

# Création de l'objet KFold
kf = KFold(n_splits=5, shuffle=True, random_state=42)

# Création du modèle d'arbre de décision
tree = DecisionTreeClassifier(criterion='entropy', random_state=42)

# Définition du scorer pour la précision micro
micro_precision_scorer = make_scorer(precision_score, average='micro')

# Validation croisée
scores = cross_val_score(tree, X_encoded, y, cv=kf, scoring=micro_precision_scorer)

# Affichage des résultats
print("Scores de précision micro pour chaque fold :")
for i, score in enumerate(scores):
    print(f"Fold {i+1}: {score:.4f}")
print(f"\nPrécision micro moyenne : {scores.mean():.4f}")
print(f"Écart-type de la précision micro : {scores.std():.4f}")
