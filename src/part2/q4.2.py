import pandas as pd
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import KFold, cross_val_score
from sklearn.metrics import make_scorer, precision_score
import matplotlib.pyplot as plt

# Chargement et préparation des données
df = pd.read_csv('../../data/music_mouv_data.csv')
df_filtered = df[df['emotion'].isin(['Joyful Activation', 'Tension'])]

# Filtrage pour ne conserver que les variables musicales
X_music = df_filtered[['danceability','energy','key','loudness','mode','speechiness','accousticness','instrumentalness','liveness','valence','tempo']]

# Dépendante variable (émotion)
y = df_filtered['emotion']

# Création de l'objet KFold
kf = KFold(n_splits=5, shuffle=True, random_state=42)

# Scorer pour la précision micro
micro_precision_scorer = make_scorer(precision_score, average='micro')

### 1. Entraînement et évaluation avec un arbre de décision ###

# Création du modèle d'arbre de décision
tree = DecisionTreeClassifier(criterion='entropy', random_state=42)

# Validation croisée et affichage de l'arbre pour le premier pli
fold = 1
for train_index, test_index in kf.split(X_music):
    X_train, X_test = X_music.iloc[train_index], X_music.iloc[test_index]
    y_train, y_test = y.iloc[train_index], y.iloc[test_index]

    # Entraînement du modèle
    tree.fit(X_train, y_train)

    if fold == 1:  # Affichage de l'arbre du premier pli
        plt.figure(figsize=(20,10))
        plot_tree(tree, feature_names=X_music.columns, class_names=tree.classes_, filled=True, rounded=True)
        plt.title(f"Arbre de décision pour le pli {fold} (variables musicales)")
        plt.show()

    fold += 1

# Validation croisée pour évaluer la précision micro
scores = cross_val_score(tree, X_music, y, cv=kf, scoring=micro_precision_scorer)

# Affichage des résultats pour l'arbre de décision
print("### Arbre de décision (variables musicales) ###")
print("Scores de précision micro pour chaque fold :")
for i, score in enumerate(scores):
    print(f"Fold {i+1}: {score:.4f}")
print(f"\nPrécision micro moyenne : {scores.mean():.4f}")
print(f"Écart-type de la précision micro : {scores.std():.4f}")


### 2. Entraînement et évaluation avec un Random Forest ###

# Création du modèle Random Forest
forest = RandomForestClassifier(n_estimators=100, criterion='entropy', random_state=42)

# Validation croisée et affichage d'un arbre de la forêt pour le premier pli
fold = 1
for train_index, test_index in kf.split(X_music):
    X_train, X_test = X_music.iloc[train_index], X_music.iloc[test_index]
    y_train, y_test = y.iloc[train_index], y.iloc[test_index]

    # Entraînement du modèle
    forest.fit(X_train, y_train)

    if fold == 1:  # Visualisation d'un arbre du premier pli
        plt.figure(figsize=(20,10))
        plot_tree(forest.estimators_[0], feature_names=X_music.columns, class_names=forest.classes_, filled=True, rounded=True)
        plt.title(f"Un arbre de la forêt pour le pli {fold} (variables musicales)")
        plt.show()

    fold += 1

# Validation croisée pour évaluer la précision micro
scores = cross_val_score(forest, X_music, y, cv=kf, scoring=micro_precision_scorer)

# Affichage des résultats pour le Random Forest
print("### Random Forest (variables musicales) ###")
print("Scores de précision micro pour chaque fold :")
for i, score in enumerate(scores):
    print(f"Fold {i+1}: {score:.4f}")
print(f"\nPrécision micro moyenne : {scores.mean():.4f}")
print(f"Écart-type de la précision micro : {scores.std():.4f}")
