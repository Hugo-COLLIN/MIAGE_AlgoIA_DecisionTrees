import pandas as pd
from sklearn.tree import DecisionTreeClassifier, plot_tree
import matplotlib.pyplot as plt

# Chargement et préparation des données
df = pd.read_csv('../../data/music_mouv_data.csv')
df_filtered = df[df['emotion'].isin(['Joyful Activation', 'Tension'])]

# Identifier les colonnes non numériques et les encoder
X = df_filtered.drop(['emotion', 'participant_id'], axis=1)

# Si certaines colonnes contiennent des chaînes, il faut les encoder
X_encoded = pd.get_dummies(X, drop_first=True)
y = df_filtered['emotion']

# Création de l'arbre de décision
tree = DecisionTreeClassifier(max_depth=3, criterion='entropy', random_state=42)
tree.fit(X_encoded, y)

# Visualisation de l'arbre
plt.figure(figsize=(20,10))
plot_tree(tree, feature_names=X_encoded.columns, class_names=tree.classes_, filled=True, rounded=True)
plt.title("Arbre de décision (profondeur 3, critère d'entropie)")
plt.show()
