import pandas as pd
from sklearn.model_selection import KFold

df = pd.read_csv('../../data/music_mouv_data.csv')

# Filtrez les données pour ne garder que les lignes où l'émotion est 'joie' ou 'tension'
df_filtered = df[df['emotion'].isin(['Joyful Activation', 'Tension'])]

# Sélectionnez toutes les colonnes numériques sauf 'participant_id' et 'test_number'
x = df_filtered.drop(['emotion', 'participant_id'], axis=1)
y = df_filtered['emotion']

# Créez un objet KFold
kf = KFold(n_splits=5, shuffle=True, random_state=42)

folds = kf.split(x)

for train_index, test_index in folds:
    x_train, x_test = x.iloc[train_index], x.iloc[test_index]
    y_train, y_test = y.iloc[train_index], y.iloc[test_index]
    print('Taille de l\'ensemble d\'entraînement:', len(x_train))
    print('Taille de l\'ensemble de test:', len(x_test))
    print('---')
