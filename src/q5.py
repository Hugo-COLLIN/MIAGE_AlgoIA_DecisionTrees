import pandas as pd

df = pd.read_csv('../data/music_mouv_data.csv')

# Filtrez les données pour ne garder que les lignes où l'émotion est 'joie' ou 'tension'
df_filtered = df[df['emotion'].isin(['Joyful Activation', 'Tension'])]

# Sélectionnez toutes les colonnes numériques sauf 'participant_id' et 'test_number'
X = df_filtered.drop(['emotion', 'participant_id'], axis=1)
y = df_filtered['emotion']
