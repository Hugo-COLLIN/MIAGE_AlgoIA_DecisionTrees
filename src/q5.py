import pandas as pd
from sklearn.preprocessing import LabelEncoder

df = pd.read_csv('../data/music_mouv_data.csv')

# Filtrez les données pour ne garder que les lignes où l'émotion est 'joie' ou 'tension'
df_filtered = df[df['emotion'].isin(['Joyful Activation', 'Tension'])]

# Sélectionnez toutes les colonnes numériques sauf 'participant_id' et 'test_number'
X = df_filtered.select_dtypes(include=['float64', 'int64']).drop(['participant_id'], axis=1)

# Encodez les émotions en valeurs numériques
le = LabelEncoder()
y = le.fit_transform(df_filtered['emotion'])

print("Dimensions de X:", X.shape)
print("Dimensions de y:", y.shape)
