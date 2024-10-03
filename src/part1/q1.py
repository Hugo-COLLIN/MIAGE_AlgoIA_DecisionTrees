import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('../../data/music_mouv_data.csv')

print(df.head())

# Compter le nombre d'occurrences de chaque émotion
emotion_counts = df['emotion'].value_counts()

# Créer un histogramme avec matplotlib
plt.figure(figsize=(10, 6))
emotion_counts.plot(kind='bar')
plt.title('Distribution des émotions induites')
plt.xlabel('Émotion')
plt.ylabel('Nombre de passations')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()
