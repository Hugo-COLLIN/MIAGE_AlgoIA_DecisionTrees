import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('../data/music_mouv_data.csv')

print(df.head())

emotion_counts = df['emotion'].value_counts()

plt.figure(figsize=(10, 6))
emotion_counts.plot(kind='bar')
plt.title('Distribution des émotions induites')
plt.xlabel('Émotion')
plt.ylabel('Nombre de passations')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()
