import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('../data/music_mouv_data.csv')
df['song'] = df['artist_name'] + ' - ' + df['track_name']

# Identifier les chansons qui apparaissent trois fois
songs_3_times = df['song'].value_counts()[df['song'].value_counts() == 3].index

# Créer un graphique pour chaque chanson
fig, axes = plt.subplots(1, 3, figsize=(18, 6))
fig.suptitle('Distribution des émotions pour les chansons apparaissant 3 fois')

for i, song in enumerate(songs_3_times):
    song_data = df[df['song'] == song]
    emotion_counts = song_data['emotion'].value_counts()

    axes[i].scatter(emotion_counts.index, [1]*len(emotion_counts), s=emotion_counts.values*100, alpha=0.6)
    axes[i].set_title(song)
    axes[i].set_yticks([])
    axes[i].set_xlabel('Émotion')
    axes[i].tick_params(axis='x', rotation=45)

plt.tight_layout()
plt.show()
