import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('../data/music_mouv_data.csv')

df['song'] = df['artist_name'] + ' - ' + df['track_name']

song_counts = df['song'].value_counts()

appearance_counts = {1: 0, 2: 0, 3: 0}
for count in song_counts:
    if count <= 3:
        appearance_counts[count] += 1


plt.figure(figsize=(10, 6))
plt.bar(appearance_counts.keys(), appearance_counts.values())
plt.title('Distribution des apparitions des chansons')
plt.xlabel('Nombre d\'apparitions')
plt.ylabel('Nombre de chansons')
plt.xticks([1, 2, 3])
for i, v in appearance_counts.items():
    plt.text(i, v, str(v), ha='center', va='bottom')
plt.tight_layout()
plt.show()
