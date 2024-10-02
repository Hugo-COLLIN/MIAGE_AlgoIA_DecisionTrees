import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

df = pd.read_csv('../data/music_mouv_data.csv')

# Sélectionner les colonnes quantitatives (caractéristiques physiologiques et musicales)
features = df.select_dtypes(include=['float64', 'int64']).columns

# Calculer la matrice de corrélation de Spearman
corr_matrix = df[features].corr(method='spearman')

# Créer la carte thermique avec seaborn
plt.figure(figsize=(20, 16))
sns.heatmap(corr_matrix, annot=False, cmap='coolwarm', vmin=-1, vmax=1, center=0)
plt.title('Corrélations de Spearman entre les caractéristiques physiologiques et musicales')
plt.tight_layout()
plt.show()
