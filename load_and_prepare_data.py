import pandas as pd

# Charger le fichier
df = pd.read_csv("sentiment140.csv", encoding="latin-1", header=None)

#Garder uniquement les colonnes utiles : la colonne 0 (sentiment) et la 5 (texte)
df = df[[0,5]]

#Renommer les colonnes
df.columns = ['sentiment', 'text']

#Remplacer les veleurs 4 (positif) par 1
df['sentiment'] = df['sentiment'].replace(4, 1)

#Afficher les 5 premières lignes pour vérifier
print(df.head())