import re
import nltk
import pandas as pd
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

#Télécharger les stopwords si nécessaire
nltk.download('punkt')
nltk.download('stopwords')

#Fonction de nettoyage
def clean_text(text):
    # Mettre en minuscules
    text = text.lower()
    # Supprimer les mentions @
    text = re.sub(r'@\w+', '', text)
    # Supprimer les liens
    text = re.sub(r'http\S+', '', text)
    # Supprimer la ponctuation et les chiffres
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    # Tokenisation simple sans nltk
    tokens = text.split()
    # Supprimer les stopwords
    stop_words = set(stopwords.words('english'))
    tokens = [word for word in tokens if word not in stop_words]
    return ' '.join(tokens)

