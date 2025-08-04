import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

# Téléchargement des ressources NLTK
nltk.download('stopwords')

# Fonction de nettoyage (version sans word_tokenize pour éviter les bugs)
def clean_text(text):
    text = text.lower()
    text = re.sub(r'@\w+', '', text)
    text = re.sub(r'http\S+', '', text)
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    tokens = text.split()
    stop_words = set(stopwords.words('english'))
    tokens = [word for word in tokens if word not in stop_words]
    return ' '.join(tokens)

#  Chargement du dataset
df = pd.read_csv("sentiment140.csv", encoding='latin-1', header=None)
df.columns = ['sentiment', 'id', 'date', 'query', 'user', 'text']
df = df[['sentiment', 'text']]

#  Filtrage et nettoyage
df = df[df['sentiment'].isin([0, 4])]
df['label'] = df['sentiment'].apply(lambda x: 1 if x == 4 else 0)
df['text'] = df['text'].fillna('')
df['clean_text'] = df['text'].apply(clean_text)

#  Échantillon rapide pour test
df = df.sample(10000, random_state=42)

#  Vectorisation TF-IDF
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(df['clean_text'])
y = df['label']

#  Train/Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

#  Modèle
model = LogisticRegression()
model.fit(X_train, y_train)

#  Évaluation
y_pred = model.predict(X_test)
print("\n Classification Report:")
print(classification_report(y_test, y_pred))

#  Prédiction test
new_text = "I like it"
cleaned = clean_text(new_text)
vectorized = vectorizer.transform([cleaned])
prediction = model.predict(vectorized)[0]

print(f"\n Phrase test : {new_text}")
print(f" Prédiction : {'Positive' if prediction == 1 else 'Negative'}")
