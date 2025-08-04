# 🧠 Analyse de sentiment (Sentiment140)

Ce projet applique du Machine Learning pour détecter si un tweet est **positif** ou **négatif**.

---

## 🚀 Étapes principales

1. **Nettoyage des textes** avec NLTK (`preprocess_text.py`)
2. **Chargement + préparation des données** (`load_and_prepare_data.py`)
3. **Vectorisation** avec TF-IDF
4. **Entraînement** d’un modèle de régression logistique (`main.py`)
5. **Évaluation** et test manuel

---

## ▶️ Lancer le projet

```bash
git clone https://github.com/siwar-lassoued/sentiment-analysis.git
cd sentiment-analysis

python -m venv .venv
.venv\Scripts\activate

pip install -r requirements.txt
python main.py

