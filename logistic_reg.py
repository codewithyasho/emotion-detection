# ✅ Optimized Logistic Regression for Emotion Classification using Bag of Words

import pandas as pd
import numpy as np
import re
import string
import emoji
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report
import joblib

nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('omw-1.4')

# 🔹 Load data
df = pd.read_csv("train.txt", sep=";", header=None, names=["text", "label"])
df.drop_duplicates(inplace=True)

# 🔹 Preprocessing functions


def clean_text(text):
    text = text.lower()
    text = re.sub(r'http\S+', '', text)
    text = re.sub(r'\d+', '', text)
    text = re.sub(r'[%s]' % re.escape(string.punctuation), '', text)
    text = emoji.replace_emoji(text, '')
    return text


stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()


def preprocess(text):
    text = clean_text(text)
    tokens = text.split()
    tokens = [lemmatizer.lemmatize(w) for w in tokens if w not in stop_words]
    return " ".join(tokens)


# Apply preprocessing
df["clean_text"] = df["text"].apply(preprocess)

# Encode labels
le = LabelEncoder()
df["label_encoded"] = le.fit_transform(df["label"])

# 🔹 Bag of Words vectorization with n-grams
bow = CountVectorizer(ngram_range=(1, 2), max_features=10000)
X = bow.fit_transform(df["clean_text"])
y = df["label_encoded"]

# 🔹 Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y)

# 🔹 Logistic Regression with Hyperparameter Tuning
params = {
    "C": [0.1, 1, 10],
    "penalty": ["l2"]
}

grid = GridSearchCV(
    LogisticRegression(max_iter=1000, random_state=42, n_jobs=-1),
    param_grid=params, cv=5, verbose=1)

grid.fit(X_train, y_train)

# 🔹 Evaluation
y_pred = grid.predict(X_test)
print("Best Parameters:", grid.best_params_)
print("\nClassification Report:\n", classification_report(
    y_test, y_pred, target_names=le.classes_))

print("Accuracy:", grid.score(X_test, y_test))

# 🔹 Save model and vectorizer
joblib.dump(grid.best_estimator_, "optimized_logistic_model.pkl")
joblib.dump(bow, "bow_vectorizer.pkl")
joblib.dump(le, "label_encoder.pkl")
print("\n✅ Model, vectorizer and label encoder saved!")
