# 🧠 Emotion Detection - Logistic Regression (Bag of Words)

This is a Machine Learning web app that detects the **emotion behind a sentence** using a trained **Logistic Regression** model built with the **Bag of Words** technique.

## 🔍 Project Overview

The model is trained on a labeled text dataset containing various emotions. It performs text preprocessing, vectorization using `CountVectorizer` (Bag of Words), and classification using Logistic Regression. The app is deployed using **Streamlit** for easy interaction.

## ✨ Features

- Predicts emotions like `joy`, `sadness`, `anger`, etc.
- Preprocessing includes stopword removal, lemmatization, punctuation cleaning, and emoji removal.
- Uses **Bag of Words** (unigrams + bigrams).
- Supports real-time predictions through a web interface.

## 🚀 How to Run

### 1. Clone the repository

```bash
git clone https://github.com/codewithyasho/emotion-detection.git
cd emotion-detection
```

## ✅ Technologies Used
Python

Scikit-learn

NLTK

Streamlit

Pandas, NumPy

CountVectorizer (Bag of Words)
