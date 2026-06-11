!pip install kagglehub -q
!pip install Sastrawi -q
!pip install scikit-learn -q

import pandas as pd
import numpy as np
import re
import string
import os
import kagglehub
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory

from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, precision_score, recall_score, f1_score

from sklearn.pipeline import Pipeline
from sklearn.base import BaseEstimator, TransformerMixin

path = kagglehub.dataset_download("satyaahb/ecommerce-ratings-and-reviews-in-bahasa-indonesia")
print("Dataset path:", path)

files = os.listdir(path)
print("Files:", files)

df = pd.read_csv(os.path.join(path, 'raw_review_googleplay.csv'))
df.head()

path = kagglehub.dataset_download("satyaahb/ecommerce-ratings-and-reviews-in-bahasa-indonesia")
print("Dataset path:", path)

files = os.listdir(path)
print("Files:", files)

df = pd.read_csv(os.path.join(path, 'raw_review_googleplay.csv'))
df.head()

from sklearn.utils import resample

df_pos = df[df['sentiment'] == 'positif']
df_neg = df[df['sentiment'] == 'negatif']

df_pos_down = resample(df_pos,
                       replace=False,
                       n_samples=len(df_neg),
                       random_state=42)


df_balanced = pd.concat([df_pos_down, df_neg]).sample(frac=1, random_state=42)

print(df_balanced['sentiment'].value_counts())

factory = StemmerFactory()
stemmer = factory.create_stemmer()

stop_factory = StopWordRemoverFactory()
stopwords = stop_factory.get_stop_words()

class TextPreprocessor(BaseEstimator, TransformerMixin):
    def __init__(self, stopwords=None, stemmer=None):
        self.stopwords = stopwords
        self.stemmer = stemmer

  def fit(self, X, y=None):
        return self

  def transform(self, X):
        return X.apply(self._preprocess)

  def _preprocess(self, text):
        text = str(text).lower()

  emoji_pattern = re.compile(
            "["
            "\U0001F600-\U0001F64F"
            "\U0001F300-\U0001F5FF"
            "\U0001F680-\U0001F6FF"
            "\U0001F1E0-\U0001F1FF"
            "]+", flags=re.UNICODE)
        text = emoji_pattern.sub(r'', text)

  text = re.sub(r'\d+', '', text)

  text = text.translate(str.maketrans('', '', string.punctuation))

  text = re.sub(r'(.)\1{2,}', r'\1', text)

  text = text.strip()

  tokens = text.split()

  if self.stopwords:
      tokens = [w for w in tokens if w not in self.stopwords]

  if self.stemmer:
      tokens = [self.stemmer.stem(w) for w in tokens]

  return ' '.join(tokens)

X = df['content']
y = df['sentiment']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print("Train class distribusi:\n", y_train.value_counts())
print("Test class distribusi:\n", y_test.value_counts())

pipeline = Pipeline([
    ('preprocess', TextPreprocessor(stopwords=stopwords, stemmer=stemmer)),
    ('tfidf', TfidfVectorizer(ngram_range=(1,2), max_features=5000)),
    ('nb', MultinomialNB())
])

pipeline.fit(X_train, y_train)

y_pred = pipeline.predict(X_test)

print("Accuracy:", accuracy_score(y_test, y_pred))

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(6,5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.show()

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
cv_scores = cross_val_score(pipeline, X, y, cv=cv, scoring='accuracy')

print("Cross Validation Accuracy per fold:", cv_scores)
print("Mean CV Accuracy:", cv_scores.mean())

review_baru = [
    "produk bagus banget dan pengiriman cepat",
    "barang rusak kualitas jelek"
]

prediksi = pipeline.predict(pd.Series(review_baru))

for r, p in zip(review_baru, prediksi):
    print("Review:", r)
    print("Sentimen:", p)
