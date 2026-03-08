import pandas as pd
import joblib
import os

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

from utils.text_cleaner import clean_text


print("Loading dataset...")

df = pd.read_csv("data/amazon_reviews.csv")

# Convert column names
df.columns = df.columns.str.lower()

# Create text column
df["text"] = df["text"].astype(str)

# Convert score to label
df["label"] = df["score"].apply(lambda x: 1 if x >= 4 else 0)

print("Cleaning text...")

df["clean"] = df["text"].apply(clean_text)

print("Training model...")

vectorizer = TfidfVectorizer(stop_words="english")

X = vectorizer.fit_transform(df["clean"])
y = df["label"]

model = LogisticRegression(max_iter=1000)

model.fit(X, y)

os.makedirs("models", exist_ok=True)

joblib.dump(model, "models/sentiment_model.pkl")
joblib.dump(vectorizer, "models/vectorizer.pkl")

print("✅ Model saved successfully!")