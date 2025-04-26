import streamlit as st

# 🚨 Moved to top: st.set_page_config must be FIRST Streamlit command
st.set_page_config(page_title="Fake News Detector", page_icon="📰", layout="wide")

import pickle
import re

import nltk
import numpy as np
import pandas as pd
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer


# Setup
@st.cache_resource
def load_nltk_resources():
    nltk.download("stopwords", quiet=True)
    nltk.download("wordnet", quiet=True)
    nltk.download("punkt", quiet=True)
    return WordNetLemmatizer(), set(stopwords.words("english"))


lemmatizer, stop_words = load_nltk_resources()


# Load the model and vectorizer
@st.cache_resource
def load_model():
    try:
        with open("models/fake_news_model.pkl", "rb") as f:
            model = pickle.load(f)

        with open("models/tfidf_vectorizer.pkl", "rb") as f:
            vectorizer = pickle.load(f)

        return model, vectorizer
    except FileNotFoundError:
        st.error("Model files not found. Please run the training script first.")
        return None, None


model, vectorizer = load_model()


def preprocess_text(text):
    """Preprocess the text data."""
    text = text.lower()
    text = re.sub(r"[^a-zA-Z\s]", "", text)
    text = re.sub(r"\s+", " ", text).strip()

    words = nltk.word_tokenize(text)
    words = [lemmatizer.lemmatize(word) for word in words if word not in stop_words]

    return " ".join(words)


def predict_news(title, text, model, vectorizer):
    """Predict whether a news article is real or fake."""
    if not title and not text:
        return None, None

    # Combine title and text
    full_text = title + " " + text

    # Preprocess text
    processed_text = preprocess_text(full_text)

    # Vectorize the text
    X = vectorizer.transform([processed_text])

    # Make prediction
    prediction = model.predict(X)[0]
    proba = model.predict_proba(X)[0]

    # Return prediction and confidence
    return "REAL" if prediction == 1 else "FAKE", proba


# Streamlit App
st.title("📰 Fake News Detector")
st.markdown(
    """
This application uses machine learning to predict whether a news article is real or fake.
Enter the news article title and content below to get a prediction.
"""
)

# Input fields

st.subheader("Input News Information")
news_title = st.text_input("News Title", placeholder="Enter the news title here...")
news_content = st.text_area(
    "News Content", height=300, placeholder="Enter the news content here..."
)
if st.button("Predict", type="primary"):
    if model is None or vectorizer is None:
        st.error("Model files not found. Please run the training script first.")
    elif not news_title and not news_content:
        st.error("Please enter a news title or content to make a prediction.")
    else:
        with st.spinner("Analyzing..."):
            result, probabilities = predict_news(
                news_title, news_content, model, vectorizer
            )
            if result == "FAKE":
                fake_prob = probabilities[0]
                st.error(f"Prediction: **{result}**")
                st.progress(fake_prob, text=f"Confidence: {fake_prob:.2%}")
            else:
                real_prob = probabilities[1]
                st.success(f"Prediction: **{result}**")
                st.progress(real_prob, text=f"Confidence: {real_prob:.2%}")

st.markdown("---")
st.markdown("### About")
st.markdown(
    """
This fake news detector was built using:
- **Data**: A dataset of labeled real and fake news articles
- **Model**: Logistic Regression with TF-IDF vectorization
- **Frontend**: Streamlit
"""
)
