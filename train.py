import os
import pickle
import re
import warnings

import matplotlib.pyplot as plt
import nltk
import numpy as np
import pandas as pd
import seaborn as sns
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split

warnings.filterwarnings('ignore')

# Set random seed
np.random.seed(42)

# Download required NLTK data
nltk.download('stopwords', quiet=True)
nltk.download('wordnet', quiet=True)
nltk.download('punkt', quiet=True)

# Initialize lemmatizer
lemmatizer = WordNetLemmatizer()

def load_data():
    """Load and combine the fake and real news datasets."""
    print("Loading data...")
    
    # Load the datasets
    fake_news = pd.read_csv('Fake.csv')
    true_news = pd.read_csv('True.csv')

    # Add labels
    fake_news['label'] = 0  # FAKE = 0
    true_news['label'] = 1  # REAL = 1

    # Combine the datasets
    df = pd.concat([fake_news, true_news], axis=0)
    
    # Fill any NaN values
    df['title'].fillna('', inplace=True)
    df['text'].fillna('', inplace=True)
    
    # Combine title and text with more weight on the title
    df['text'] = df.apply(lambda x: f"{x['title']} {x['title']} {x['text']}", axis=1)

    # Shuffle
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)

    print(f"Total samples: {len(df)}")
    print(f"Fake news: {len(df[df['label'] == 0])}")
    print(f"Real news: {len(df[df['label'] == 1])}")
    
    return df

def preprocess_text(text):
    """Preprocess the text data."""
    if isinstance(text, str):
        text = text.lower()
        text = re.sub(r'[^a-zA-Z\s]', '', text)
        text = re.sub(r'\s+', ' ', text).strip()

        words = nltk.word_tokenize(text)
        stop_words = set(stopwords.words('english'))
        words = [lemmatizer.lemmatize(word) for word in words if word not in stop_words and len(word) > 2]

        return ' '.join(words)
    return ""

def prepare_data(df):
    """Prepare the data for training and testing."""
    print("Preprocessing text data...")
    df['processed_text'] = df['text'].apply(preprocess_text)
    
    # Remove empty processed texts
    df = df[df['processed_text'].str.strip() != '']
    
    X_train, X_test, y_train, y_test = train_test_split(
        df['processed_text'], df['label'], test_size=0.2, random_state=42, stratify=df['label']
    )
    return X_train, X_test, y_train, y_test

def train_model(X_train, X_test, y_train, y_test):
    """Train and evaluate the model."""
    print("Vectorizing text data...")
    vectorizer = TfidfVectorizer(
        max_features=20480,
        ngram_range=(1, 2),
        min_df=3,
        max_df=0.9
    )
    X_train_tfidf = vectorizer.fit_transform(X_train)
    X_test_tfidf = vectorizer.transform(X_test)

    print("Training logistic regression model...")
    model = LogisticRegression(
        C=1.0,
        max_iter=1000, 
        class_weight='balanced', 
        random_state=42,
        solver='liblinear'
    )
    model.fit(X_train_tfidf, y_train)

    y_pred = model.predict(X_test_tfidf)

    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    print("\nModel Performance:")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1-Score: {f1:.4f}")

    # Plot confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.savefig('confusion_matrix.png')
    plt.close()

    return model, vectorizer, accuracy

def save_model(model, vectorizer):
    """Save the trained model and vectorizer."""
    # Create a models directory if it doesn't exist
    if not os.path.exists('models'):
        os.makedirs('models')
    
    # Save the model and vectorizer
    print("Saving model and vectorizer...")
    with open('models/fake_news_model.pkl', 'wb') as f:
        pickle.dump(model, f)
    
    with open('models/tfidf_vectorizer.pkl', 'wb') as f:
        pickle.dump(vectorizer, f)
    
    print("Model and vectorizer saved successfully!")

def main():
    """Main function to run the fake news detection pipeline."""
    print("Starting fake news detection model training...")
    df = load_data()
    X_train, X_test, y_train, y_test = prepare_data(df)

    model, vectorizer, accuracy = train_model(X_train, X_test, y_train, y_test)
    save_model(model, vectorizer)

    print(f"\nModel training completed with accuracy: {accuracy:.4f}")
    print("The model has been saved for use with the Streamlit app.")

if __name__ == "__main__":
    main()