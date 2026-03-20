import pandas as pd
import numpy as np
import pickle
import os
import requests
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.metrics import accuracy_score, confusion_matrix

# Configuration
DATASET_PATH = 'dataset/news.csv'
DATASET_URL = 'https://raw.githubusercontent.com/lutzhamel/fake-news/master/data/fake_or_real_news.csv'
MODEL_PATH = 'model.pkl'
VECTORIZER_PATH = 'vectorizer.pkl'

def download_dataset():
    if not os.path.exists(DATASET_PATH):
        print(f"Downloading dataset from {DATASET_URL}...")
        os.makedirs(os.path.dirname(DATASET_PATH), exist_ok=True)
        response = requests.get(DATASET_URL)
        with open(DATASET_PATH, 'wb') as f:
            f.write(response.content)
        print("Dataset downloaded successfully.")
    else:
        print("Dataset already exists.")

def train():
    download_dataset()
    
    # Load dataset
    print("Loading dataset...")
    df = pd.read_csv(DATASET_PATH)
    
    # Preprocessing
    # The dataset has 'title', 'text', and 'label' (FAKE/REAL)
    # We combine title and text for better performance
    df['content'] = df['title'] + " " + df['text']
    
    # Labels
    labels = df.label
    
    # Split the dataset
    print("Splitting dataset into training and testing sets...")
    x_train, x_test, y_train, y_test = train_test_split(df['content'], labels, test_size=0.2, random_state=7)
    
    # Initialize TF-IDF Vectorizer
    print("Initializing TF-IDF Vectorizer...")
    tfidf_vectorizer = TfidfVectorizer(stop_words='english', max_df=0.7)
    
    # Fit and transform train set, transform test set
    print("Fitting and transforming data...")
    tfidf_train = tfidf_vectorizer.fit_transform(x_train)
    tfidf_test = tfidf_vectorizer.transform(x_test)
    
    # Initialize PassiveAggressiveClassifier
    print("Training Passive Aggressive Classifier...")
    pac = PassiveAggressiveClassifier(max_iter=50)
    pac.fit(tfidf_train, y_train)
    
    # Predict on test set and calculate accuracy
    y_pred = pac.predict(tfidf_test)
    score = accuracy_score(y_test, y_pred)
    print(f'Accuracy: {round(score*100, 2)}%')
    
    # Save the model and vectorizer
    print(f"Saving model to {MODEL_PATH} and vectorizer to {VECTORIZER_PATH}...")
    with open(MODEL_PATH, 'wb') as f:
        pickle.dump(pac, f)
    with open(VECTORIZER_PATH, 'wb') as f:
        pickle.dump(tfidf_vectorizer, f)
    
    print("Training complete and artifacts saved.")

if __name__ == "__main__":
    train()
