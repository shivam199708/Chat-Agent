import json
import os
import pickle

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors
import numpy as np

def train_retrieval_agent(
    data_path="data/processed/dialog_pairs.jsonl",
    model_output_dir="models/"
):
    """
    Train a simple retrieval-based agent:
    - Vectorizes contexts using TF-IDF
    - Fits a Nearest Neighbor model
    - Saves model and vectorizer to disk
    """
    # Load data
    with open(data_path, "r", encoding="utf-8") as f:
        data = [json.loads(line) for line in f]

    contexts = [item["context"] for item in data]
    responses = [item["response"] for item in data]

    # Create output dir
    os.makedirs(model_output_dir, exist_ok=True)

    # Vectorize contexts
    vectorizer = TfidfVectorizer(ngram_range=(1,2), max_features=5000)
    context_vectors = vectorizer.fit_transform(contexts)

    # Fit Nearest Neighbors
    nn_model = NearestNeighbors(n_neighbors=1, metric="cosine")
    nn_model.fit(context_vectors)

    # Save everything
    with open(os.path.join(model_output_dir, "vectorizer.pkl"), "wb") as f:
        pickle.dump(vectorizer, f)
    with open(os.path.join(model_output_dir, "nn_model.pkl"), "wb") as f:
        pickle.dump(nn_model, f)
    with open(os.path.join(model_output_dir, "responses.pkl"), "wb") as f:
        pickle.dump(responses, f)

    print("✅ Retrieval agent trained and saved!")

if __name__ == "__main__":
    train_retrieval_agent()
