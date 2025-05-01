import json
import os
import pickle
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.neighbors import NearestNeighbors

def train_semantic_agent(
    data_path="data/processed/dialog_pairs.jsonl",
    model_output_dir="models/"
):
    # Load data
    with open(data_path, "r", encoding="utf-8") as f:
        data = [json.loads(line) for line in f]

    contexts = [item["context"] for item in data]
    responses = [item["response"] for item in data]

    # Create output dir
    os.makedirs(model_output_dir, exist_ok=True)

    # Load sentence transformer
    model = SentenceTransformer("all-MiniLM-L6-v2")
    context_vectors = model.encode(contexts, show_progress_bar=True, normalize_embeddings=True)

    # Fit Nearest Neighbor
    nn_model = NearestNeighbors(n_neighbors=1, metric="cosine")
    nn_model.fit(context_vectors)

    # Save all artifacts
    with open(os.path.join(model_output_dir, "sentence_model.pkl"), "wb") as f:
        pickle.dump(model, f)
    with open(os.path.join(model_output_dir, "nn_model.pkl"), "wb") as f:
        pickle.dump(nn_model, f)
    with open(os.path.join(model_output_dir, "context_vectors.npy"), "wb") as f:
        np.save(f, context_vectors)
    with open(os.path.join(model_output_dir, "responses.pkl"), "wb") as f:
        pickle.dump(responses, f)

    print("✅ Semantic retrieval agent trained and saved!")

if __name__ == "__main__":
    train_semantic_agent()
