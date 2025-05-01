import pickle
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.neighbors import NearestNeighbors

def load_semantic_agent(model_dir="models/"):
    with open(model_dir + "sentence_model.pkl", "rb") as f:
        model = pickle.load(f)
    with open(model_dir + "nn_model.pkl", "rb") as f:
        nn_model = pickle.load(f)
    with open(model_dir + "context_vectors.npy", "rb") as f:
        context_vectors = np.load(f)
    with open(model_dir + "responses.pkl", "rb") as f:
        responses = pickle.load(f)
    return model, nn_model, context_vectors, responses

def chat(model, nn_model, context_vectors, responses):
    print("🤖 Semantic Chat Agent Ready! (type 'exit' to quit)\n")
    while True:
        user_input = input("You: ")
        if user_input.lower() in ("exit", "quit"):
            print("👋 Goodbye!")
            break
        user_vec = model.encode([user_input], normalize_embeddings=True)
        dist, idx = nn_model.kneighbors(user_vec, n_neighbors=1)
        idx = idx[0][0]
        print("Agent:", responses[idx])

if __name__ == "__main__":
    model, nn_model, context_vectors, responses = load_semantic_agent()
    chat(model, nn_model, context_vectors, responses)
