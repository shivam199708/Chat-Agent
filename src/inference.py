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

def chat(model, nn_model, context_vectors, responses, top_k=3):
    print("🤖 Smart Ubuntu Chat Agent (Top-K + Context) — type 'exit' to quit.\n")
    previous_input = ""

    while True:
        user_input = input("You: ")
        if user_input.lower() in ("exit", "quit"):
            print("👋 Goodbye!")
            break

        # Build context-aware input
        full_query = (previous_input + " " + user_input).strip()

        # Encode and retrieve top-k
        user_vec = model.encode([full_query], normalize_embeddings=True)
        dist, idxs = nn_model.kneighbors(user_vec, n_neighbors=top_k)

        print("\nAgent suggestions:")
        for i, idx in enumerate(idxs[0]):
            print(f"{i+1}. {responses[idx]}")
        print()

        previous_input = user_input  

if __name__ == "__main__":
    model, nn_model, context_vectors, responses = load_semantic_agent()
    chat(model, nn_model, context_vectors, responses, top_k=3)
