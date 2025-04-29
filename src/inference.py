import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors

def load_agent(model_dir="models/"):
    """
    Load the retrieval agent from disk
    """
    with open(model_dir + "vectorizer.pkl", "rb") as f:
        vectorizer = pickle.load(f)
    with open(model_dir + "nn_model.pkl", "rb") as f:
        nn_model = pickle.load(f)
    with open(model_dir + "responses.pkl", "rb") as f:
        responses = pickle.load(f)

    return vectorizer, nn_model, responses

def chat(vectorizer, nn_model, responses):
    """
    Simple chat loop
    """
    print("🤖 Chat Agent Ready! (type 'exit' to quit)\n")

    while True:
        user_input = input("You: ")
        if user_input.lower() in ("exit", "quit"):
            print("👋 Goodbye!")
            break

        # Embed user input
        user_vec = vectorizer.transform([user_input])

        # Find nearest neighbor
        dist, idx = nn_model.kneighbors(user_vec, n_neighbors=1)
        idx = idx[0][0]

        # Respond
        print("Agent:", responses[idx])

if __name__ == "__main__":
    vectorizer, nn_model, responses = load_agent()
    chat(vectorizer, nn_model, responses)
