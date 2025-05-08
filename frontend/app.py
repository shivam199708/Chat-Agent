import streamlit as st
import pickle
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.neighbors import NearestNeighbors

@st.cache_resource
def load_agent(model_dir="models/"):
    with open(model_dir + "sentence_model.pkl", "rb") as f:
        model = pickle.load(f)
    with open(model_dir + "nn_model.pkl", "rb") as f:
        nn_model = pickle.load(f)
    with open(model_dir + "context_vectors.npy", "rb") as f:
        context_vectors = np.load(f)
    with open(model_dir + "responses.pkl", "rb") as f:
        responses = pickle.load(f)
    return model, nn_model, context_vectors, responses

model, nn_model, context_vectors, responses = load_agent()

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "last_input" not in st.session_state:
    st.session_state.last_input = ""

st.title("🧠 Ubuntu Chat Agent (Top-K + Context Aware)")
st.caption("Powered by Sentence Transformers and Smart Retrieval")

with st.form("chat_form", clear_on_submit=True):
    user_input = st.text_input("You:", placeholder="Ask about Ubuntu...")
    submitted = st.form_submit_button("Send")

    if submitted and user_input:
        # Combine previous user input for context-aware query
        full_query = (st.session_state.last_input + " " + user_input).strip()
        user_vec = model.encode([full_query], normalize_embeddings=True)
        dist, idxs = nn_model.kneighbors(user_vec, n_neighbors=3)

        suggestions = [responses[i] for i in idxs[0]]

        # Store current interaction
        st.session_state.chat_history.append(("You", user_input))
        for response in suggestions:
            st.session_state.chat_history.append(("Agent", response))

        # Update previous input
        st.session_state.last_input = user_input

# Display conversation
for role, msg in st.session_state.chat_history:
    if role == "You":
        st.markdown(f"**🧑 You:** {msg}")
    else:
        st.markdown(f"**🤖 Agent:** {msg}")
