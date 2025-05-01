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

st.title("🧠 Ubuntu Chat Agent (Semantic Search)")
st.caption("Now powered by Sentence Transformers!")

with st.form("chat_form", clear_on_submit=True):
    user_input = st.text_input("You:", placeholder="Ask about Ubuntu...")
    submitted = st.form_submit_button("Send")

    if submitted and user_input:
        user_vec = model.encode([user_input], normalize_embeddings=True)
        dist, idx = nn_model.kneighbors(user_vec, n_neighbors=1)
        idx = idx[0][0]
        response = responses[idx]
        st.session_state.chat_history.append(("You", user_input))
        st.session_state.chat_history.append(("Agent", response))

for role, msg in st.session_state.chat_history:
    if role == "You":
        st.markdown(f"**🧑 You:** {msg}")
    else:
        st.markdown(f"**🤖 Agent:** {msg}")
