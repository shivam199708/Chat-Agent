import streamlit as st
import pickle
import numpy as np
import json
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

def save_feedback(log, path="feedback_log.jsonl"):
    with open(path, "a", encoding="utf-8") as f:
        for item in log:
            f.write(json.dumps(item) + "\n")

model, nn_model, context_vectors, responses = load_agent()

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "last_input" not in st.session_state:
    st.session_state.last_input = ""
if "feedback_log" not in st.session_state:
    st.session_state.feedback_log = []

st.title("🧠 Ubuntu Chat Agent (Top-K + Context Aware + Feedback)")
st.caption("Powered by Sentence Transformers and Smart Retrieval")

with st.form("chat_form", clear_on_submit=True):
    user_input = st.text_input("You:", placeholder="Ask about Ubuntu...")
    submitted = st.form_submit_button("Send")

    if submitted and user_input:
        full_query = (st.session_state.last_input + " " + user_input).strip()
        user_vec = model.encode([full_query], normalize_embeddings=True)
        dist, idxs = nn_model.kneighbors(user_vec, n_neighbors=3)

        suggestions = [responses[i] for i in idxs[0]]

        # Overwrite chat history with only the current user input and responses
        st.session_state.chat_history = [("You", user_input)] + [("Agent", res, user_input) for res in suggestions]

        st.session_state.last_input = user_input

# Display conversation with feedback buttons
for i, entry in enumerate(st.session_state.chat_history):
    if len(entry) == 2:
        role, msg = entry
    else:
        role, msg, context = entry

    if role == "You":
        st.markdown(f"**🧑 You:** {msg}")
    else:
        st.markdown(f"**🤖 Agent:** {msg}")
        col1, col2 = st.columns([1, 1])
        with col1:
            if st.button(f"👍 {i}", key=f"thumbs_up_{i}"):
                st.session_state.feedback_log.append({"context": context, "response": msg, "feedback": "up"})
        with col2:
            if st.button(f"👎 {i}", key=f"thumbs_down_{i}"):
                st.session_state.feedback_log.append({"context": context, "response": msg, "feedback": "down"})

# Save feedback to file
if st.button("💾 Save Feedback"):
    save_feedback(st.session_state.feedback_log)
    st.success("✅ Feedback saved to feedback_log.jsonl")
