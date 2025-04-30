import streamlit as st
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors

# Load trained model and data
@st.cache_resource
def load_agent(model_dir="models/"):
    with open(model_dir + "vectorizer.pkl", "rb") as f:
        vectorizer = pickle.load(f)
    with open(model_dir + "nn_model.pkl", "rb") as f:
        nn_model = pickle.load(f)
    with open(model_dir + "responses.pkl", "rb") as f:
        responses = pickle.load(f)
    return vectorizer, nn_model, responses

vectorizer, nn_model, responses = load_agent()

# Initialize session state for chat history
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# App layout
st.title("💬 Ubuntu Chat Agent")
st.caption("Ask anything about Ubuntu and get a response from a retrieval-based agent.")

with st.form(key="chat_form", clear_on_submit=True):
    user_input = st.text_input("You:", placeholder="e.g., How do I install Ubuntu?")
    submitted = st.form_submit_button("Send")

    if submitted and user_input:
        user_vec = vectorizer.transform([user_input])
        dist, idx = nn_model.kneighbors(user_vec, n_neighbors=1)
        idx = idx[0][0]
        agent_reply = responses[idx]

        # Update chat history
        st.session_state.chat_history.append(("You", user_input))
        st.session_state.chat_history.append(("Agent", agent_reply))

# Display full chat history
for role, msg in st.session_state.chat_history:
    if role == "You":
        st.markdown(f"**🧑 You:** {msg}")
    else:
        st.markdown(f"**🤖 Agent:** {msg}")