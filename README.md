# 🧠 Ubuntu Chat Agent

A retrieval-based chatbot trained on the Ubuntu Dialogue Corpus using Sentence Transformers. Built with Python, Streamlit, and deployed via Streamlit Cloud. Ask questions related to Ubuntu and get real-time responses from a model that understands semantic meaning.

---

## 🚀 Demo
**Live App**: [Click here to try it out](https://chat-agent-ci6erwl7and4p9puahgofz.streamlit.app/)

---

## 📌 Features

- ✅ Semantic search using `all-MiniLM-L6-v2`
- ✅ SentenceTransformer-based embedding model
- ✅ Smart NearestNeighbors retrieval
- ✅ Streamlit web app interface
- ✅ Multi-turn conversation with chat history
- ✅ GitHub Actions CI/CD workflow
- ✅ Deployed on Streamlit Cloud

---

## 📂 Project Structure

```
customer-chat-agent/
├── .github/workflows/            
├── .streamlit/config.toml       
├── data/                         
├── frontend/app.py              
├── models/                      
├── notebooks/                  
├── requirements.txt             
├── src/                         
│   ├── data_loading.py
│   ├── preprocess.py
│   ├── train_agent.py
│   └── inference.py
└── README.md
```

---

## 🛠️ How It Works

1. Load and preprocess Ubuntu Dialogue dataset
2. Convert to context-response pairs
3. Encode contexts using SentenceTransformer (`all-MiniLM-L6-v2`)
4. Fit NearestNeighbors model using cosine similarity
5. Streamlit UI lets users input queries
6. On input, query is embedded and nearest match is retrieved and displayed

---

## 📦 Requirements

- Python 3.10+
- `sentence-transformers`
- `scikit-learn`
- `streamlit`
- `pandas`, `numpy`

Install dependencies:
```bash
pip install -r requirements.txt
```

Run locally:
```bash
streamlit run frontend/app.py
```

---

## 🧪 Example Queries
- "How to install Ubuntu alongside Windows?"
- "My WiFi isn’t working on Ubuntu."
- "How to partition a hard drive in Linux?"

---

## 🤖 Future Improvements
- Top-k response ranking
- UI styling (chat bubbles, avatars)
- Hybrid retrieval + generative model
- Saving chat history to database

---