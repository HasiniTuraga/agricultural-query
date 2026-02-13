import streamlit as st
import faiss
import numpy as np
import pandas as pd
import os
from sentence_transformers import SentenceTransformer
import google.generativeai as genai

# ==============================
# 🔹 CONFIGURATION
# ==============================

st.set_page_config(page_title="🌾 AI Farmer Assistant", layout="wide")

st.title("🌾 Generative AI Farmer Assistant")
st.write("AI-powered agricultural query assistant with Offline + Online Mode")

# ==============================
# 🔹 GOOGLE GEMINI API SETUP
# ==============================

api_key = st.sidebar.text_input("AIzaSyB_2EnAS9jxB7FEmpFLWBVIcUunX2kdaeE", type="password")

if api_key:
    genai.configure(api_key=api_key)

# ==============================
# 🔹 SAMPLE DATASET (You can replace with CSV later)
# ==============================

data = [
    {"question": "How to control aphids in mustard?",
     "answer": "Spray neem oil 5ml per litre or use imidacloprid 0.3ml per litre."},

    {"question": "What fertilizer during flowering in wheat?",
     "answer": "Apply urea 45 kg per acre during flowering stage."},

    {"question": "How to apply for PM Kisan Samman Nidhi?",
     "answer": "Visit pmkisan.gov.in and register with Aadhaar and bank details."},

    {"question": "How to treat blight in potato?",
     "answer": "Spray Mancozeb 2g per litre at 7 day intervals."},

    {"question": "How to control whitefly in cotton?",
     "answer": "Use Thiamethoxam 0.25g per litre or neem based bio pesticide."}
]

df = pd.DataFrame(data)

# ==============================
# 🔹 LOAD EMBEDDING MODEL
# ==============================

@st.cache_resource
def load_model():
    return SentenceTransformer('all-MiniLM-L6-v2')

model = load_model()

# ==============================
# 🔹 CREATE FAISS INDEX
# ==============================

@st.cache_resource
def create_faiss_index():
    embeddings = model.encode(df["question"].tolist())
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(np.array(embeddings))
    return index, embeddings

index, embeddings = create_faiss_index()

# ==============================
# 🔹 SEARCH FUNCTION (OFFLINE)
# ==============================

def search_query(query, top_k=2):
    query_vector = model.encode([query])
    distances, indices = index.search(np.array(query_vector), top_k)
    results = df.iloc[indices[0]]
    return results

# ==============================
# 🔹 GEMINI RESPONSE (ONLINE)
# ==============================

def generate_gemini_response(query, context):
    try:
        model_gemini = genai.GenerativeModel("gemini-1.5-flash")

        prompt = f"""
        You are an agricultural expert assistant.

        Context:
        {context}

        User Question:
        {query}

        Give a clear, simple answer for farmers.
        """

        response = model_gemini.generate_content(prompt)
        return response.text

    except Exception as e:
        return f"Error: {str(e)}"

# ==============================
# 🔹 STREAMLIT UI
# ==============================

mode = st.sidebar.radio("Select Mode", ["Offline (FAISS Only)", "Online (Gemini + FAISS)"])

query = st.text_input("Ask your agriculture question:")

if st.button("Get Answer"):

    if query.strip() == "":
        st.warning("Please enter a question.")
    else:
        with st.spinner("Processing..."):

            results = search_query(query)
            context_text = ""

            st.subheader("📚 Offline Retrieved Answers")

            for i, row in results.iterrows():
                st.write("**Q:**", row["question"])
                st.write("**A:**", row["answer"])
                st.write("---")
                context_text += f"Q: {row['question']}\nA: {row['answer']}\n\n"

            if mode == "Online (Gemini + FAISS)" and api_key:
                st.subheader("🤖 Gemini Generated Answer")

                gemini_response = generate_gemini_response(query, context_text)
                st.success(gemini_response)

            elif mode == "Online (Gemini + FAISS)" and not api_key:
                st.error("Please enter your Google API key in sidebar.")

st.markdown("---")
st.caption("Built using Streamlit + FAISS + Sentence Transformers + Google Gemini")

