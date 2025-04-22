# rag_chatbot.py

import streamlit as st
import faiss
import json
import numpy as np
from sentence_transformers import SentenceTransformer
import os
from google import genai
from google.genai import types


# Load models
embed_model = SentenceTransformer('./all-MiniLM-L6-v2')
index = faiss.read_index("faiss_index.index")
with open("metadata.json") as f:
    metadata = json.load(f)

# Setup Gemini client
genai_api_key = "AIzaSyCIhzKAOCeRUL-GX2q0jbJL6-vgxUMPIeM"
client = genai.Client(api_key=genai_api_key)

# Define Gemini interaction
def ask_gemini(prompt: str, use_web: bool = False) -> str:
    contents = [
        types.Content(
            role="user",
            parts=[types.Part.from_text(text=prompt)],
        )
    ]

    tools = [types.Tool(google_search=types.GoogleSearch())] if use_web else []

    generate_content_config = types.GenerateContentConfig(
        tools=tools,
        response_mime_type="text/plain",
    )

    chunks = client.models.generate_content_stream(
        model="gemini-2.0-flash",
        contents=contents,
        config=generate_content_config,
    )

    return "".join([chunk.text for chunk in chunks if chunk.text])

# Streamlit UI
st.title("RAG Chatbot using Gemini 2.0 Flash + FAISS")

query = st.text_area("Ask a question:")

col1, col2 = st.columns(2)

with col1:
    if st.button("🔍 Ask from Local Data"):
        q_embed = embed_model.encode([query])
        D, I = index.search(q_embed, k=3)
        context = "\n\n".join([metadata[str(i)] for i in I[0]])
        final_prompt = f"Use the following information to answer:\n\n{context}\n\nQuestion: {query}"
        answer = ask_gemini(final_prompt, use_web=False)
        st.markdown("**Answer:**")
        st.write(answer)

with col2:
    if st.button("🌐 Ask with Web Search"):
        answer = ask_gemini(query, use_web=True)
        st.markdown("**Answer:**")
        st.write(answer)
