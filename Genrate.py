# generate_embeddings.py

import pandas as pd
from sentence_transformers import SentenceTransformer
import faiss
import os
import json

# Load your Excel file
df = pd.read_excel("E:\MAINTAINANCE-BOT\TEST1\samples_300.xlsx")

# Combine relevant columns into one string per row
def row_to_text(row):
    return f"""
    System: {row['System']}
    Subsystem / Part: {row['Subsystem / Part']}
    Issue Type: {row['Issue Type']}
    Symptom: {row['Symptom / Failure Mode']}
    Diagnostic Clue: {row['Diagnostic Clue']}
    Inspection: {row['Inspection / Checkpoint']}
    Action: {row['Corrective Action']}
    """.strip()

df['combined'] = df.apply(row_to_text, axis=1)

# Generate embeddings
model = SentenceTransformer('all-MiniLM-L6-v2')
embeddings = model.encode(df['combined'].tolist(), show_progress_bar=True)

# Save FAISS index
dimension = embeddings.shape[1]
index = faiss.IndexFlatL2(dimension)
index.add(embeddings)
faiss.write_index(index, "faiss_index.index")

# Save metadata
with open("metadata.json", "w") as f:
    json.dump(df['combined'].to_dict(), f)

print("FAISS index and metadata saved.")
