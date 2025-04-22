# download_model.py

from sentence_transformers import SentenceTransformer

# Download and cache the model locally
print("Downloading model...")
model = SentenceTransformer('all-MiniLM-L6-v2')

# Save model to folder in your project directory
model.save('./all-MiniLM-L6-v2')
print("✅ Model saved to ./all-MiniLM-L6-v2")
