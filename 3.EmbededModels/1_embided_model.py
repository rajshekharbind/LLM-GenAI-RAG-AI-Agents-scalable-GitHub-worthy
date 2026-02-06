from langchain_huggingface import HuggingFaceEmbeddings
from dotenv import load_dotenv
import os

# Load environment variables from .env file
load_dotenv()

# Create Embedding Model using HuggingFace (free, no API key needed)
print("Loading HuggingFace embedding model...")
embedding = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# Generate embedding for a query
print("Generating embedding...")
result = embedding.embed_query("Delhi is the capital of India")

# Print embedding vector
print(result)
print("Embedding length:", len(result))
