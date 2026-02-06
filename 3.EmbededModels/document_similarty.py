from langchain_huggingface import HuggingFaceEmbeddings
from dotenv import load_dotenv
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

# Load environment variables (.env)
load_dotenv()

# Create HuggingFace Embedding model (free, no API key needed)
embedding = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# Documents (knowledge base)
documents = [
    "Virat Kohli is an Indian cricketer known for his aggressive batting and leadership.",
    "MS Dhoni is a former Indian captain famous for his calm demeanor and finishing skills.",
    "Sachin Tendulkar, also known as the 'God of Cricket', holds many batting records.",
    "Rohit Sharma is known for his elegant batting and record-breaking double centuries.",
    "Jasprit Bumrah is an Indian fast bowler known for his unorthodox action and yorkers."
]

# Query
query = "tell me about bumrah"

# Generate embeddings
doc_embeddings = embedding.embed_documents(documents)
query_embedding = embedding.embed_query(query)

# Compute cosine similarity
scores = cosine_similarity(
    [query_embedding],
    doc_embeddings
)[0]

# Get best matching document
index, score = sorted(
    list(enumerate(scores)),
    key=lambda x: x[1]
)[-1]

# Output
print("Query:", query)
print("Most Similar Document:")
print(documents[index])
print("Similarity Score:", score)
