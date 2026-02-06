import os
from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma

# --------------------------------------------------
# 1. Set OpenAI API Key
# --------------------------------------------------
load_dotenv()  # Load environment variables from .env file

# Verify API key is set
if not os.getenv("OPENAI_API_KEY"):
    raise ValueError("OPENAI_API_KEY not found. Please set it in your .env file or environment variables.")


# --------------------------------------------------
# 2. Create Documents
# --------------------------------------------------
docs = [
    Document(
        page_content="Virat Kohli is one of the most successful and consistent batsmen in IPL history.",
        metadata={"team": "Royal Challengers Bangalore"}
    ),
    Document(
        page_content="Rohit Sharma is the most successful captain in IPL history, leading Mumbai Indians to five titles.",
        metadata={"team": "Mumbai Indians"}
    ),
    Document(
        page_content="MS Dhoni, known as Captain Cool, led Chennai Super Kings to multiple IPL titles.",
        metadata={"team": "Chennai Super Kings"}
    ),
    Document(
        page_content="Jasprit Bumrah is considered one of the best fast bowlers in T20 cricket.",
        metadata={"team": "Mumbai Indians"}
    ),
    Document(
        page_content="Ravindra Jadeja is a dynamic all-rounder contributing with both bat and ball.",
        metadata={"team": "Chennai Super Kings"}
    ),
]


# --------------------------------------------------
# 3. Create Vector Store
# --------------------------------------------------
# Use absolute path to avoid database location issues
script_dir = os.path.dirname(os.path.abspath(__file__))
chroma_db_path = os.path.join(script_dir, "chroma_db")

vector_store = Chroma(
    collection_name="ipl_players",
    embedding_function=OpenAIEmbeddings(),
    persist_directory=chroma_db_path
)


# --------------------------------------------------
# 4. Add Documents
# --------------------------------------------------
vector_store.add_documents(docs)


# --------------------------------------------------
# 5. View Stored Documents
# --------------------------------------------------
print("\n--- All Documents ---")
print(vector_store.get(include=["documents", "metadatas"]))


# --------------------------------------------------
# 6. Similarity Search
# --------------------------------------------------
print("\n--- Similarity Search ---")
results = vector_store.similarity_search(
    query="Who among these is a bowler?",
    k=2
)

for r in results:
    print(r.page_content, r.metadata)


# --------------------------------------------------
# 7. Similarity Search With Score
# --------------------------------------------------
print("\n--- Similarity Search With Score ---")
results_with_score = vector_store.similarity_search_with_score(
    query="Who among these is a bowler?",
    k=2
)

for doc, score in results_with_score:
    print("Score:", score)
    print(doc.page_content)


# --------------------------------------------------
# 8. Metadata Filtering
# --------------------------------------------------
print("\n--- Metadata Filtering (CSK) ---")
filtered = vector_store.similarity_search_with_score(
    query="",
    filter={"team": "Chennai Super Kings"}
)

for doc, score in filtered:
    print(doc.page_content, doc.metadata)


# --------------------------------------------------
# 9. Update Document
# --------------------------------------------------
print("\n--- Update Document ---")
doc_ids = vector_store.get()["ids"]

updated_doc = Document(
    page_content="Virat Kohli, former RCB captain, is known for aggressive leadership and consistent batting.",
    metadata={"team": "Royal Challengers Bangalore"}
)

vector_store.update_document(
    document_id=doc_ids[0],
    document=updated_doc
)

print("Document updated successfully.")


# --------------------------------------------------
# 10. Delete Document
# --------------------------------------------------
print("\n--- Delete Document ---")
vector_store.delete(ids=[doc_ids[0]])
print("Document deleted successfully.")


# --------------------------------------------------
# 11. Final View
# --------------------------------------------------
print("\n--- Final Documents ---")
print(vector_store.get(include=["documents", "metadatas"]))
