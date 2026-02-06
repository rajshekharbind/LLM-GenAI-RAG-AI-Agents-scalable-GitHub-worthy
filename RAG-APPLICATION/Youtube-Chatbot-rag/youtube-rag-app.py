import os
from dotenv import load_dotenv

from youtube_transcript_api import YouTubeTranscriptApi
from langchain_text_splitters import RecursiveCharacterTextSplitter

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEndpoint

from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import (
    RunnableParallel,
    RunnablePassthrough,
    RunnableLambda
)


load_dotenv()
HUGGINGFACEHUB_ACCESS_TOKEN = os.getenv("HUGGINGFACEHUB_ACCESS_TOKEN")
if not HUGGINGFACEHUB_ACCESS_TOKEN:
    raise ValueError("HUGGINGFACEHUB_ACCESS_TOKEN not found in .env file")

# ---------------------------------------------------------
# 2. YouTube Transcript Loader
# ---------------------------------------------------------
def load_youtube_transcript(video_id: str) -> str:
    try:
        # Create an instance and fetch the transcript
        api = YouTubeTranscriptApi()
        fetched_transcript = api.fetch(video_id, languages=["en"])
        # FetchedTranscript is iterable, each item has .text attribute
        transcript = " ".join(chunk.text for chunk in fetched_transcript)
        return transcript
    except Exception as e:
        raise RuntimeError(f"Could not fetch transcript: {str(e)}")

# ---------------------------------------------------------
# 3. Text Chunking
# ---------------------------------------------------------
def split_text(text: str):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )
    return splitter.create_documents([text])

# ---------------------------------------------------------
# 4. Build Vector Store (FAISS)
# ---------------------------------------------------------
def build_vector_store(docs):
    # Using HuggingFace embeddings (free, no API credits needed)
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )
    vector_store = FAISS.from_documents(docs, embeddings)
    return vector_store

# ---------------------------------------------------------
# 5. Retriever
# ---------------------------------------------------------
def get_retriever(vector_store):
    return vector_store.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 4}
    )

# ---------------------------------------------------------
# 6. Prompt Template
# ---------------------------------------------------------
PROMPT = PromptTemplate(
    template="""
You are a helpful assistant.
Answer ONLY from the provided YouTube transcript context.
If the answer is not present, say "I don't know."

Context:
{context}

Question:
{question}
""",
    input_variables=["context", "question"]
)

# ---------------------------------------------------------
# 7. Helper to format docs
# ---------------------------------------------------------
def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

# ---------------------------------------------------------
# 8. Build RAG Chain
# ---------------------------------------------------------
def build_chain(retriever):
    # Using HuggingFace API with a compatible model for text-generation
    llm = HuggingFaceEndpoint(
        repo_id="google/flan-t5-large",
        temperature=0.2,
        max_new_tokens=512,
        huggingfacehub_api_token=HUGGINGFACEHUB_ACCESS_TOKEN
    )

    parallel_chain = RunnableParallel(
        {
            "context": retriever | RunnableLambda(format_docs),
            "question": RunnablePassthrough()
        }
    )

    chain = parallel_chain | PROMPT | llm | StrOutputParser()
    return chain

# ---------------------------------------------------------
# 9. Main App
# ---------------------------------------------------------
def main():
    # 🔁 Replace with your YouTube video ID 
    # To get video ID: from URL https://www.youtube.com/watch?v=VIDEO_ID
    # Example: "dQw4w9WgXcQ" from https://www.youtube.com/watch?v=dQw4w9WgXcQ
    video_id = "jNQXAC9IVRw"  # "How to learn any language in six months" TED talk

    print("🔹 Loading transcript...")
    transcript = load_youtube_transcript(video_id)

    print("🔹 Splitting text...")
    docs = split_text(transcript)

    print("🔹 Creating vector store...")
    vector_store = build_vector_store(docs)

    retriever = get_retriever(vector_store)
    rag_chain = build_chain(retriever)

    print("\n✅ YouTube Chatbot Ready!")
    print("Type 'exit' to quit.\n")

    while True:
        question = input("You: ")
        if question.lower() == "exit":
            break

        answer = rag_chain.invoke(question)
        print(f"\nBot: {answer}\n")

# ---------------------------------------------------------
# 10. Run
# ---------------------------------------------------------
if __name__ == "__main__":
    main()
