"""
Test script to verify the RAG chain works properly
Run this to debug issues with the chatbot
"""

import os
from dotenv import load_dotenv

# Load environment
load_dotenv()

print("🔍 Testing YouTube RAG Chatbot Components...\n")

# Test 1: Check environment variables
print("1️⃣ Checking environment variables...")
hf_token = os.getenv("HUGGINGFACEHUB_ACCESS_TOKEN")
if hf_token:
    print(f"✅ HuggingFace token found: {hf_token[:10]}...")
else:
    print("❌ HuggingFace token NOT found in .env file!")
    exit(1)

# Test 2: Import packages
print("\n2️⃣ Testing imports...")
try:
    from youtube_transcript_api import YouTubeTranscriptApi
    from langchain_text_splitters import RecursiveCharacterTextSplitter
    from langchain_huggingface import HuggingFaceEmbeddings
    from langchain_community.vectorstores import FAISS
    from langchain_community.llms import HuggingFaceHub
    from langchain_core.prompts import PromptTemplate
    from langchain_core.output_parsers import StrOutputParser
    from langchain_core.runnables import RunnableParallel, RunnablePassthrough, RunnableLambda
    print("✅ All imports successful")
except Exception as e:
    print(f"❌ Import error: {e}")
    exit(1)

# Test 3: Load YouTube transcript
print("\n3️⃣ Testing YouTube transcript loading...")
try:
    api = YouTubeTranscriptApi()
    video_id = "jNQXAC9IVRw"
    print(f"Loading video: {video_id}")
    transcript = api.fetch(video_id, languages=["en"])
    text = " ".join(chunk.text for chunk in transcript)
    print(f"✅ Transcript loaded: {len(text)} characters")
    print(f"Sample: {text[:100]}...")
except Exception as e:
    print(f"❌ Transcript error: {e}")
    exit(1)

# Test 4: Text splitting
print("\n4️⃣ Testing text splitting...")
try:
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    docs = splitter.create_documents([text])
    print(f"✅ Split into {len(docs)} chunks")
except Exception as e:
    print(f"❌ Splitting error: {e}")
    exit(1)

# Test 5: Embeddings
print("\n5️⃣ Testing embeddings (this may take a minute on first run)...")
try:
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    test_embedding = embeddings.embed_query("test")
    print(f"✅ Embeddings working: {len(test_embedding)} dimensions")
except Exception as e:
    print(f"❌ Embeddings error: {e}")
    exit(1)

# Test 6: Vector store
print("\n6️⃣ Testing vector store...")
try:
    vector_store = FAISS.from_documents(docs, embeddings)
    print(f"✅ Vector store created")
except Exception as e:
    print(f"❌ Vector store error: {e}")
    exit(1)

# Test 7: HuggingFace LLM
print("\n7️⃣ Testing HuggingFace LLM (this may take time)...")
try:
    llm = HuggingFaceHub(
        repo_id="google/flan-t5-large",
        model_kwargs={"temperature": 0.5, "max_length": 256},
        huggingfacehub_api_token=hf_token,
        task="text2text-generation"
    )
    test_response = llm.invoke("What is 2+2?")
    print(f"✅ LLM working")
    print(f"Sample response: {test_response}")
except Exception as e:
    print(f"❌ LLM error: {e}")
    import traceback
    traceback.print_exc()
    print("\n💡 The model might be loading or there's an API issue.")
    print("Try waiting a minute and running the test again.")
    exit(1)

# Test 8: Full RAG chain
print("\n8️⃣ Testing full RAG chain...")
try:
    retriever = vector_store.as_retriever(search_type="similarity", search_kwargs={"k": 4})
    
    prompt = PromptTemplate(
        template="""Answer the question based on the context.

Context: {context}

Question: {question}

Answer:""",
        input_variables=["context", "question"]
    )
    
    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)
    
    chain = (
        RunnableParallel(
            {
                "context": retriever | RunnableLambda(format_docs),
                "question": RunnablePassthrough()
            }
        )
        | prompt
        | llm
        | StrOutputParser()
    )
    
    print("Testing with question: 'What is shown in this video?'")
    answer = chain.invoke("What is shown in this video?")
    print(f"✅ RAG chain working!")
    print(f"Answer: {answer}")
    
except Exception as e:
    print(f"❌ RAG chain error: {e}")
    import traceback
    traceback.print_exc()
    exit(1)

print("\n" + "="*50)
print("✅ ALL TESTS PASSED!")
print("Your YouTube RAG Chatbot should work correctly.")
print("="*50)
