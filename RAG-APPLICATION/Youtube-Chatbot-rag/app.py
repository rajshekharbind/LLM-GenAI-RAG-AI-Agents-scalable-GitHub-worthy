import os
import logging
from dotenv import load_dotenv
from flask import Flask, render_template, request, jsonify
from flask_cors import CORS

from youtube_transcript_api import YouTubeTranscriptApi
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.llms import HuggingFaceHub
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableParallel, RunnablePassthrough, RunnableLambda

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()
HUGGINGFACEHUB_ACCESS_TOKEN = os.getenv("HUGGINGFACEHUB_ACCESS_TOKEN")
if not HUGGINGFACEHUB_ACCESS_TOKEN:
    raise ValueError("HUGGINGFACEHUB_ACCESS_TOKEN not found in .env file")

# Flask app
app = Flask(__name__)
CORS(app)

# Global variables for the RAG chain
vector_store = None
rag_chain = None
current_video_id = None

# ---------------------------------------------------------
# YouTube Transcript Loader
# ---------------------------------------------------------
def load_youtube_transcript(video_id: str) -> str:
    try:
        api = YouTubeTranscriptApi()
        fetched_transcript = api.fetch(video_id, languages=["en"])
        transcript = " ".join(chunk.text for chunk in fetched_transcript)
        return transcript
    except Exception as e:
        raise RuntimeError(f"Could not fetch transcript: {str(e)}")

# ---------------------------------------------------------
# Text Chunking
# ---------------------------------------------------------
def split_text(text: str):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )
    return splitter.create_documents([text])

# ---------------------------------------------------------
# Build Vector Store
# ---------------------------------------------------------
def build_vector_store(docs):
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )
    vector_store = FAISS.from_documents(docs, embeddings)
    return vector_store

# ---------------------------------------------------------
# Build RAG Chain
# ---------------------------------------------------------
def build_chain(vector_store):
    retriever = vector_store.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 4}
    )
    
    llm = HuggingFaceHub(
        repo_id="google/flan-t5-large",
        model_kwargs={"temperature": 0.5, "max_length": 256},
        huggingfacehub_api_token=HUGGINGFACEHUB_ACCESS_TOKEN,
        task="text2text-generation"
    )
    
    prompt = PromptTemplate(
        template="""You are a helpful assistant.
Answer ONLY from the provided YouTube transcript context.
If the answer is not present, say "I don't know."

Context:
{context}

Question:
{question}

Answer:""",
        input_variables=["context", "question"]
    )
    
    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)
    
    parallel_chain = RunnableParallel(
        {
            "context": retriever | RunnableLambda(format_docs),
            "question": RunnablePassthrough()
        }
    )
    
    chain = parallel_chain | prompt | llm | StrOutputParser()
    return chain

# ---------------------------------------------------------
# Flask Routes
# ---------------------------------------------------------
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/load_video', methods=['POST'])
def load_video():
    global vector_store, rag_chain, current_video_id
    
    try:
        data = request.json
        video_id = data.get('video_id')
        
        if not video_id:
            return jsonify({'error': 'Video ID is required'}), 400
        
        logger.info(f"Loading video: {video_id}")
        
        # Load transcript
        transcript = load_youtube_transcript(video_id)
        logger.info(f"Transcript loaded: {len(transcript)} characters")
        
        # Split text
        docs = split_text(transcript)
        logger.info(f"Text split into {len(docs)} chunks")
        
        # Build vector store
        logger.info("Building vector store...")
        vector_store = build_vector_store(docs)
        
        # Build RAG chain
        logger.info("Building RAG chain...")
        rag_chain = build_chain(vector_store)
        
        current_video_id = video_id
        logger.info(f"Video {video_id} loaded successfully")
        
        return jsonify({
            'success': True,
            'message': f'Video loaded successfully! You can now ask questions.',
            'video_id': video_id
        })
        
    except Exception as e:
        logger.error(f"Error loading video: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/ask', methods=['POST'])
def ask_question():
    global rag_chain
    
    try:
        if not rag_chain:
            logger.warning("Attempt to ask question without loading video")
            return jsonify({'error': 'Please load a video first'}), 400
        
        data = request.json
        question = data.get('question')
        
        if not question:
            return jsonify({'error': 'Question is required'}), 400
        
        logger.info(f"Question received: {question}")
        
        # Get answer from RAG chain
        answer = rag_chain.invoke(question)
        logger.info(f"Answer generated successfully: {len(answer)} characters")
        
        # Ensure we have a valid answer
        if not answer or answer.strip() == "":
            answer = "I couldn't find a relevant answer in the video transcript."
        
        return jsonify({
            'success': True,
            'answer': answer.strip()
        })
        
    except Exception as e:
        logger.error(f"Error in ask_question: {str(e)}", exc_info=True)
        return jsonify({'error': f'Error: {str(e)}'}), 500

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
