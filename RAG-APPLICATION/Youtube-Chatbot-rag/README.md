# YouTube RAG Chatbot - Web UI

A beautiful, responsive web interface for asking questions about YouTube videos using RAG (Retrieval Augmented Generation).

## Features

✨ **Modern UI** - Clean, gradient-based design with smooth animations
📱 **Fully Responsive** - Works perfectly on desktop, tablet, and mobile
🎬 **Easy Video Loading** - Just paste a YouTube URL or video ID
💬 **Interactive Chat** - Real-time Q&A with the video content
🔍 **Smart Search** - Uses HuggingFace embeddings and FAISS vector store
🤖 **AI-Powered** - Powered by Google FLAN-T5 language model

## How to Run

1. **Start the Flask server:**
   ```bash
   python app.py
   ```

2. **Open your browser:**
   ```
   http://localhost:5000
   ```

3. **Load a video:**
   - Paste a YouTube URL (e.g., `https://www.youtube.com/watch?v=jNQXAC9IVRw`)
   - Or just the video ID (e.g., `jNQXAC9IVRw`)
   - Click "Load Video"

4. **Ask questions:**
   - Type your question in the input box
   - Press Enter or click "Ask"
   - Get AI-powered answers based on the video transcript

## Example Usage

**Video:** `jNQXAC9IVRw` (How to learn any language in six months)

**Sample Questions:**
- "What is the main topic of this video?"
- "What are the key tips mentioned?"
- "How long does it take to learn a language?"

## Technology Stack

- **Backend:** Flask (Python)
- **Frontend:** HTML5, CSS3, JavaScript
- **AI/ML:** 
  - LangChain for orchestration
  - HuggingFace for embeddings and LLM
  - FAISS for vector storage
  - YouTube Transcript API

## Files

- `app.py` - Flask backend server
- `templates/index.html` - Web UI (HTML, CSS, JavaScript)
- `youtube-rag-app.py` - Original CLI version

## Notes

- Video must have English captions available
- First load may take longer as it downloads the embedding model
- Questions are answered based only on the video transcript
- The UI is fully responsive and works on all devices

## Troubleshooting

**"Could not fetch transcript" error:**
- Make sure the video has English captions
- Check that the video is publicly available
- Try a different video

**Slow responses:**
- First question may take longer as the model loads
- Subsequent questions will be faster
- Consider using a local LLM for better performance

Enjoy chatting with YouTube videos! 🎉
