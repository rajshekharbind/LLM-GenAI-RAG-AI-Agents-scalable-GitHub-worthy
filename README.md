# LLM-GenAI-RAG-AI-Agents-Scalable

A comprehensive learning and implementation repository for building scalable AI applications using LangChain, featuring LLMs, Chat Models, RAG systems, AI Agents, and advanced prompt engineering techniques.

## 📋 Table of Contents

- [Overview](#overview)
- [Project Structure](#project-structure)
- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)
- [Quick Start](#quick-start)
- [Technologies](#technologies)
- [Key Components](#key-components)
- [Examples](#examples)
- [Contributing](#contributing)
- [License](#license)

## 🎯 Overview

This repository contains production-ready implementations of:
- **Large Language Models (LLMs)** - Integration with OpenAI, HuggingFace, and other providers
- **Chat Models** - Multi-provider chat model implementations
- **Embedding Models** - Document embedding and semantic search
- **RAG (Retrieval Augmented Generation)** - Complete RAG pipeline with vector stores
- **AI Agents** - Intelligent agents with tools and toolkits
- **Advanced Chains** - Sequential, parallel, and conditional processing chains
- **Structured Output** - Pydantic and TypedDict for type-safe outputs
- **Prompt Engineering** - Advanced prompt templates and optimization techniques

## 📁 Project Structure

```
LANG-CHAINS/
├── 1.LLMs/                          # LLM implementations
│   ├── 1_llm_demo.py
│   ├── stroutputparser.py
│   └── stroutputparser_json.py
│
├── 2.ChatModels/                    # Chat model integrations
│   ├── ChatModel_Open_api.py
│   ├── chatmode_anthropic.py
│   ├── chatmode_google.py
│   ├── 4_chatmodel_hf_api.py
│   └── 5_chatmodel_hf_local.py
│
├── 3.EmbededModels/                 # Embedding models & similarity search
│   ├── 1_embided_model.py
│   └── document_similarty.py
│
├── chains/                          # Chain patterns
│   ├── simple_chain.py
│   ├── sequential_chain.py
│   ├── parallel_chain.py
│   └── conditional_chain.py
│
├── dictionary/                      # Data structure patterns
│   ├── pydantic_demo.py
│   ├── typed_dic.py
│   ├── with_structured_output_pydantic.py
│   └── with_structured_output_typeddict.py
│
├── prompt/                          # Prompt engineering
│   ├── prompt_template.py
│   ├── chat_prompt_template.py
│   ├── message_placeholder.py
│   ├── chatbot.py
│   ├── prompt_ui.py
│   ├── temperature.py
│   └── prompt_generator.py
│
├── RAG-APPLICATION/                 # Complete RAG implementation
│   ├── langchain-document-loder/
│   │   ├── pdf_loader.py
│   │   ├── csv_loader.py
│   │   ├── text_loader.py
│   │   ├── directory_loader.py
│   │   └── webbase_loader.py
│   ├── langchain-text-spilitter/
│   │   ├── length_based.py
│   │   ├── markdown_splitting.py
│   │   ├── python_code_splitting.py
│   │   ├── semantic_meaning_based.py
│   │   └── text_structure_based.py
│   ├── vector-store/
│   │   ├── vector_store_demo.py
│   │   └── chroma_db/
│   ├── Tools/
│   │   ├── Custom-Tools.py
│   │   ├── Built-in-Tool - DuckDuckGo-Search.py
│   │   ├── Built-in-Tool - Shell-Tool.py
│   │   └── Using-StructuredTool.py
│   ├── Toolkit/
│   │   └── toolkit.py
│   ├── BaseTool-Class/
│   │   └── basetool.py
│   └── Youtube-Chatbot-rag/
│       ├── app.py
│       ├── youtube-rag-app.py
│       └── templates/index.html
│
├── Runnables-in-LangChain/          # LangChain Runnables patterns
│   ├── runnable_lambda.py
│   ├── runnable_sequence.py
│   ├── runnable_parallel.py
│   ├── runnable_branch.py
│   ├── runnable_passthrough.py
│   ├── llmchain.py
│   ├── retrievalQAchain.py
│   └── simple_llm_app.py
│
├── structure/                       # Structure demos
│   ├── pydantic_demo.py
│   └── typeddict_demo.py
│
├── requirements.txt                 # Python dependencies
├── test.py                         # Testing utilities
└── .gitignore                      # Git ignore file
```

## ✨ Features

### LLMs
- OpenAI API integration
- HuggingFace model support
- String output parsing
- JSON output parsing

### Chat Models
- Multi-provider integration (OpenAI, Anthropic, Google, HuggingFace)
- API and local model support
- Streaming responses
- Context management

### Embeddings
- Document embedding generation
- Semantic similarity search
- Vector space operations

### RAG System
- **Document Loading**: PDF, CSV, Text, Web, Directory
- **Text Splitting**: Length-based, semantic, markdown, code-aware
- **Vector Stores**: ChromaDB integration with persistent storage
- **Retrieval**: Semantic search and ranking

### AI Agents
- Custom tool creation
- Built-in tool integration (DuckDuckGo, Shell)
- Tool chaining and orchestration
- Structured outputs

### Chains
- Simple chains for basic workflows
- Sequential chains for multi-step processes
- Parallel chains for concurrent operations
- Conditional chains for branching logic

### Advanced Features
- Message placeholders for dynamic prompts
- Chat history management
- Temperature control for response variation
- Structured outputs with Pydantic
- TypedDict for type-safe data structures

## 🚀 Installation

### Prerequisites
- Python 3.8+
- pip or conda

### Setup

1. **Clone the Repository**
```bash
git clone https://github.com/rajshekharbind/LLM-GenAI-RAG-AI-Agents-scalable-GitHub-worthy.git
cd LANG-CHAINS-REPO
```

2. **Create Virtual Environment**
```bash
# Using venv
python -m venv venv

# Activate (Windows)
venv\Scripts\activate

# Activate (Linux/Mac)
source venv/bin/activate
```

3. **Install Dependencies**
```bash
pip install -r requirements.txt
```

4. **Configure Environment Variables**
```bash
# Create .env file with your API keys
cp .env.example .env

# Edit .env and add:
# OPENAI_API_KEY=your_key_here
# HUGGINGFACE_API_KEY=your_key_here
# ANTHROPIC_API_KEY=your_key_here
# GOOGLE_API_KEY=your_key_here
```

## 📖 Usage

### Basic LLM Usage
```python
from langchain.llms import OpenAI

llm = OpenAI(api_key="your_api_key")
response = llm("What is AI?")
print(response)
```

### Chat Model Usage
```python
from langchain.chat_models import ChatOpenAI
from langchain.schema import HumanMessage

chat = ChatOpenAI()
messages = [HumanMessage(content="Hello!")]
response = chat(messages)
```

### RAG Pipeline
```python
from langchain.document_loaders import PDFLoader
from langchain.vectorstores import Chroma
from langchain.embeddings import OpenAIEmbeddings

# Load documents
loader = PDFLoader("document.pdf")
documents = loader.load()

# Create embeddings
embeddings = OpenAIEmbeddings()
vectorstore = Chroma.from_documents(documents, embeddings)

# Retrieve
results = vectorstore.similarity_search("query")
```

### Agent with Tools
```python
from langchain.agents import AgentType, initialize_agent
from langchain.tools import DuckDuckGoSearchRun

tools = [DuckDuckGoSearchRun()]
agent = initialize_agent(tools, llm, agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION)

result = agent.run("What is the capital of France?")
```

## 🎬 Quick Start Examples

### Example 1: Simple Prompt Template
See `prompt/prompt_template.py`

### Example 2: Document Q&A
See `RAG-APPLICATION/Youtube-Chatbot-rag/app.py`

### Example 3: AI Agent with Tools
See `RAG-APPLICATION/Building-end-to-end-AI-Agent-in-llm/ai-agent.py`

### Example 4: Runnable Chains
See `Runnables-in-LangChain/runnable_sequence.py`

## 🛠 Technologies

- **LangChain**: Core framework for building LLM applications
- **OpenAI**: GPT-3.5, GPT-4 models
- **Anthropic**: Claude models
- **Google**: Vertex AI
- **HuggingFace**: Open-source models and transformers
- **ChromaDB**: Vector database
- **Pydantic**: Data validation
- **Flask/Streamlit**: Web frameworks
- **Python 3.8+**: Development language

## 🔑 Key Components Explained

### 1. LLMs Folder
Direct LLM integration for text completion and generation tasks.

### 2. ChatModels Folder
Conversational AI with context awareness and multi-turn support.

### 3. Embeddings
Converting text to dense vectors for semantic search and similarity.

### 4. RAG Application
Complete retrieval-augmented generation pipeline combining retrieval and generation.

### 5. Chains
Composable pipelines for complex workflows.

### 6. Agents
Autonomous systems that use tools to accomplish goals.

### 7. Runnables
Modern LangChain patterns for building production applications.

## 📚 Examples & Tutorials

Each folder contains runnable examples. To execute:

```bash
# Run any Python file
python 1.LLMs/1_llm_demo.py

# For interactive notebooks
jupyter notebook
```

## 🤝 Contributing

Contributions are welcome! Please:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📝 Best Practices

- Keep API keys in `.env` file (never commit)
- Use virtual environment for isolated dependencies
- Follow PEP 8 style guidelines
- Add docstrings to functions
- Test code before pushing
- Use type hints for better code clarity
- Document complex logic with comments

## 🐛 Troubleshooting

### Common Issues

1. **API Key not found**
   - Ensure `.env` file exists in root directory
   - Verify API keys are correctly set

2. **Module not found**
   - Run `pip install -r requirements.txt`
   - Ensure virtual environment is activated

3. **Vector store errors**
   - Ensure ChromaDB is installed
   - Check database file permissions

## 📞 Support

For issues and questions:
- Open an issue on GitHub
- Review existing examples in each folder
- Check LangChain documentation: https://python.langchain.com

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 👤 Author

**Raj Shekhar**
- GitHub: [@rajshekharbind](https://github.com/rajshekharbind)
- Email: rajshekhar@github.com

## 🔗 Links

- **Repository**: [LLM-GenAI-RAG-AI-Agents](https://github.com/rajshekharbind/LLM-GenAI-RAG-AI-Agents-scalable-GitHub-worthy)
- **LangChain Docs**: https://python.langchain.com
- **OpenAI API**: https://platform.openai.com
- **HuggingFace**: https://huggingface.co

## 🎓 Learning Resources

- LangChain Official Documentation
- OpenAI API Documentation
- Semantic Search and Embeddings
- Prompt Engineering Best Practices
- Retrieval Augmented Generation (RAG) Patterns

---

**Last Updated**: February 2026

**Status**: ✅ Active Development

**Star** ⭐ if you find this project helpful!
