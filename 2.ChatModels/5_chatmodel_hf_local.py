# Install required libraries (run once)
# pip install -U langchain langchain-huggingface transformers accelerate torch

from langchain_huggingface import ChatHuggingFace, HuggingFacePipeline
#import os

# Optional: Set Hugging Face cache directory (Windows-friendly)
#os.environ["HF_HOME"] = "D:/huggingface_cache"

# Create HuggingFace Pipeline
llm = HuggingFacePipeline.from_model_id(
    model_id="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    task="text-generation",
    pipeline_kwargs={
        "temperature": 0.5,
        "max_new_tokens": 100,
        "do_sample": True
    }
)

# Wrap pipeline with LangChain Chat Model
model = ChatHuggingFace(llm=llm)

# Invoke the model
result = model.invoke("What is the capital of India?")

# Print response
print(result.content)
