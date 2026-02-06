from huggingface_hub import InferenceClient
from dotenv import load_dotenv
import os
load_dotenv()

hf_token = os.getenv("HUGGINGFACEHUB_ACCESS_TOKEN")
if not hf_token:
    raise ValueError("HUGGINGFACEHUB_ACCESS_TOKEN not found in environment variables. Please add it to your .env file.")

# Using HuggingFace Inference Client directly
print("Initializing HuggingFace Inference Client...")
client = InferenceClient(api_key=hf_token)

print("Invoking model...")
# Using conversational API instead
messages = [
    {"role": "user", "content": "What is the capital of India?"}
]
response = client.chat_completion(
    messages=messages,
    model="mistralai/Mistral-7B-Instruct-v0.2"
)
print("Result:")
print(response.choices[0].message.content)