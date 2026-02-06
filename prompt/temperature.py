from huggingface_hub import InferenceClient
from dotenv import load_dotenv
import os

load_dotenv()

hf_token = os.getenv("HUGGINGFACEHUB_ACCESS_TOKEN")
if not hf_token:
    print("❌ HUGGINGFACEHUB_ACCESS_TOKEN not found in environment variables")
    exit()

client = InferenceClient(api_key=hf_token)

messages = [
    {"role": "user", "content": "Write a 5 line poem on cricket"}
]

result = client.chat_completion(
    model="mistralai/Mistral-7B-Instruct-v0.2",
    messages=messages,
    max_tokens=500,
    temperature=0.7
)

print(result.choices[0].message.content)