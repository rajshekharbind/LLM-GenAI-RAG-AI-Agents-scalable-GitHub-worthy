from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
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
    SystemMessage(content='You are a helpful assistant'),
    HumanMessage(content='Tell me about LangChain')
]

# Convert messages to API format
api_messages = []
for msg in messages:
    if isinstance(msg, SystemMessage):
        api_messages.append({"role": "system", "content": msg.content})
    elif isinstance(msg, HumanMessage):
        api_messages.append({"role": "user", "content": msg.content})
    elif isinstance(msg, AIMessage):
        api_messages.append({"role": "assistant", "content": msg.content})

result = client.chat_completion(
    model="mistralai/Mistral-7B-Instruct-v0.2",
    messages=api_messages,
    max_tokens=500,
    temperature=0.7
)

messages.append(AIMessage(content=result.choices[0].message.content))
print(messages)
