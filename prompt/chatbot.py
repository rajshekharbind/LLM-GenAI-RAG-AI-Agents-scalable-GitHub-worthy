from huggingface_hub import InferenceClient
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from dotenv import load_dotenv
import os

load_dotenv()

hf_token = os.getenv("HUGGINGFACEHUB_ACCESS_TOKEN")
if not hf_token:
    print("❌ HUGGINGFACEHUB_ACCESS_TOKEN not found in environment variables")
    exit()

client = InferenceClient(api_key=hf_token)

chat_history = [
    SystemMessage(content='You are a helpful AI assistant')
]

while True:
    user_input = input('You: ')
    chat_history.append(HumanMessage(content=user_input))
    if user_input == 'exit':
        break
    
    # Convert chat history to API format
    messages = []
    for msg in chat_history:
        if isinstance(msg, SystemMessage):
            messages.append({"role": "system", "content": msg.content})
        elif isinstance(msg, HumanMessage):
            messages.append({"role": "user", "content": msg.content})
        elif isinstance(msg, AIMessage):
            messages.append({"role": "assistant", "content": msg.content})
    
    result = client.chat_completion(
        model="mistralai/Mistral-7B-Instruct-v0.2",
        messages=messages,
        max_tokens=500,
        temperature=0.7
    )
    
    response_content = result.choices[0].message.content
    chat_history.append(AIMessage(content=response_content))
    print("AI: ", response_content)

print(chat_history)