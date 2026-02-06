from langchain_core.prompts import ChatPromptTemplate
from huggingface_hub import InferenceClient
from dotenv import load_dotenv
import os

load_dotenv()

chat_template = ChatPromptTemplate([
    ('system', 'You are a helpful {domain} expert'),
    ('human', 'Explain in simple terms, what is {topic}')
])

prompt = chat_template.invoke({'domain':'cricket','topic':'Dusra'})

print("Generated Prompt:")
print(prompt)
print("\n" + "="*50 + "\n")

# Use HuggingFace to generate response
hf_token = os.getenv("HUGGINGFACEHUB_ACCESS_TOKEN")
if not hf_token:
    print("❌ HUGGINGFACEHUB_ACCESS_TOKEN not found in environment variables")
else:
    client = InferenceClient(api_key=hf_token)
    
    # Convert prompt messages to format expected by chat_completion
    messages = []
    for msg in prompt.to_messages():
        if msg.__class__.__name__ == 'SystemMessage':
            messages.append({"role": "system", "content": msg.content})
        elif msg.__class__.__name__ == 'HumanMessage':
            messages.append({"role": "user", "content": msg.content})
        elif msg.__class__.__name__ == 'AIMessage':
            messages.append({"role": "assistant", "content": msg.content})
    
    result = client.chat_completion(
        model="mistralai/Mistral-7B-Instruct-v0.2",
        messages=messages,
        max_tokens=500,
        temperature=0.7
    )
    
    print("HuggingFace Response:")
    print(result.choices[0].message.content)