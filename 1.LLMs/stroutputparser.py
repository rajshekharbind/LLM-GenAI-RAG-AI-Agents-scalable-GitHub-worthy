from huggingface_hub import InferenceClient
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser, JsonOutputParser
from langchain_core.runnables import RunnableLambda
from dotenv import load_dotenv
import os
import json

load_dotenv()
hf_token = os.getenv("HUGGINGFACEHUB_ACCESS_TOKEN")
if not hf_token:
    raise ValueError("HUGGINGFACEHUB_ACCESS_TOKEN not found in environment variables. Please add it to your .env file.")

# Initialize the HuggingFace Inference Client
client = InferenceClient(api_key=hf_token)

# Create a custom Runnable wrapper for HuggingFace InferenceClient
def hf_chat(prompt):
    """Custom chat function that wraps HuggingFace InferenceClient"""
    # Convert StringPromptValue to string if needed
    prompt_text = str(prompt) if hasattr(prompt, '__str__') else prompt
    messages = [{"role": "user", "content": prompt_text}]
    response = client.chat_completion(
        messages=messages,
        model="mistralai/Mistral-7B-Instruct-v0.2"
    )
    return response.choices[0].message.content

# Convert function to RunnableLambda so it can be used in chains
hf_chat_runnable = RunnableLambda(hf_chat)

# Define prompt templates
template1 = PromptTemplate(
    template="Analyze this review and return the response in valid JSON format with these fields: key_themes (list), summary (string), sentiment (pos/neg), pros (list), cons (list), name (string). {review_text}",
    input_variables=["review_text"]
)

template2 = PromptTemplate(
    template="The review is as follows: {text}.\nPlease provide the analysis in JSON format with the following fields: key_themes (list), summary (string), sentiment (pos/neg), pros (list), cons (list), name (string).",
    input_variables=["text"]
)

# Create chains using the pipe operator (|)
chain1 = template1 | hf_chat_runnable | StrOutputParser()
chain2 = template2 | hf_chat_runnable | StrOutputParser()

# Test data
review_text = "I recently upgraded to the Samsung Galaxy S24 Ultra, and I'm absolutely impressed! Great camera, fast performance, and excellent battery life."

# Test with first chain
print("=" * 80)
print("CHAIN 1 - Template Analysis")
print("=" * 80)
result1 = chain1.invoke({"review_text": review_text})
print(result1)

# Test with second chain
print("\n" + "=" * 80)
print("CHAIN 2 - Template Analysis")
print("=" * 80)
result2 = chain2.invoke({"text": review_text})
print(result2)

# Create a combined chain that uses both templates sequentially
combined_chain = template1 | hf_chat_runnable | StrOutputParser()
print("\n" + "=" * 80)
print("COMBINED CHAIN RESULT")
print("=" * 80)
combined_result = combined_chain.invoke({"review_text": review_text})
print(combined_result)
