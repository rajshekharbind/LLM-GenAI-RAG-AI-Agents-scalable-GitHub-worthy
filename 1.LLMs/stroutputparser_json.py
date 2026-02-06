from huggingface_hub import InferenceClient
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.runnables import RunnableLambda
import os
import json

load_dotenv()
hf_token = os.getenv("HUGGINGFACEHUB_ACCESS_TOKEN")
if not hf_token:
    raise ValueError("HUGGINGFACEHUB_ACCESS_TOKEN not found in environment variables.")

# Initialize HuggingFace client
client = InferenceClient(api_key=hf_token)

# Create custom HuggingFace chat runnable
def hf_chat(prompt):
    """Custom chat function that wraps HuggingFace InferenceClient"""
    prompt_text = str(prompt) if hasattr(prompt, '__str__') else prompt
    messages = [{"role": "user", "content": prompt_text}]
    response = client.chat_completion(
        messages=messages,
        model="mistralai/Mistral-7B-Instruct-v0.2"
    )
    return response.choices[0].message.content

hf_chat_runnable = RunnableLambda(hf_chat)

# Initialize parser
parser = JsonOutputParser()

# Define prompt template
template1 = PromptTemplate(
    template="Analyze this review and return the response in valid JSON format with these fields: key_themes (list), summary (string), sentiment (pos/neg), pros (list), cons (list), name (string). Review: {review_text}\n{format_instructions}",
    input_variables=["review_text"],
    partial_variables={"format_instructions": parser.get_format_instructions()}
)

# Create chain
chain = template1 | hf_chat_runnable | parser

# Test data
review_text = "I recently upgraded to the Samsung Galaxy S24 Ultra, and I'm absolutely impressed! Great camera, fast performance, and excellent battery life."

# Invoke chain with input
result = chain.invoke({"review_text": review_text})
print(json.dumps(result, indent=2))