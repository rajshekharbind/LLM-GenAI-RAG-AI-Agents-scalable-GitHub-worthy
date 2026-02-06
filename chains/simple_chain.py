from huggingface_hub import InferenceClient
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv
import os

load_dotenv()

# Get Hugging Face token
hf_token = os.getenv("HUGGINGFACEHUB_ACCESS_TOKEN")

# Create client
client = InferenceClient(api_key=hf_token)

# Create prompt template
prompt = PromptTemplate(
    template="Generate 5 interesting facts about {topic}",
    input_variables=["topic"]
)

# Execute chain manually
prompt_text = prompt.format(topic="cricket")

# Call API
response = client.chat_completion(
    messages=[{"role": "user", "content": prompt_text}],
    model="mistralai/Mistral-7B-Instruct-v0.2"
)

# Parse result
result = response.choices[0].message.content
print(result)