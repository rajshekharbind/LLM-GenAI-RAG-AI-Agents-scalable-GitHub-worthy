from huggingface_hub import InferenceClient
from dotenv import load_dotenv
from typing import Optional, Literal
from pydantic import BaseModel, Field
import os
import json
import re

load_dotenv()

hf_token = os.getenv("HUGGINGFACEHUB_ACCESS_TOKEN")
if not hf_token:
    raise ValueError("HUGGINGFACEHUB_ACCESS_TOKEN not found in environment variables. Please add it to your .env file.")

client = InferenceClient(api_key=hf_token)

# schema - Using Pydantic BaseModel
class Review(BaseModel):
    key_themes: list[str] = Field(description="Write down all the key themes discussed in the review in a list")
    summary: str = Field(description="A brief summary of the review")
    sentiment: Literal["pos", "neg"] = Field(description="Return sentiment of the review either negative, positive or neutral")
    pros: Optional[list[str]] = Field(default=None, description="Write down all the pros inside a list")
    cons: Optional[list[str]] = Field(default=None, description="Write down all the cons inside a list")
    name: Optional[str] = Field(default=None, description="Write the name of the reviewer")

# Review text
review_text = """I recently upgraded to the Samsung Galaxy S24 Ultra, and I must say, it's an absolute powerhouse! The Snapdragon 8 Gen 3 processor makes everything lightning fast—whether I'm gaming, multitasking, or editing photos. The 5000mAh battery easily lasts a full day even with heavy use, and the 45W fast charging is a lifesaver.

The S-Pen integration is a great touch for note-taking and quick sketches, though I don't use it often. What really blew me away is the 200MP camera—the night mode is stunning, capturing crisp, vibrant images even in low light. Zooming up to 100x actually works well for distant objects, but anything beyond 30x loses quality.

However, the weight and size make it a bit uncomfortable for one-handed use. Also, Samsung's One UI still comes with bloatware—why do I need five different Samsung apps for things Google already provides? The $1,300 price tag is also a hard pill to swallow.

Pros:
Insanely powerful processor (great for gaming and productivity)
Stunning 200MP camera with incredible zoom capabilities
Long battery life with fast charging
S-Pen support is unique and useful
                                 
Review by Nitish Singh
"""

# Using HuggingFace Inference Client with Pydantic schema
messages = [
    {"role": "user", "content": f"""Analyze this review and return the response in valid JSON format with these fields: key_themes (list), summary (string), sentiment (pos/neg), pros (list), cons (list), name (string).

Review:
{review_text}"""}
]

print("Calling HuggingFace API...")
response = client.chat_completion(
    messages=messages,
    model="mistralai/Mistral-7B-Instruct-v0.2"
)

# Parse response and validate with Pydantic
result_text = response.choices[0].message.content
print("\nRaw Response:")
print(result_text)

# Try to extract JSON and validate with Pydantic
try:
    json_match = re.search(r'\{.*\}', result_text, re.DOTALL)
    if json_match:
        json_str = json_match.group()
        result_data = json.loads(json_str)
        result = Review(**result_data)
        print("\n✓ Validated Result (using Pydantic):")
        print(result)
        print("\nResult as dict:")
        print(result.model_dump())
    else:
        print("\nWarning: Could not find JSON in response")
except json.JSONDecodeError as e:
    print(f"\nJSON parsing error: {e}")
except Exception as e:
    print(f"\nValidation error: {e}")
    print("Raw response returned instead")
