from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv

load_dotenv()
model = ChatGoogleGenerativeAI(
    model="gemini-3-pro-preview",  
    temperature=0
)
response = model.invoke("Hello, how are you?")
print(response.content)

