from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
load_dotenv()

model =  ChatOpenAI(model="gpt-4o")
response = model.invoke("Hello, how are you?")
print(response)


