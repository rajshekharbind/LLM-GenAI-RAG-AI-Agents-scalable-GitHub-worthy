from langchain.llms  import OpenAI
from dotenv import load_dotenv
from langchain.prompts import PromptTemplate

load_dotenv()

# Initialize the LLM
llm = OpenAI(model_name='gpt-3.5-turbo', temperature=0.7)

#create a prompt template
prompt = PromptTemplate(
    input_variables=['topic'],
    template='Provide a detailed explanation about the topic: {topic}'
)

#define inputs
inputs = {'topic':'Quantum Computing'}

#format the prompt with manually provided prompttemplate
formatted_prompt = prompt.format(topic=inputs['topic'])

# call the llm directly
blog_title = llm.predict(formatted_prompt)

#print the output
print("Blog Title: ", blog_title)