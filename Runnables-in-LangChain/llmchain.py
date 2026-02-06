from langchain.llms import OpenAi
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate

#Load the LLM model
llm = OpenAi(model_name='gpt-3.5-turbo', temperature=0)

#Create a prompt template
prompt = PromptTemplate(
    input_variables=['topic'],
    template='Answer the following question: {topic}'
)

#Create the LLM chain
chain = LLMChain(llm=llm, prompt=prompt)

#Run the chain with a specific topic
topic = input('Enter the topic: ')
output = chain.run(topic=topic)

print("Output: ", output)