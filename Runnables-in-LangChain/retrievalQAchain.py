from langchain_openai import ChatOpenAI
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain.llms import OpenAI
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import TextLoader


# Load the documents
loader = TextLoader('documents/sample.pdf')
documents = loader.load()

# Split the documents into smaller chunks
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
docks = text_splitter.split_documents(documents)

# Convert the text into embeddings & store in FAISS vectorstore
vectorstore = FAISS.from_documents(docks, OpenAIEmbeddings())

# Create a retriever
retriever = vectorstore.as_retriever(search_type='similarity', search_kwargs={'k':2})

# Initialize the LLM
llm = OpenAI(model_name='gpt-3.5-turbo', temperature=0)

# Create the RetrievalQA chain
qa_chain = RetrievalQA.from_chain_type(llm=llm, chain_type='stuff', retriever=retriever)

# Define the query
query = "What are the advantages of support vector machines?"
answer = qa_chain.run(query)

# Print the answer

print("Answer: ", answer)