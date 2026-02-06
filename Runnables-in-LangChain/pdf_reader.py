from langchain.document_loaders import TextLoader, PDFMinerLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.llms import OpenAI


#laod the documents
loader = TextLoader('documents/sample.pdf')
documents = loader.load()

#split the documents into smaller chunks
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
texts = text_splitter.split_documents(documents)

#convert the text into embeddings & store in FAISS vectorstore

vectorstore = FAISS.from_documents(texts, OpenAIEmbeddings())

#create a retriever
retriever = vectorstore.as_retriever(search_type='similarity', search_kwargs={'k':2})

# Manually Retrieve relevant documents
query = "What are the advantages of support vector machines?"
relevant_docs = retriever.get_relevant_documents(query)
print(f"Number of relevant documents: {len(relevant_docs)}")

# comine Retriever with text into a single prompt for LLM
retrieved_texts = "\n".join([doc.page_content for doc in relevant_docs])

# Initialize the LLM
llm = OpenAI(model_name='gpt-3.5-turbo', temperature=0)

#Manually Pass the retrieved text to LLM
prompt = f"Answer the following question based on the provided context:\n\nContext: {retrieved_texts}\n\nQuestion: {query}\n\nAnswer:"
answer = llm.predict(prompt)

#print the answer
print("Answer: ", answer)

