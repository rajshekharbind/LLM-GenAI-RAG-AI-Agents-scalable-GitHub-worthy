from langchain_community.document_loaders import PyPDFLoader

loader = PyPDFLoader('C:\\Users\\bramh\\OneDrive\\Desktop\\LANG-CHAINS\\RAG-APPLICATION\\langchain-document-loder\\dl-curriculum.pdf')

docs = loader.load()

print(len(docs))

print(docs[0].page_content)
print(docs[1].metadata)