from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader

loader = DirectoryLoader(
    path='C:\\Users\\bramh\\OneDrive\\Desktop\\LANG-CHAINS\\RAG-APPLICATION\\langchain-document-loder\\books',
    glob='*.pdf',
    loader_cls=PyPDFLoader
)

docs = loader.lazy_load()

for document in docs:
    print(document.metadata)

print(len(docs))
print(docs[0].page_content)
print(docs[1].metadata)
print(docs[2].metadata)
