from langchain_community.document_loaders import CSVLoader

loader = CSVLoader(file_path='C:\\Users\\bramh\\OneDrive\\Desktop\\LANG-CHAINS\\RAG-APPLICATION\\langchain-document-loder\\Social_Network_Ads.csv')

docs = loader.load()

print(len(docs))
print(docs[1])