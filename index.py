import bs4
from langchain_community.document_loaders.pdf import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

### Indexing ###

# 1. Loading documents

loader = PyPDFLoader(
    file_path="sample_docs/1706.03762.pdf"
)

# loading docs 
docs = loader.load()

# 2. Splitting documents into Chunks
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
)
