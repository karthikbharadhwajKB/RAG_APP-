from langchain_community.document_loaders.pdf import PyPDFLoader
from langchain_community.document_loaders.text import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

def load_document(path, doc_type):
    if doc_type == "pdf":
        loader = PyPDFLoader(
            file_path="sample_docs/1706.03762.pdf"
        )
    elif doc_type == "text":
        loader = TextLoader(
            file_path="sample_docs/1706.03762.txt"
        )
    docs = loader.load()
    return docs


def split_document(document, chunk_size, chunk_overlap):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )
    chunks = splitter.split_documents(documents=document)
    return chunks


def get_embedding_model(provider, embed_model_name):
    if provider == "openai":
        from langchain_openai import OpenAIEmbeddings
        embed_model = OpenAIEmbeddings(
            model_name=embed_model_name
        )
    else: 
        print("Provider is not supported...!")
    
