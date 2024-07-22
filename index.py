from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from utils import (load_document,
                   split_document,
                   get_embedding_model)

### Indexing ###

def ingest_docs(options):

    # loading document
    document = load_document(
        path = options["path"], 
        doc_type=options["doc_type"]
        )

    # splitting document into chunks 
    chunks = split_document(
        document=document, 
        chunk_size=options["chunk_size"], 
        chunk_overlap=options["chunk_overlap"]
        )
    
    # loading embeddings 
    embedding_model = get_embedding_model(
        provider=options["provider"], 
        model_name=options["embed_model_name"]
        )