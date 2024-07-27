from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from utils import (load_document,
                   split_document,
                   get_embedding_model)

def ingest_docs(options):
    """
    Ingest documents into vector store. 

    Loads documents from file system.
    split the whole document into chunks. 
    embed the chunks. 
    stores them into vector store. 

    Args: 
        - options: dictionary 

    Returns: 
        - None 

    Raises: 
        - None

    """
    

