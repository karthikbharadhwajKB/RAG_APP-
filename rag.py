from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from utils import (load_document,
                   split_document,
                   get_embedding_model)
from langchain import hub
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

# loading all the environment variables
load_dotenv()


def RAG(options: dict) -> str:
    ### RAG Pipeline  ###
    # LLM model 
    chat_llm = ChatOpenAI(
      model="gpt-3.5-turbo"
    )

    # loading document
    documents = load_document(
        path = options["path"], 
        doc_type=options["doc_type"]
        )

    # splitting document into chunks 
    chunks = split_document(
        document=documents, 
        chunk_size=options["chunk_size"], 
        chunk_overlap=options["chunk_overlap"]
        )
    
    # loading embeddings 
    embedding_model = get_embedding_model(
        provider=options["provider"], 
        model_name=options["embed_model_name"]
        )
    
    # loading vector store 
    vector_store = FAISS.from_documents(
        documents=chunks, 
        embedding_model=embedding_model
        )
    
    # retriever 
    retriever = vector_store.as_retriever(

    )
    
    # output Parser
    output_parser = StrOutputParser()

    # RAG prompt from langchain hub
    prompt = hub.pull("rlm/rag-prompt")

    # formatting: doc post-processing 
    def format_docs(docs):
        """
        Format the documents into a string.
        """
        return "\n\n".join(doc.page_content for doc in docs)
    
    # RAG chain 
    rag_chain = (
        { "context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt 
        | chat_llm
        | output_parser
    )

    response = rag_chain.invoke(options["query"])
    return response