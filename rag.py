from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
# loading all the environment variables
load_dotenv()

### RAG Pipeline  ###
# LLM model 
chat_llm = ChatOpenAI(
    model="gpt-3.5-turbo"
)
