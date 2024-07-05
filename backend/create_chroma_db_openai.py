import os
from langchain_community.document_loaders import DirectoryLoader
from langchain_openai import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma

from dotenv import load_dotenv

load_dotenv()

openai_api_key = os.getenv("OPENAI_API_KEY")
if not openai_api_key:
    raise ValueError("OPENAI_API_KEY environment variable is not set")



def load_docs(directory):
    loader = DirectoryLoader(directory)
    documents = loader.load()
    return documents

def split_docs(documents, chunk_size=500, chunk_overlap=50):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )
    docs = text_splitter.split_documents(documents)
    return docs

if __name__ == "__main__":
    directory = "docs"
    chroma_db_directory = "chromadb"  # Save the database in a separate chromadb directory

    embeddings = OpenAIEmbeddings(api_key=openai_api_key)
    documents = load_docs(directory)
    docs = split_docs(documents)

    db = Chroma.from_documents(
        documents=docs, 
        embedding=embeddings,
        persist_directory=chroma_db_directory  # Save the database in the chromadb directory
    )
    print("Chroma DB created and saved successfully.")
