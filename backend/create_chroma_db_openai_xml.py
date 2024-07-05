import os
import xml.etree.ElementTree as ET
from langchain_openai import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain.schema import Document
from dotenv import load_dotenv

load_dotenv()

openai_api_key = os.getenv("OPENAI_API_KEY")
if not openai_api_key:
    raise ValueError("OPENAI_API_KEY environment variable is not set")

def parse_xml_file(file_path):
    tree = ET.parse(file_path)
    root = tree.getroot()
    elements = []

    # Traverse and collect text from the XML structure
    for elem in root.iter():
        text_content = elem.text if elem.text else ""
        elements.append((elem.tag, text_content.strip()))

    return elements

def load_docs(directory):
    documents = []
    for filename in os.listdir(directory):
        if filename.endswith(".xml"):
            file_path = os.path.join(directory, filename)
            elements = parse_xml_file(file_path)
            text_content = " ".join(f"{tag}: {text}" for tag, text in elements if text)
            documents.append(Document(page_content=text_content, metadata={"source": filename}))
    return documents

def split_docs(documents, chunk_size=500, chunk_overlap=50):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )
    docs = []
    for doc in documents:
        splits = text_splitter.split_text(doc.page_content)
        for i, split in enumerate(splits):
            docs.append(Document(page_content=split, metadata={"source": doc.metadata["source"], "chunk": i}))
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
