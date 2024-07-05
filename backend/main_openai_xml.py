from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import os
from langchain.chains import ConversationalRetrievalChain
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from fastapi_socketio import SocketManager
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage
import asyncio
from fastapi.staticfiles import StaticFiles

from dotenv import load_dotenv

load_dotenv()

openai_api_key = os.getenv("OPENAI_API_KEY")
if not openai_api_key:
    raise ValueError("OPENAI_API_KEY environment variable is not set")

class QueryRequest(BaseModel):
    query: str

app = FastAPI()

# Enable CORS
origins = ["http://localhost:5173"]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Add Socket Manager
socket_manager = SocketManager(app=app)

# Initialize embeddings, model, and database
embeddings = OpenAIEmbeddings(api_key=openai_api_key)
llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0, api_key=openai_api_key)

directory = "chromadb"
db = Chroma(
    embedding_function=embeddings,
    persist_directory=directory
)

# Create retriever
retriever = db.as_retriever()

# Define contextualize question prompt
contextualize_q_system_prompt = """Given a chat history and the latest user question \
which might reference context in the chat history, formulate a standalone question \
which can be understood without the chat history. Do NOT answer the question, \
just reformulate it if needed and otherwise return it as is."""
contextualize_q_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", contextualize_q_system_prompt),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ]
)

# Create history aware retriever
history_aware_retriever = create_history_aware_retriever(
    llm, retriever, contextualize_q_prompt
)

# Define QA system prompt
qa_system_prompt = """You are an assistant for question-answering tasks. \
Use the following pieces of retrieved context to answer the question. \
If you don't know the answer, just say that you don't know. \
Use three sentences maximum and keep the answer concise.\

{context}"""
qa_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", qa_system_prompt),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ]
)

# Create question answer chain
question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)

# Create final retrieval chain
rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)

chat_history = []

def get_chat_history():
    formatted_history = []
    for entry in chat_history:
        formatted_history.append(HumanMessage(content=entry['query']))
        formatted_history.append(AIMessage(content=entry['answer']))
    return formatted_history

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    global chat_history
    # Clear the chat history when a new WebSocket connection is established
    chat_history = []
    await websocket.send_text("History cleared")
    try:
        while True:
            data = await websocket.receive_text()
            formatted_history = get_chat_history()
            response = rag_chain.invoke({"input": data, "chat_history": formatted_history})

            # Log the raw response object
            print("Raw Response: ", response)

            # Ensure the source metadata is included in the response
            answer = response['answer'] if 'answer' in response else response
            documents = response.get('context', [])

            sources = set()  # Use a set to store unique sources

            if documents:
                for doc in documents:
                    # Log each document content and metadata
                    print("Document Content Length: ", len(doc.page_content))
                    print("Document Content (Excerpt): ", doc.page_content[:200])  # Log the first 200 characters
                    print("Document Metadata: ", doc.metadata)
                    
                    # Extract and log the source metadata
                    source = doc.metadata.get('source')
                    if source:
                        source_path = f"/docs/{source}"
                        sources.add(source_path)  # Add to the set to ensure uniqueness
                        # Log each source added
                        print("Source Added: ", source_path)
                
                # Log sources list
                print("Sources List: ", sources)

                # Construct and append source links to the answer
                if sources:
                    source_links = ", ".join(f"[{os.path.basename(source)}]({source})" for source in sources)
                    print("Source Links: ", source_links)
                    answer += f"\n\nSources: {source_links}"

            # Log the final formatted answer
            print("Formatted Answer: ", answer)

            chat_history.append({"query": data, "answer": answer})

            # Ensure 'answer' is a string before slicing
            if isinstance(answer, str):
                for i in range(0, len(answer), 10):
                    await websocket.send_text(answer[i:i+10])
                    await asyncio.sleep(0.1)  # simulate streaming
            else:
                print("Unexpected type for answer, expected a string.")

    except WebSocketDisconnect:
        print("WebSocket connection closed")
    except Exception as e:
        print(f"Unexpected error: {e}")
    finally:
        try:
            await websocket.close()
        except:
            pass







@app.get("/api/history")
async def history():
    global chat_history
    return {"history": chat_history}

# Serve static files from the "docs" directory
app.mount("/docs", StaticFiles(directory="docs"), name="docs")
