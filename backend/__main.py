import os
import logging
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage, BaseMessage
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
from accelerate import init_empty_weights, infer_auto_device_map, load_checkpoint_and_dispatch
from fastapi_socketio import SocketManager
import asyncio
from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(level=logging.DEBUG)

class QueryRequest(BaseModel):
    query: str

app = FastAPI()

origins = ["http://localhost:5173"]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

socket_manager = SocketManager(app=app)

logging.info("Initializing HuggingFace embeddings...")
embeddings = HuggingFaceEmbeddings(model_name="BAAI/bge-small-en-v1.5")

directory = "chromadb"
logging.info("Initializing Chroma database...")
db = Chroma(
    embedding_function=embeddings,
    persist_directory=directory
)

os.environ["HF_HOME"] = os.path.join(os.getcwd(), "models")
model_path = os.path.join(os.getcwd(), "models", "models--mistralai--Mistral-7B-Instruct-v0.1", "snapshots", "86370fc1f5e0aa51b50dcdf6eada80697b570099")

hf_api_token = os.getenv("HUGGINGFACE_API_TOKEN")

logging.info("Loading Mistral tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(model_path, cache_dir=os.environ["HF_HOME"])

logging.info("Initializing empty model for memory efficient loading...")
config = AutoConfig.from_pretrained(model_path, cache_dir=os.environ["HF_HOME"])
with init_empty_weights():
    model = AutoModelForCausalLM.from_config(config)

offload_folder = os.path.join(os.getcwd(), "offload")

logging.info("Loading model checkpoint with accelerate...")
device_map = infer_auto_device_map(model)
model = load_checkpoint_and_dispatch(model, model_path, device_map=device_map, offload_folder=offload_folder)

logging.info("Model loaded successfully.")

def generate_response(prompt, max_length=200):
    if isinstance(prompt, list):
        prompt = " ".join(message.content for message in prompt)
    logging.debug(f"generate_response received prompt: {prompt}")
    if not isinstance(prompt, str):
        raise ValueError("Prompt must be a string")
    logging.info("Tokenizing input prompt...")
    inputs = tokenizer(prompt, return_tensors="pt")
    logging.info("Generating model response...")
    outputs = model.generate(inputs["input_ids"], max_length=max_length)
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    logging.debug(f"Generated response: {response}")
    return response

logging.info("Creating retriever...")
retriever = db.as_retriever()

contextualize_q_system_prompt = """Given a chat history and the latest user question \
which might reference context in the chat history, formulate a standalone question \
which can be understood without the chat history. Do NOT answer the question, \
just reformulate it if needed and otherwise return it as is."""
contextualize_q_prompt = ChatPromptTemplate.from_messages(
    [
        SystemMessage(content=contextualize_q_system_prompt),
        MessagesPlaceholder(variable_name="chat_history"),
        HumanMessage(content="{input}"),
    ]
)

# Ensuring `input` and `chat_history` are present in the prompt
contextualize_q_prompt.input_variables = ["input", "chat_history"]

logging.info("Creating history aware retriever...")
history_aware_retriever = create_history_aware_retriever(
    llm=model,
    retriever=retriever,
    prompt=contextualize_q_prompt
)

qa_system_prompt = """You are an assistant for question-answering tasks. \
Use the following pieces of retrieved context to answer the question. \
If you don't know the answer, just say that you don't know. \
Use three sentences maximum and keep the answer concise.\

{context}"""
qa_prompt = ChatPromptTemplate.from_messages(
    [
        SystemMessage(content=qa_system_prompt),
        HumanMessage(content="{input}"),
    ]
)

# Ensuring `input` and `context` are present in the prompt
qa_prompt.input_variables = ["input", "context"]

logging.info("Creating question answer chain...")
question_answer_chain = create_retrieval_chain(
    retriever=history_aware_retriever,
    combine_docs_chain=qa_prompt
)

chat_history = []

def get_chat_history():
    formatted_history = []
    for entry in chat_history:
        formatted_history.append(HumanMessage(content=entry['query']))
        formatted_history.append(AIMessage(content=entry['answer']))
    return formatted_history

def extract_answer(response):
    # Check if 'answer' is in response and if it's a ChatPromptValue
    if isinstance(response, dict) and 'answer' in response:
        answer_obj = response['answer']
        if isinstance(answer_obj, dict) and 'messages' in answer_obj:
            messages = answer_obj['messages']
            for message in messages:
                if isinstance(message, AIMessage):
                    return message.content
                elif isinstance(message, dict) and 'content' in message:
                    return message['content']
    return "No answer found"

@app.post("/api/chat")
async def chat(query_request: QueryRequest):
    global chat_history
    formatted_history = get_chat_history()
    logging.debug(f"chat endpoint received query: {query_request.query}")
    response = question_answer_chain.invoke({"input": query_request.query, "chat_history": formatted_history})
    logging.debug(f"Raw response from question_answer_chain: {response}")
    answer = extract_answer(response)
    logging.debug(f"Extracted answer: {answer}")
    chat_history.append({"query": query_request.query, "answer": answer})
    return {"answer": answer}

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    global chat_history
    chat_history = []
    await websocket.send_text("History cleared")
    logging.info("WebSocket connection established and history cleared")
    try:
        while True:
            data = await websocket.receive_text()
            logging.debug(f"WebSocket received data: {data}")
            if not isinstance(data, str):
                raise ValueError("Data received must be a string")
            formatted_history = get_chat_history()
            logging.debug(f"Formatted chat history: {formatted_history}")

            response = question_answer_chain.invoke({"input": data, "chat_history": formatted_history})
            logging.debug(f"Raw response from question_answer_chain: {response}")

            answer = extract_answer(response)
            logging.debug(f"Extracted answer: {answer}")

            chat_history.append({"query": data, "answer": answer})

            if isinstance(answer, str):
                for i in range(0, len(answer), 10):
                    await websocket.send_text(answer[i:i+10])
                    await asyncio.sleep(0.1)
                logging.info("Sent answer through WebSocket")

    except WebSocketDisconnect:
        logging.info("WebSocket connection closed")
    except Exception as e:
        logging.error(f"Unexpected error: {e}")
    finally:
        try:
            await websocket.close()
            logging.info("WebSocket closed successfully")
        except Exception as e:
            logging.error(f"Error closing WebSocket: {e}")

@app.get("/api/history")
async def history():
    global chat_history
    return {"history": chat_history}
