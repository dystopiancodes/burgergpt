import os
import logging
import torch
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from langchain.chains import ConversationalRetrievalChain
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig, pipeline
from fastapi_socketio import SocketManager
from langchain.memory import ConversationBufferMemory
import asyncio
from dotenv import load_dotenv
from langchain_huggingface import HuggingFacePipeline

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

model_name = "mistralai/Mistral-7B-Instruct-v0.1"

logging.info("Loading Mistral tokenizer and model...")
tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
    trust_remote_code=True,
)

if torch.cuda.is_available():
    model = model.to("cuda")

generation_config = GenerationConfig.from_pretrained(model_name)
generation_config.max_new_tokens = 1024
generation_config.temperature = 0.0001
generation_config.top_p = 0.95
generation_config.do_sample = True
generation_config.repetition_penalty = 1.15

pipe = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    config=generation_config,
)

class CustomHuggingFacePipeline(HuggingFacePipeline):
    def generate(self, prompt, **kwargs):
        max_new_tokens = kwargs.get("max_new_tokens", 1024)
        result = self.pipeline(prompt, max_new_tokens=max_new_tokens)
        # Ensure the response is extracted correctly
        return result[0]['generated_text']

llm = CustomHuggingFacePipeline(pipeline=pipe)

memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

retriever = db.as_retriever()

custom_template = """You are an assistant for question-answering tasks. Given the
following conversation and a follow-up question, rephrase the follow-up question
to be a standalone question. If you don't know the answer, just say that you don't know.
Use three sentences maximum and keep the answer concise.

{context}

{question}"""

CUSTOM_QUESTION_PROMPT = ChatPromptTemplate.from_messages(
    [
        SystemMessage(content=custom_template),
        MessagesPlaceholder(variable_name="chat_history"),
        HumanMessage(content="{input}"),
    ]
)

qa_system_prompt = """You are an assistant for question-answering tasks. Use the following pieces of retrieved context to answer the question. If you don't know the answer, just say that you don't know. Use three sentences maximum and keep the answer concise.

{context}"""

qa_prompt = ChatPromptTemplate.from_messages(
    [
        SystemMessage(content=qa_system_prompt),
        HumanMessage(content="{input}"),
    ]
)

qa_chain = ConversationalRetrievalChain.from_llm(
    llm=llm,
    retriever=retriever,
    memory=memory,
    condense_question_prompt=CUSTOM_QUESTION_PROMPT,
    return_source_documents=True
)

@app.post("/api/chat")
async def chat(query_request: QueryRequest):
    response = qa_chain.invoke({"question": query_request.query})
    answer = response['answer'].strip()
    return {"answer": answer}

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    memory.chat_memory.clear()
    await websocket.send_text("History cleared")
    logging.info("WebSocket connection established and history cleared")
    try:
        while True:
            data = await websocket.receive_text()
            logging.debug(f"WebSocket received data: {data}")
            if not isinstance(data, str):
                raise ValueError("Data received must be a string")

            response = qa_chain.invoke({"question": data})
            answer = response['answer'].strip()
            logging.debug(f"Generated answer: {answer}")

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
    return {"history": memory.chat_memory.messages}
