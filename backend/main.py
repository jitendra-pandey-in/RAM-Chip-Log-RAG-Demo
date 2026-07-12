# backend/main.py
# Author: Jitendra Pandey
# Date Created: 2024-06-19
# Date Modified: 2024-06-19

from fastapi import FastAPI
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from pathlib import Path
import os
import logging
import traceback
import requests

from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.llms import Ollama
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("uvicorn.error")

app = FastAPI(title="RAM/Chip RAG API", debug=True)

INDEX_PATH = Path(__file__).resolve().parent / "ram_index"
OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://localhost:11434")
RUN_MODE = os.getenv("RUN_MODE").lower()

MODEL_BY_MODE = {
    "cpu": "llama3.2:3b",
    "gpu": "qwen2.5:7b",
}

OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", MODEL_BY_MODE.get(RUN_MODE, "llama3.2:3b"))

prompt_template = PromptTemplate.from_template(
    "You are a helpful assistant. Use the context below to answer the user's question.\n"
    "If the context is insufficient, say so clearly.\n\n"
    "Context:\n{context}\n\n"
    "Question: {question}\n"
    "Answer:"
)

output_parser = StrOutputParser()

embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
llm = Ollama(model=OLLAMA_MODEL, base_url=OLLAMA_HOST, temperature=0.2)

vectorstore = None
retriever = None
qa_chain = None

def check_ollama():
    try:
        r = requests.get(OLLAMA_HOST, timeout=5)
        r.raise_for_status()
        return True
    except Exception:
        return False

def build_query_chain():
    def get_context(x):
        docs = retriever.invoke(x["question"])
        return "\n\n".join(doc.page_content for doc in docs)

    chain = (
        {
            "context": get_context,
            "question": lambda x: x["question"],
        }
        | prompt_template
        | llm
        | output_parser
    )
    return chain

def load_index():
    global vectorstore, retriever, qa_chain

    faiss_file = INDEX_PATH / "index.faiss"
    pkl_file = INDEX_PATH / "index.pkl"

    if not faiss_file.exists() or not pkl_file.exists():
        logger.warning("FAISS index files not found in %s", INDEX_PATH)
        return False

    vectorstore = FAISS.load_local(
        str(INDEX_PATH),
        embeddings,
        allow_dangerous_deserialization=True,
    )
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
    qa_chain = build_query_chain()
    return True

@app.on_event("startup")
def startup_event():
    try:
        if not check_ollama():
            logger.warning("Ollama is not reachable at %s", OLLAMA_HOST)
        if not load_index():
            logger.warning("Index not loaded. Run backend/index.py first.")
        else:
            logger.info("Backend initialized successfully with model=%s mode=%s", OLLAMA_MODEL, RUN_MODE)
    except Exception:
        logger.exception("Startup failed")

class QueryRequest(BaseModel):
    question: str

@app.get("/health")
def health():
    return {
        "status": "ok",
        "run_mode": RUN_MODE,
        "model": OLLAMA_MODEL,
        "ollama_reachable": check_ollama(),
        "index_loaded": qa_chain is not None and retriever is not None,
    }

@app.post("/query")
def query(req: QueryRequest):
    if qa_chain is None or retriever is None:
        return JSONResponse(
            {"error": "The FAISS index has not been built yet. Run backend/index.py first."},
            status_code=503,
        )

    if not check_ollama():
        return JSONResponse(
            {"error": f"Ollama is not reachable at {OLLAMA_HOST}"},
            status_code=503,
        )

    try:
        result = qa_chain.invoke({"question": req.question})
        docs = retriever.invoke(req.question)
        sources = [doc.page_content for doc in docs]

        return JSONResponse(
            {
                "answer": result,
                "sources": sources,
                "model": OLLAMA_MODEL,
                "run_mode": RUN_MODE,
            }
        )
    except Exception as e:
        logger.exception("Exception in /query")
        return JSONResponse(
            {
                "error": str(e),
                "traceback": traceback.format_exc(),
            },
            status_code=500,
        )