# backend/main.py
# Author: Jitendra Pandey
# Date Created: 2026-07-12
# Date Modified: 2026-07-13

# Import required libraries and frameworks
from fastapi import FastAPI
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import os
import logging
import traceback
import requests

# Import LangChain core and integration pipelines for building the RAG workflow
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.llms import Ollama
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

# Import local system design configurations for RAG
from backend.config import INDEX_PATH, OLLAMA_HOST, RUN_MODE, OLLAMA_MODEL

# Configure logging root skeleton to match FastAPI Uvicorn logger objects
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("uvicorn.error")

# Initialize the FastAPI app runtime with debug controls enabled
app = FastAPI(title="RAM/Chip RAG API", debug=True)

# Define the system pre-prompt structure and structural variables for LLM runs
prompt_template = PromptTemplate.from_template(
    "You are a helpful assistant. Use the context below to answer the user's question.\n"
    "If the context is insufficient, say so clearly.\n\n"
    "Context:\n{context}\n\n"
    "Question: {question}\n"
    "Answer:"
)

# Initialize standard text formatter parser for the chain output
output_parser = StrOutputParser()

# Instantiate local HuggingFace model piece to compute vector embeddings
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# Establish Ollama LLM target config with host target and temperature profile
llm = Ollama(model=OLLAMA_MODEL, base_url=OLLAMA_HOST, temperature=0.2)

# Declare global refs to allow on-demand loads across the app hooks
vectorstore = None
retriever = None
qa_chain = None

# Helper action function to test if the target Ollama core is online
def check_ollama():
    try:
        r = requests.get(OLLAMA_HOST, timeout=5)
        r.raise_for_status()
        return True
    except Exception:
        return False

# Function specifying the computational graph for user question processing
def build_query_chain():
    # Lambda-style inline worker to fetch context via the retriever obj
    def get_context(x):
        docs = retriever.invoke(x["question"])
        return "\n\n".join(doc.page_content for doc in docs)

    # Link everything using the LangChain Expression Language (LCEL) pipe pipeline
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

# Engine to load serialized vector databases from disk into memory stack
def load_index():
    global vectorstore, retriever, qa_chain

    # Build target path identifiers for the local store archives
    faiss_file = INDEX_PATH / "index.faiss"
    pkl_file = INDEX_PATH / "index.pkl"

    # Safe guard checks to verify if vector stores exist on disk surface
    if not faiss_file.exists() or not pkl_file.exists():
        logger.warning("FAISS index files not found in %s", INDEX_PATH)
        return False

    # Read FAISS archives safely by acknowledging pickle risk profile
    vectorstore = FAISS.load_local(
        str(INDEX_PATH),
        embeddings,
        allow_dangerous_deserialization=True,
    )
    
    # Configure retriever hook to extract the top 3 matching source results
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
    
    # Assemble LCEL graph pipeline components now that store is valid
    qa_chain = build_query_chain()
    return True

# Application startup trigger hook for runtime prereq checks
@app.on_event("startup")
def startup_event():
    try:
        # Log warning notices if preconditions are absent or unresponsive
        if not check_ollama():
            logger.warning("Ollama is not reachable at %s", OLLAMA_HOST)
        if not load_index():
            logger.warning("Index not loaded. Run backend/index.py first.")
        else:
            logger.info("Backend initialized successfully with model=%s mode=%s", OLLAMA_MODEL, RUN_MODE)
    except Exception:
        logger.exception("Startup failed")

# Pydantic model architecture map to control req structure and schemas
class QueryRequest(BaseModel):
    question: str

# API live system state health report entrance hook
@app.get("/health")
def health():
    return {
        "status": "ok",
        "run_mode": RUN_MODE,
        "model": OLLAMA_MODEL,
        "ollama_reachable": check_ollama(),
        "index_loaded": qa_chain is not None and retriever is not None,
    }

# Primary endpoint for processing incoming search questions
@app.post("/query")
def query(req: QueryRequest):
    # Error exit strategy if the vector store failed or did not load at startup
    if qa_chain is None or retriever is None:
        return JSONResponse(
            {"error": "The FAISS index has not been built yet. Run backend/index.py first."},
            status_code=503,
        )

    # Error exit strategy if the deployed host for LLM generation stays timed out
    if not check_ollama():
        return JSONResponse(
            {"error": f"Ollama is not reachable at {OLLAMA_HOST}"},
            status_code=503,
        )

    try:
        # Execute full process for prompt combination + LLM pipeline processing
        result = qa_chain.invoke({"question": req.question})
        
        # Extract separate original document content parts for frontend reference viewing
        docs = retriever.invoke(req.question)
        sources = [doc.page_content for doc in docs]

        # Dispatch completely processed RAG bundle package to the user
        return JSONResponse(
            {
                "answer": result,
                "sources": sources,
                "model": OLLAMA_MODEL,
                "run_mode": RUN_MODE,
            }
        )
    except Exception as e:
        # Intercept faults and ship clean error diagnostic bundles for log viewers
        logger.exception("Exception in /query")
        return JSONResponse(
            {
                "error": str(e),
                "traceback": traceback.format_exc(),
            },
            status_code=500,
        )