# backend/main.py
# AUTHOR: JITENDRA PANDEY
# DATE CREATED: 2026-07-12
# DATE MODIFIED: 2026-07-13

# IMPORT CORE STANDARD AND THIRD-PARTY FRAMEWORK MODULES
from fastapi import FastAPI
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import os
import logging
import traceback
import requests

# IMPORT LANGCHAIN CORE AND INTEGRATION PIPELINES FOR BUILDING THE RAG WORKFLOW
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.llms import Ollama
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

# IMPORT LOCAL SYSTEM DESIGN CONFIGURATIONS FOR RAG
from backend.config import INDEX_PATH, OLLAMA_HOST, RUN_MODE, OLLAMA_MODEL

# CONFIGURE LOGGING ROOT SKELETON TO MATCH FASTAPI UVICORN LOGGER OBJECTS
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("uvicorn.error")

# INITIALIZE THE FASTAPI APP RUNTIME WITH DEBUG CONTROLS ENABLED
app = FastAPI(title="RAM/Chip RAG API", debug=True)

# DEFINE THE SYSTEM PRE-PROMPT STRUCTURE AND STRUCTURAL VARIABLES FOR LLM RUNS
prompt_template = PromptTemplate.from_template(
    "You are a helpful assistant. Use the context below to answer the user's question.\n"
    "If the context is insufficient, say so clearly.\n\n"
    "Context:\n{context}\n\n"
    "Question: {question}\n"
    "Answer:"
)

# INITIALIZE STANDARD TEXT FORMATTER PARSER FOR THE CHAIN OUTPUT
output_parser = StrOutputParser()

# INSTANTIATE LOCAL HUGGINGFACE MODEL PIECE TO COMPUTE VECTOR EMBEDDINGS
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# ESTABLISH OLLAMA LLM TARGET CONFIG WITH HOST TARGET AND TEMPERATURE PROFILE
llm = Ollama(model=OLLAMA_MODEL, base_url=OLLAMA_HOST, temperature=0.2)

# DECLARE GLOBAL REFS TO ALLOW ON-DEMAND LOADS ACROSS THE APP HOOKS
vectorstore = None
retriever = None
qa_chain = None

# HELPER ACTION FUNCTION TO TEST IF THE TARGET OLLAMA CORE IS ONLINE
def check_ollama():
    try:
        r = requests.get(OLLAMA_HOST, timeout=5)
        r.raise_for_status()
        return True
    except Exception:
        return False

# FUNCTION SPECIFYING THE COMPUTATIONAL GRAPH FOR USER QUESTION PROCESSING
def build_query_chain():
    # LAMBDA-STYLE INLINE WORKER TO FETCH CONTEXT VIA THE RETRIEVER OBJ
    def get_context(x):
        docs = retriever.invoke(x["question"])
        return "\n\n".join(doc.page_content for doc in docs)

    # LINK EVERYTHING USING THE LANGCHAIN EXPRESSION LANGUAGE (LCEL) PIPE PIPELINE
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

# ENGINE TO LOAD SERIALIZED VECTOR DATABASES FROM DISK INTO MEMORY STACK
def load_index():
    global vectorstore, retriever, qa_chain

    # BUILD TARGET PATH IDENTIFIERS FOR THE LOCAL STORE ARCHIVES
    faiss_file = INDEX_PATH / "index.faiss"
    pkl_file = INDEX_PATH / "index.pkl"

    # SAFE GUARD CHECKS TO VERIFY IF VECTOR STORES EXIST ON DISK SURFACE
    if not faiss_file.exists() or not pkl_file.exists():
        logger.warning("FAISS index files not found in %s", INDEX_PATH)
        return False

    # READ FAISS ARCHIVES SAFELY BY ACKNOWLEDGING PICKLE RISK PROFILE
    vectorstore = FAISS.load_local(
        str(INDEX_PATH),
        embeddings,
        allow_dangerous_deserialization=True,
    )
    
    # CONFIGURE RETRIEVER HOOK TO EXTRACT THE TOP 3 MATCHING SOURCE RESULTS
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
    
    # ASSEMBLE LCEL GRAPH PIPELINE COMPONENTS NOW THAT STORE IS VALID
    qa_chain = build_query_chain()
    return True

# APPLICATION STARTUP TRIGGER HOOK FOR RUNTIME PREREQ CHECKS
@app.on_event("startup")
def startup_event():
    try:
        # LOG WARNING NOTICES IF PRECONDITIONS ARE ABSENT OR UNRESPONSIVE
        if not check_ollama():
            logger.warning("Ollama is not reachable at %s", OLLAMA_HOST)
        if not load_index():
            logger.warning("Index not loaded. Run backend/index.py first.")
        else:
            logger.info("Backend initialized successfully with model=%s mode=%s", OLLAMA_MODEL, RUN_MODE)
    except Exception:
        logger.exception("Startup failed")

# PYDANTIC MODEL ARCHITECTURE MAP TO CONTROL REQ STRUCTURE AND SCHEMAS
class QueryRequest(BaseModel):
    question: str

# API LIVE SYSTEM STATE HEALTH REPORT ENTRANCE HOOK
@app.get("/health")
def health():
    return {
        "status": "ok",
        "run_mode": RUN_MODE,
        "model": OLLAMA_MODEL,
        "ollama_reachable": check_ollama(),
        "index_loaded": qa_chain is not None and retriever is not None,
    }

# PRIMARY ENDPOINT FOR PROCESSING INCOMING SEARCH QUESTIONS
@app.post("/query")
def query(req: QueryRequest):
    # ERROR EXIT STRATEGY IF THE VECTOR STORE FAILED OR DID NOT LOAD AT STARTUP
    if qa_chain is None or retriever is None:
        return JSONResponse(
            {"error": "The FAISS index has not been built yet. Run backend/index.py first."},
            status_code=503,
        )

    # ERROR EXIT STRATEGY IF THE DEPLOYED HOST FOR LLM GENERATION STAYS TIMED OUT
    if not check_ollama():
        return JSONResponse(
            {"error": f"Ollama is not reachable at {OLLAMA_HOST}"},
            status_code=503,
        )

    try:
        # EXECUTE FULL PROCESS FOR PROMPT COMBINATION + LLM PIPELINE PROCESSING
        result = qa_chain.invoke({"question": req.question})
        
        # EXTRACT SEPARATE ORIGINAL DOCUMENT CONTENT PARTS FOR FRONTEND REFERENCE VIEWING
        docs = retriever.invoke(req.question)
        sources = [doc.page_content for doc in docs]

        # DISPATCH COMPLETELY PROCESSED RAG BUNDLE PACKAGE TO THE USER
        return JSONResponse(
            {
                "answer": result,
                "sources": sources,
                "model": OLLAMA_MODEL,
                "run_mode": RUN_MODE,
            }
        )
    except Exception as e:
        # INTERCEPT FAULTS AND SHIP CLEAN ERROR DIAGNOSTIC BUNDLES FOR LOG VIEWERS
        logger.exception("Exception in /query")
        return JSONResponse(
            {
                "error": str(e),
                "traceback": traceback.format_exc(),
            },
            status_code=500,
        )