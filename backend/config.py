# backend/config.py
# Author: Jitendra Pandey
# Date Created: 2024-06-19
# Date Modified: 2024-06-19

import os

RUN_MODE = os.getenv("RUN_MODE", "cpu").lower()

MODEL_BY_MODE = {
    "cpu": "llama3.2:3b",
    "gpu": "qwen2.5:7b",
}

OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", MODEL_BY_MODE.get(RUN_MODE, "llama3.2:3b"))
OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://localhost:11434")
INDEX_PATH = os.getenv("INDEX_PATH", "ram_index")
LOG_PATH = os.getenv("LOG_PATH", "C:/RAG-Demo/data/ram_log.txt")
