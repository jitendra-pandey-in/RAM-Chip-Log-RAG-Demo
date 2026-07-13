# backend/config.py
# Author: Jitendra Pandey
# Date Created: 2026-07-12
# Date Modified: 2026-07-12

import os

RUN_MODE = os.getenv("RUN_MODE", "cpu").lower()


def resolve_project_root():
    cwd = os.getcwd()
    if os.path.basename(cwd).lower() == "backend":
        return os.path.dirname(cwd)
    return cwd


PROJECT_ROOT = os.getenv("PROJECT_ROOT", resolve_project_root())
MODEL_BY_MODE = {
    "cpu": "llama3.2:3b",
    "gpu": "qwen2.5:7b",
}

OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", MODEL_BY_MODE.get(RUN_MODE, "llama3.2:3b"))
OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://localhost:11434")
INDEX_PATH = os.getenv("INDEX_PATH", os.path.join(PROJECT_ROOT, "backend", "ram_index"))
LOG_PATH = os.getenv("LOG_PATH", os.path.join(PROJECT_ROOT, "data", "ram_log.txt"))
