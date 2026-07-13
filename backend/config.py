# backend/config.py
# Author: Jitendra Pandey
# Date Created: 2026-07-12
# Date Modified: 2026-07-13

# Import required libraries
import os
import subprocess
from pathlib import Path

# Define a function to check if NVIDIA GPU is available:
def is_nvidia_gpu_available():
    try:
        # Run the 'nvidia-smi' command to check for active drivers and hardware:
        subprocess.check_output('nvidia-smi')
        return True
    except (Exception, FileNotFoundError):
        return False

# Use the function to check GPU availability:
gpubit = is_nvidia_gpu_available()
print(f"Is NVIDIA GPU available: {gpubit}")

# Set run_mode based on GPU availability:
if gpubit:
    RUN_MODE1 = "gpu"
else:
    RUN_MODE1 = "cpu"

RUN_MODE = os.getenv("RUN_MODE", RUN_MODE1)

# Define a function to resolve the project root directory:
def resolve_project_root():
    cwd = os.getcwd()
    if os.path.basename(cwd).lower() == "backend":
        return os.path.dirname(cwd)
    return cwd

# Set project root and configuration variables:
PROJECT_ROOT = os.getenv("PROJECT_ROOT", resolve_project_root())
MODEL_BY_MODE = {
    "cpu": "llama3.2:3b",
    "gpu": "qwen2.5:7b",
}

OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", MODEL_BY_MODE.get(RUN_MODE, "llama3.2:3b"))
OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://localhost:11434")
INDEX_PATH = Path(os.getenv("INDEX_PATH", os.path.join(PROJECT_ROOT, "backend", "ram_index")))
LOG_PATH = os.getenv("LOG_PATH", os.path.join(PROJECT_ROOT, "data", "ram_log.txt"))