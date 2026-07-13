# backend/config.py
# AUTHOR: JITENDRA PANDEY
# DATE CREATED: 2026-07-12
# DATE MODIFIED: 2026-07-13

#IMPORT REQUIRED LIBRARIES
import os
import subprocess
from pathlib import Path

# DEFINE A FUNCTION TO CHECK IF NVIDIA GPU IS AVAILABLE:
def is_nvidia_gpu_available():
    try:
        # RUN THE 'nvidia-smi' COMMAND TO CHECK FOR ACTIVE DRIVERS AND HARDWARE:
        subprocess.check_output('nvidia-smi')
        return True
    except (Exception, FileNotFoundError):
        return False

# USE THE FUNCTION TO CHECK GPU AVAILABILITY:
gpubit = is_nvidia_gpu_available()
print(f"Is NVIDIA GPU available: {gpubit}")

# SET RUN_MODE BASED ON GPU AVAILABILITY:
if gpubit:
    RUN_MODE1 = "gpu"
else:
    RUN_MODE1 = "cpu"

RUN_MODE = os.getenv("RUN_MODE", RUN_MODE1)

# DEFINE A FUNCTION TO RESOLVE THE PROJECT ROOT DIRECTORY:
def resolve_project_root():
    cwd = os.getcwd()
    if os.path.basename(cwd).lower() == "backend":
        return os.path.dirname(cwd)
    return cwd

# SET PROJECT ROOT AND CONFIGURATION VARIABLES:
PROJECT_ROOT = os.getenv("PROJECT_ROOT", resolve_project_root())
MODEL_BY_MODE = {
    "cpu": "llama3.2:3b",
    "gpu": "qwen2.5:7b",
}

OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", MODEL_BY_MODE.get(RUN_MODE, "llama3.2:3b"))
OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://localhost:11434")
INDEX_PATH = Path(os.getenv("INDEX_PATH", os.path.join(PROJECT_ROOT, "backend", "ram_index")))
LOG_PATH = os.getenv("LOG_PATH", os.path.join(PROJECT_ROOT, "data", "ram_log.txt"))