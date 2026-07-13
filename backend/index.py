# backend/index.py
# AUTHOR: JITENDRA PANDEY
# DATE CREATED: 2026-07-12
# DATE MODIFIED: 2026-07-13

from pathlib import Path
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import TextLoader
from config import LOG_PATH, INDEX_PATH

# DEFINE A FUNCTION TO READ, LOAD AND INDEX THE LOG FILE:
def build_index():
    # INITIALIZE THE 'loader' WITH THE PATH TO THE LOG FILE:
    loader = TextLoader(str(Path(LOG_PATH)), encoding="utf-8")
    # LOAD THE FILE INTO 'LangChain' DOCUMENTS OBJECT:
    docs = loader.load()

    # INITIALIZE THE TEXT SPLITTER TO BREAK TEXT INTO SMALLER CHUNKS:
    splitter = RecursiveCharacterTextSplitter(chunk_size=300, chunk_overlap=50)
    # CHOP THE DOCUMENTS INTO INDIVIDUAL OVERLAPPING CHUNKS:
    chunks = splitter.split_documents(docs)

    # INITIALIZE THE EMBEDDING MODEL TO CONVERT TEXT CHUNKS INTO VECTORS:
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    # CREATE A FAISS VECTOR STORE AND POPULATE IT WITH THE CHUNKS AND EMBEDDINGS:
    vectorstore = FAISS.from_documents(chunks, embeddings)
    # ENSURE THE TARGET DIRECTORY EXISTS OR CREATE IT IF MISSING:
    Path(INDEX_PATH).mkdir(parents=True, exist_ok=True)
    # SAVE THE FAISS INDEX LOCALLY TO THE SPECIFIED DIRECTORY PATH:
    vectorstore.save_local(str(INDEX_PATH))
    # PRINT A SUCCESS MESSAGE CONFIRMING THE INDEX WAS BUILT AND SAVED:
    print(f"Index built and saved to {INDEX_PATH}")

# EXECUTE THE MAIN SCRIPT BLOCK WHEN THIS FILE IS RUN DIRECTLY:
if __name__ == "__main__":
    build_index()