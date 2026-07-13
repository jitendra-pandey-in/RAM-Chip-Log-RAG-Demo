# backend/index.py
# Author: Jitendra Pandey
# Date Created: 2026-07-12
# Date Modified: 2026-07-12

from pathlib import Path
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import TextLoader

from config import LOG_PATH, INDEX_PATH


def build_index():
    loader = TextLoader(str(Path(LOG_PATH)), encoding="utf-8")
    docs = loader.load()

    splitter = RecursiveCharacterTextSplitter(chunk_size=300, chunk_overlap=50)
    chunks = splitter.split_documents(docs)

    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    vectorstore = FAISS.from_documents(chunks, embeddings)
    Path(INDEX_PATH).mkdir(parents=True, exist_ok=True)
    vectorstore.save_local(str(INDEX_PATH))
    print(f"Index built and saved to {INDEX_PATH}")

if __name__ == "__main__":
    build_index()
