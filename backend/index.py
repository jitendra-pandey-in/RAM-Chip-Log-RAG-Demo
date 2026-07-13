# backend/index.py
# Author: Jitendra Pandey
# Date Created: 2026-07-12
# Date Modified: 2026-07-13

# Import required libraries
from pathlib import Path
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import TextLoader
from config import LOG_PATH, INDEX_PATH

# Define a function to read, load and index the log file:
def build_index():
    # Initialize the 'loader' with the path to the log file:
    loader = TextLoader(str(Path(LOG_PATH)), encoding="utf-8")
    # Load the file into 'LangChain' documents object:
    docs = loader.load()

    # Initialize the text splitter to break text into smaller chunks:
    splitter = RecursiveCharacterTextSplitter(chunk_size=300, chunk_overlap=50)
    # Chop the documents into individual overlapping chunks:
    chunks = splitter.split_documents(docs)

    # Initialize the embedding model to convert text chunks into vectors:
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    # Create a FAISS vector store and populate it with the chunks and embeddings:
    vectorstore = FAISS.from_documents(chunks, embeddings)
    # Ensure the target directory exists or create it if missing:
    Path(INDEX_PATH).mkdir(parents=True, exist_ok=True)
    # Save the FAISS index locally to the specified directory path:
    vectorstore.save_local(str(INDEX_PATH))
    # Print a success message confirming the index was built and saved:
    print(f"Index built and saved to {INDEX_PATH}")

# Execute the main script block when this file is run directly:
if __name__ == "__main__":
    build_index()