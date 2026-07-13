# frontend/app.py
# Author: Jitendra Pandey
# Date Created: 2026-07-12
# Date Modified: 2026-07-13

# Import required libraries
import os
import requests
import streamlit as st

# Fetch the FastAPI backend URL from environment variables or default to localhost
API_URL = os.getenv("API_URL", "http://localhost:8000/query")

# Configure the Streamlit page display and layout styling
st.set_page_config(page_title="RAG Engine for RAM/Chip Design Log Exploration", layout="wide")
st.title("RAG Engine for RAM/Chip Design Log Exploration")
st.caption("CPU/GPU selectable RAG over your logs using FastAPI + Streamlit + Ollama")

# Initialize the chat history in the session state if it does not exist yet
if "messages" not in st.session_state:
    st.session_state.messages = []

# Iterate through and render all previous messages in the chat history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])

# Capture new user input from the chat interface
if user_input := st.chat_input("Ask a question about the logs..."):
    # Append user message to session state and render it to the screen immediately
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.write(user_input)

    # Display a loading spinner while waiting for the RAG backend to respond
    with st.spinner("Searching indexed logs..."):
        try:
            # Send a POST request with the user question to the FastAPI backend
            resp = requests.post(API_URL, json={"question": user_input}, timeout=120)
            data = resp.json()
            
            # Check if the backend returned an explicit error message
            if "error" in data:
                answer = f"Error: {data['error']}"
                sources = []
            else:
                # Extract the answer, metadata, and source chunks from the json response
                answer = data.get("answer", "No answer returned.")
                sources = data.get("sources", [])
                run_mode = data.get("run_mode", "")
                model = data.get("model", "")
                # Format the final response to display interference metadata at the front
                answer = f"[{run_mode}:{model}] {answer}"
        except Exception as e:
            # Handle connection timeouts or network failures gracefully
            answer = f"Backend error: {e}"
            sources = []

    # Append the assistant's response to the history and render it
    st.session_state.messages.append({"role": "assistant", "content": answer})
    with st.chat_message("assistant"):
        st.write(answer)
        
        # If source chunks are returned, render them inside a collapsible expander
        if sources:
            with st.expander("Sources"):
                for i, s in enumerate(sources, 1):
                    st.markdown(f"**Chunk {i}**\n\n{s}")