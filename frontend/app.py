# frontend/app.py
# AUTHOR: JITENDRA PANDEY
# DATE CREATED: 2026-07-12
# DATE MODIFIED: 2026-07-13

# IMPORT REQUIRED LIBRARIES FOR ENVIRONMENT VARIABLES, HTTP REQUESTS, AND THE USER INTERFACE
import os
import requests
import streamlit as st

# FETCH THE FASTAPI BACKEND URL FROM ENVIRONMENT VARIABLES OR DEFAULT TO LOCALHOST
API_URL = os.getenv("API_URL", "http://localhost:8000/query")

# CONFIGURE THE STREAMLIT PAGE DISPLAY AND LAYOUT STYLING
st.set_page_config(page_title="RAM/Chip Design Log Q&A", layout="wide")
st.title("RAM / Chip Design Log Q&A")
st.caption("CPU/GPU selectable RAG over your logs using FastAPI + Streamlit + Ollama")

# INITIALIZE THE CHAT HISTORY IN THE SESSION STATE IF IT DOES NOT EXIST YET
if "messages" not in st.session_state:
    st.session_state.messages = []

# ITERATE THROUGH AND RENDER ALL PREVIOUS MESSAGES IN THE CHAT HISTORY
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])

# CAPTURE NEW USER INPUT FROM THE CHAT INTERFACE
if user_input := st.chat_input("Ask a question about the logs..."):
    # APPEND USER MESSAGE TO SESSION STATE AND RENDER IT TO THE SCREEN IMMEDIATELY
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.write(user_input)

    # DISPLAY A LOADING SPINNER WHILE WAITING FOR THE RAG BACKEND TO RESPOND
    with st.spinner("Searching indexed logs..."):
        try:
            # SEND A POST REQUEST WITH THE USER QUESTION TO THE FASTAPI BACKEND
            resp = requests.post(API_URL, json={"question": user_input}, timeout=120)
            data = resp.json()
            
            # CHECK IF THE BACKEND RETURNED AN EXPLICIT ERROR MESSAGE
            if "error" in data:
                answer = f"Error: {data['error']}"
                sources = []
            else:
                # EXTRACT THE ANSWER, METADATA, AND SOURCE CHUNKS FROM THE JSON RESPONSE
                answer = data.get("answer", "No answer returned.")
                sources = data.get("sources", [])
                run_mode = data.get("run_mode", "")
                model = data.get("model", "")
                # FORMAT THE FINAL RESPONSE TO DISPLAY INTERFERENCE METADATA AT THE FRONT
                answer = f"[{run_mode}:{model}] {answer}"
        except Exception as e:
            # HANDLE CONNECTION TIMEOUTS OR NETWORK FAILURES GRACEFULLY
            answer = f"Backend error: {e}"
            sources = []

    # APPEND THE ASSISTANT'S RESPONSE TO THE HISTORY AND RENDER IT
    st.session_state.messages.append({"role": "assistant", "content": answer})
    with st.chat_message("assistant"):
        st.write(answer)
        
        # IF SOURCE CHUNKS ARE RETURNED, RENDER THEM INSIDE A COLLAPSIBLE EXPANDER
        if sources:
            with st.expander("Sources"):
                for i, s in enumerate(sources, 1):
                    st.markdown(f"**Chunk {i}**\n\n{s}")