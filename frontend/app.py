# frontend/app.py
# Author: Jitendra Pandey
# Date Created: 2024-06-19
# Date Modified: 2024-06-19

import os
import requests
import streamlit as st

API_URL = os.getenv("API_URL", "http://localhost:8000/query")

st.set_page_config(page_title="RAM/Chip Design Log Q&A", layout="wide")
st.title("RAM / Chip Design Log Q&A")
st.caption("CPU/GPU selectable RAG over your logs using FastAPI + Streamlit + Ollama.")

if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])

if user_input := st.chat_input("Ask a question about the logs..."):
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.write(user_input)

    with st.spinner("Searching logs..."):
        try:
            resp = requests.post(API_URL, json={"question": user_input}, timeout=120)
            data = resp.json()
            if "error" in data:
                answer = f"Error: {data['error']}"
                sources = []
            else:
                answer = data.get("answer", "No answer returned.")
                sources = data.get("sources", [])
                run_mode = data.get("run_mode", "")
                model = data.get("model", "")
                answer = f"[{run_mode}:{model}] {answer}"
        except Exception as e:
            answer = f"Backend error: {e}"
            sources = []

    st.session_state.messages.append({"role": "assistant", "content": answer})
    with st.chat_message("assistant"):
        st.write(answer)
        if sources:
            with st.expander("Sources"):
                for i, s in enumerate(sources, 1):
                    st.markdown(f"**Chunk {i}**\n\n{s}")