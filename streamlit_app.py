import streamlit as st
from utilities.pdf_loader import extract_text_from_pdf, chunk_text
from utilities.vector_store import EmbeddingIndex
from utilities.rag_pipeline import RAG
import tempfile
import os

MODEL_OPTIONS = {
    "Llama 3 Instruct (8B)": "meta-llama/Llama-3.1-8B-Instruct",
    "Mistral Chat": "mistralai/Mistral-7B-Instruct-v0.3",
    "Zephyr Lite": "HuggingFaceH4/zephyr-7b-beta",
    "Phi 3 Mini": "microsoft/Phi-3-mini-4k-instruct",
    "Gemma 2 Chat (9B)": "google/gemma-2-9b-it"
}

st.set_page_config(page_title="AskMyPDFs", layout="wide")
st.title("📘 AskMyPDFs — Simple RAG Chat")

st.sidebar.header("Settings")
model_choice = st.sidebar.selectbox("Select Model", list(MODEL_OPTIONS.keys()))
hf_model = MODEL_OPTIONS[model_choice]
st.sidebar.write(f"Using model: `{hf_model}`")

uploaded_files = st.file_uploader("Upload one or more PDFs", type="pdf", accept_multiple_files=True)
build_index = st.button("Build Knowledge Base")

if "rag_obj" not in st.session_state:
    st.session_state.rag_obj = None
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if build_index:
    if not uploaded_files:
        st.warning("Please upload at least one PDF.")
    else:
        all_texts = []
        with tempfile.TemporaryDirectory() as tmpdir:
            for file in uploaded_files:
                temp_path = os.path.join(tmpdir, file.name)
                with open(temp_path, "wb") as f:
                    f.write(file.getbuffer())
                text = extract_text_from_pdf(temp_path)
                all_texts.append(text)

        full_text = "\n\n".join(all_texts)
        chunks = chunk_text(full_text)
        emb = EmbeddingIndex()
        emb.build(chunks)

        rag = RAG(emb, model_name=hf_model)
        st.session_state.rag_obj = rag
        st.success("✅ PDFs loaded and vectorized! You can now ask questions.")

st.markdown("---")
st.subheader("Ask a question")

question = st.text_input("Enter your question here:")
ask_button = st.button("Ask")

if ask_button and question:
    if not st.session_state.rag_obj:
        st.warning("Please upload PDFs and click 'Build Knowledge Base' first.")
    else:
        rag = st.session_state.rag_obj
        answer = rag.ask(question)

        st.session_state.chat_history.append({"role": "user", "text": question})
        st.session_state.chat_history.append({"role": "assistant", "text": answer})

st.markdown("---")
st.subheader("Chat History")

for msg in st.session_state.chat_history:
    if msg["role"] == "user":
        st.markdown(f"** You:** {msg['text']}")
    else:
        st.markdown(f"** Assistant:** {msg['text']}")

if st.button("Clear Chat"):
    st.session_state.chat_history = []
    st.experimental_rerun()
