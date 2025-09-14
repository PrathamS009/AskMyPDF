# app.py
import streamlit as st
import os
from backend.query_engine import ask_question
from backend.rag_pipeline import build_or_load_index, add_to_index

st.set_page_config(page_title="Research PDF Assistant", layout="wide")

st.title("📚 Research PDF Assistant")

save_dir = os.path.join("data", "uploaded_pdfs")
os.makedirs(save_dir, exist_ok=True)

# Always load index at startup (old PDFs included)
if "index" not in st.session_state:
    with st.spinner("Loading index..."):
        st.session_state.index = build_or_load_index(save_dir)

uploaded_file = st.file_uploader("Upload a PDF", type="pdf")

if uploaded_file is not None:
    save_path = os.path.join(save_dir, uploaded_file.name)

    if not os.path.exists(save_path):  # new file only
        with open(save_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        st.success(f"Uploaded: {uploaded_file.name}")

        with st.spinner("Updating index..."):
            st.session_state.index = add_to_index(save_path)

    else:
        st.info("This PDF already exists. Using stored index.")

st.info("Ask your questions below 👇")

question = st.text_input("Enter your question about the PDFs:")

if st.button("Ask") and question:
    with st.spinner("Thinking..."):
        response = ask_question(question, st.session_state.index)
    st.write("### Answer:")
    st.write(response)

