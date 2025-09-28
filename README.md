# AskMyPDF
Assistant to ask questions on uploaded PDF's and get answers with going through all of it.

Workflow
    1. User uploads the pdf
    2. Uploaded pdf goes to pdf_loader and text chunks are created
    3. Chunks are used to create vector index in vector_store using embeddings from different LLM's
    4. Rag_pipeline is used to map user's question to similar chunks available in vector_store
    5. Chat_memory to keep chat memory
    6. Streamlit UI helps to choose the LLM and get response