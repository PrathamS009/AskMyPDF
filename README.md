## AskMyPDF
A basic lightweight RAG application that lets you ask questions over the uploaded PDFs. It extracts the texts from the PDFs, embeds them using Sentence Transformers+FAISS and queries a selected LLM model on hugging face thorugh API. 

## Project Structure:
```
utilities/
 ├── pdf_loader.py      # PDF text extraction and chunking
 ├── vector_store.py    # Embedding + FAISS index
 └── rag_pipeline.py    # RAG logic and model inference
requirements.txt        # For Python 3.10
streamlit_app.py        # Streamlit frontend
.env                    # Environment variables (HF_TOKEN)
```

## Envirnomet Setup
Make a .env file and store your Read-Only token inside it, WITHOUT QUOTES like this:
```
HF_TOKEN=---your token here---
```
## Workflow
1. Run the app: streamlit run streamlit_app.py
2. Upload PDF/s
3. Click "Build Knowledge Base"
4. Select your model
5. Ask your question
6. View the answer given in the below answer field
