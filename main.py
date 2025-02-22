from langchain import hub
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_community.document_loaders import WebBaseLoader
import bs4
import tkinter as tk
from tkinter import filedialog
from langchain_community.document_loaders import PyPDFLoader
from langchain.chains import create_history_aware_retriever
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from pydantic import BaseModel
import uvicorn
import os
from fastapi import FastAPI, UploadFile, File, HTTPException
from typing import List, Dict, Any, Optional
import shutil
import os

from dotenv import load_dotenv

load_dotenv()


app = FastAPI()

UPLOAD_FOLDER = "uploaded_files"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)  # Ensure upload folder exists

# Store chat history per session
store: Dict[str, BaseChatMessageHistory] = {}

# Store processed data per session
data_store: Dict[str, FAISS] = {}


def get_session_history(session_id: str) -> BaseChatMessageHistory:
    if session_id not in store:
        store[session_id] = ChatMessageHistory()
    return store[session_id]


class DataUploadRequest(BaseModel):
    # docs_choice: int
    urls: list[str]
    session_id: str

class QueryRequest(BaseModel):
    session_id: str
    query: str

# Store the last response in memory
latest_response: Optional[Dict[str, Any]] = None

@app.get("/")
def read_root():
    if latest_response is None:
        return {"message": "FastAPI is running! No response available yet."}
    return latest_response

@app.post("/upload/")
async def upload_file(files: List[UploadFile] = File(...)):
    """
    Endpoint to upload multiple PDF files.
    """
    if not files:
        raise HTTPException(status_code=400, detail="No files selected.")

    uploaded_files = []

    for file in files:
        if file.filename is None or not file.filename.endswith(".pdf"):
            raise HTTPException(status_code=400, detail=f"Invalid file format: {file.filename}. Only PDFs are allowed.")

        if not file.filename:
            raise HTTPException(status_code=400, detail="File name is missing.")
        file_path = os.path.join(UPLOAD_FOLDER, file.filename)

        # Save file to server
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        uploaded_files.append(file.filename)

    return {"uploaded_files": uploaded_files, "message": "Files uploaded successfully."}


@app.post("/init")
def init_session(request: DataUploadRequest):
    """
    Process all uploaded PDF files and optionally process URLs.
    """
    try:
        uploaded_files = os.listdir(UPLOAD_FOLDER)
        if not uploaded_files:
            raise HTTPException(status_code=400, detail="No files found. Please upload PDFs first.")

        all_splits = []

        # Process uploaded PDF files
        for file_name in uploaded_files:
            file_path = os.path.join(UPLOAD_FOLDER, file_name)
            
            loader = PyPDFLoader(file_path)
            docs = loader.load()

            text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
            splits = text_splitter.split_documents(docs)

            all_splits.extend(splits)

        # Process URLs if provided and not empty
        if request.urls:
            processed_urls = [url for url in request.urls if url.lower() != "done"]
            
            if processed_urls:  # Only process URLs if there are valid ones
                loader1 = WebBaseLoader(web_paths=processed_urls)
                docs1 = loader1.load()

                text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
                splits1 = text_splitter.split_documents(docs1)

                all_splits.extend(splits1)

        # Ensure there is data to store
        if not all_splits:
            raise HTTPException(status_code=400, detail="No data to process.")

        embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")

        # Use FAISS to store embeddings
        vectorstore = FAISS.from_documents(documents=all_splits, embedding=embeddings)
        data_store[request.session_id] = vectorstore  # Store the processed data

        return {"message": "Data uploaded and processed successfully!"}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    

@app.post("/query")
def query_rag(request: QueryRequest):
    try:

        global latest_response  # Make it a global variable
        session_id = request.session_id
        query = request.query

        if session_id not in data_store:
            raise HTTPException(status_code=400, detail="Session ID not found. Please upload data first.")
        
        vectorstore = data_store[session_id]  # Retrieve stored data
        retriever = vectorstore.as_retriever()

        model = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0.3)

         # Improved system prompt for an in-depth product breakdown analysis.
        product_system_prompt = (
            "You are an expert product analyst with extensive knowledge across various industries. "
            "Using the retrieved context—which may include product manuals, patents, datasheets, and other technical documents—"
            "perform a comprehensive analysis and breakdown of the product. Your analysis should be thorough and detailed, covering multiple facets of the product. "
            "Please include the following in your answer:\n\n"
            "1. An extensive overview of the product, including its primary function, market positioning, and intended use cases.\n"
            "2. A detailed breakdown of key features, innovative elements, and unique selling propositions.\n"
            "3. In-depth technical specifications and performance metrics, with comparisons to industry standards or competitors where available.\n"
            "4. Critical insights regarding the product’s advantages and any potential limitations.\n"
            "5. Actionable suggestions or recommendations for improvement, supported by evidence from the context.\n\n"
            "- If only limited context is available, provide a short, concise answer that directly addresses the question.\n"
            "- If no relevant context is available, plz respond that 'I dont know the answer'.\n\n"
            "Context:\n{context}"
        )

        # Contextualization prompt to reformulate a standalone product analysis question.
        contextualize_product_q_system_prompt = (
            "Given the chat history and the latest product analysis question—which might reference earlier context—"
            "formulate a standalone question that can be understood without additional context. "
            "Do NOT answer the question; simply rephrase it if necessary."
        )

        contextualize_product_q_prompt = ChatPromptTemplate.from_messages(
            [
                ("system", contextualize_product_q_system_prompt),
                MessagesPlaceholder("chat_history"),
                ("human", "{input}"),
            ]
        )

        # Product QA prompt that integrates the improved system prompt with the chat history.
        product_qa_prompt = ChatPromptTemplate.from_messages(
            [
                ("system", product_system_prompt),
                MessagesPlaceholder("chat_history"),
                ("human", "{input}"),
            ]
        )

        # Setup a history-aware retriever that uses the contextualized prompt.
        history_aware_retriever = create_history_aware_retriever(
            model,retriever, contextualize_product_q_prompt
        )

        # Build the document chain using the product analysis QA prompt.
        question_answer_chain = create_stuff_documents_chain(model,product_qa_prompt)

        # Finally, create the RAG chain by integrating the history-aware retriever with the QA chain.
        rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)



        conversational_rag_chain = RunnableWithMessageHistory(
            rag_chain,
            get_session_history,
            input_messages_key="input",
            history_messages_key="chat_history",
            output_messages_key="answer",
        )

        response = conversational_rag_chain.invoke(
            {"input": query},
            config={"configurable": {"session_id": session_id}},
        )
        
        latest_response = {"response": response["answer"]}  # Update the global variable

        return latest_response

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    port = int(os.getenv("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)