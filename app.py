import os
import streamlit as st

from PyPDF2 import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_ollama import OllamaLLM
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain.tools import tool
from langchain.agents import create_agent

# --- Load Environment ---

# --- Initialize Models ---
llm = OllamaLLM(model="gemma3:1b", temperature=0.3)
llm1 = ChatGoogleGenerativeAI(model="gemini-2.5-flash", google_api_key="AIzaSyCOaH-TNjc4D6CTZQiJG6l5jIdLPGR92hU")
embeddings = OllamaEmbeddings(model="nomic-embed-text:latest")

# --- Streamlit Setup ---
st.set_page_config(page_title="LangChain Agent PDF Chat", layout="wide")
st.title("📄 Chat with PDF using LangChain Agents")

# --- Helper Functions ---
def get_pdf_texts(pdf_file):
    text = ""
    if pdf_file:
        pdf_reader = PdfReader(pdf_file)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text

def get_text_chunks(text):
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200, add_start_index=True)
    return splitter.split_text(text)

def create_vector_store(chunks):
    documents = [Document(page_content=chunk) for chunk in chunks]
    vector_store = FAISS.from_documents(documents, embedding=embeddings)
    vector_store.save_local("faiss_index")
    return vector_store

def load_vector_store():
    if os.path.exists("faiss_index"):
        return FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
    return None

# --- Tool Definition ---
@tool(response_format="content_and_artifact")
def retrieve_context(query: str):
    """Retrieve information to help answer a query."""
    store = load_vector_store()
    if not store:
        return "No vector store found. Please upload and process a PDF.", []
    retrieved_docs = store.similarity_search(query, k=2)
    serialized = "\n\n".join(
        (f"Source: {doc.metadata}\nContent: {doc.page_content}") for doc in retrieved_docs
    )
    return serialized, retrieved_docs

# --- Automatically Load or Process PDF ---
pdf_file_path = "python-for-everybody.pdf"  # Replace with the path to your PDF document
if os.path.exists(pdf_file_path):
    st.sidebar.success("PDF file found, processing...")
    with open(pdf_file_path, "rb") as pdf_file:
        raw_text = get_pdf_texts(pdf_file)
        chunks = get_text_chunks(raw_text)
        vector_store = load_vector_store()
        if not vector_store:
            st.sidebar.warning("Vector store not found. Creating one...")
            create_vector_store(chunks)
        else:
            st.sidebar.success("Vector store loaded successfully!")
else:
    st.sidebar.warning("No PDF found. Please ensure your PDF is available for processing.")

# --- Agent Setup ---
system_prompt = st.sidebar.text_area(
    "System Prompt",
    value="You have access to a tool that retrieves context from a text document. Use the tool to help answer user queries. These queries are related to the Python programming language, so try to add code examples and explanations as necessary.",
    height=150
)
agent = create_agent(model=llm1, tools=[retrieve_context], system_prompt=system_prompt)

# --- Main Interface: Query Input ---
query = st.text_input("💬 Enter your query:", value="Which are the advanced topics in Python?")
if st.button("Run Agent"):
    with st.spinner("Thinking..."):
        response_container = st.empty()
        full_response = ""
        for event in agent.stream({"messages": [{"role": "user", "content": query}]}, stream_mode="values"):
            message = event["messages"][-1]
            
            # Directly access the 'content' attribute of the message
            message_content = str(message.content) if hasattr(message, 'content') else str(message)
            
            full_response += message_content + "\n"
            response_container.markdown(full_response)

        

        # for event in agent.stream({"messages": [{"role": "user", "content": query}]}, stream_mode="values"):
        #     message = event["messages"][-1]
        #     full_response += message + "\n"
        #     response_container.markdown(full_response)

# --- Footer ---
st.markdown("---")
st.caption("Powered by LangChain, Ollama, Gemini & Streamlit")
