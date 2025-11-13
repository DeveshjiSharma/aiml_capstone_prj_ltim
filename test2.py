import os
import json
import hashlib
import io
import sys
from contextlib import redirect_stdout, redirect_stderr
from datetime import datetime
from typing import Dict, List, Optional
import streamlit as st

from PyPDF2 import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings, ChatOllama,OllamaLLM
from langchain_community.vectorstores import FAISS
from langchain_community.retrievers import BM25Retriever

from langchain_core.documents import Document

# Simple local EnsembleRetriever implementation to avoid import resolution issues.
# It merges results from multiple retrievers and removes duplicate documents by page_content.
class EnsembleRetriever:
    def __init__(self, retrievers: List, weights: Optional[List[float]] = None):
        self.retrievers = retrievers
        self.weights = weights or [1.0] * len(retrievers)

    def get_relevant_documents(self, query: str):
        combined: List[Document] = []
        seen = set()
        for retriever in self.retrievers:
            try:
                docs = retriever.get_relevant_documents(query)
            except AttributeError:
                # Fallback in case a retriever exposes a different method name
                try:
                    docs = retriever.get_documents(query)
                except Exception:
                    docs = []
            for doc in docs:
                key = getattr(doc, "page_content", str(doc))
                if key not in seen:
                    seen.add(key)
                    combined.append(doc)
        return combined
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.tools import tool
from langchain.agents import create_agent
from langchain_core.messages import HumanMessage

# Try to import Redis (optional)
try:
    import redis
    REDIS_AVAILABLE = True
    redis_client = redis.Redis(host='localhost', port=6379, db=0, decode_responses=True)
    redis_client.ping()
except:
    REDIS_AVAILABLE = False
    redis_client = None

# ============================================================================
# CONFIGURATION
# ============================================================================
st.set_page_config(
    page_title="🎓 Python Learning Assistant",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize Models
llm = ChatOllama(model="mistral", temperature=0.7)  # Changed to mistral as you mentioned
embeddings = OllamaEmbeddings(model="nomic-embed-text:latest")

# Initialize session state
if "messages" not in st.session_state:
    st.session_state.messages = []
if "user_id" not in st.session_state:
    st.session_state.user_id = "student_1"
if "user_profiles" not in st.session_state:
    st.session_state.user_profiles = {}
if "stream_enabled" not in st.session_state:
    st.session_state.stream_enabled = True
if "vector_store" not in st.session_state:
    st.session_state.vector_store = None
if "ensemble_retriever" not in st.session_state:
    st.session_state.ensemble_retriever = None

# ============================================================================
# MEMORY MANAGEMENT
# ============================================================================
class LearningMemory:
    """Manages user learning preferences and progress"""
    
    @staticmethod
    def create_user_profile(user_id: str, level: str = "beginner") -> Dict:
        """Initialize a new user profile"""
        return {
            "level": level,
            "topics_covered": [],
            "learning_history": [],
            "preferences": {
                "code_examples": True,
                "detailed_explanations": True
            },
            "created_at": datetime.now().isoformat()
        }
    
    @staticmethod
    def get_user_profile(user_id: str) -> Dict:
        """Get user profile or create if doesn't exist"""
        if user_id not in st.session_state.user_profiles:
            st.session_state.user_profiles[user_id] = LearningMemory.create_user_profile(user_id)
        return st.session_state.user_profiles[user_id]
    
    @staticmethod
    def update_level(user_id: str, new_level: str):
        """Update user's skill level"""
        profile = LearningMemory.get_user_profile(user_id)
        old_level = profile["level"]
        profile["level"] = new_level
        profile["learning_history"].append({
            "event": "level_change",
            "from": old_level,
            "to": new_level,
            "timestamp": datetime.now().isoformat()
        })
    
    @staticmethod
    def add_topic(user_id: str, topic: str):
        """Track a topic the user has studied"""
        profile = LearningMemory.get_user_profile(user_id)
        if topic not in profile["topics_covered"]:
            profile["topics_covered"].append(topic)
            profile["learning_history"].append({
                "event": "topic_studied",
                "topic": topic,
                "timestamp": datetime.now().isoformat()
            })
    
    @staticmethod
    def add_interaction(user_id: str, question: str):
        """Log a learning interaction"""
        profile = LearningMemory.get_user_profile(user_id)
        
        # Extract potential topic from question
        keywords = ["loop", "function", "class", "variable", "list", "dict", "tuple", 
                   "string", "file", "exception", "module", "decorator", "generator"]
        detected_topic = None
        for keyword in keywords:
            if keyword in question.lower():
                detected_topic = keyword
                LearningMemory.add_topic(user_id, keyword)
                break
        
        profile["learning_history"].append({
            "event": "query",
            "question": question,
            "topic": detected_topic,
            "timestamp": datetime.now().isoformat()
        })
        
        # Keep only last 50 interactions
        if len(profile["learning_history"]) > 50:
            profile["learning_history"] = profile["learning_history"][-50:]
    
    @staticmethod
    def get_learning_context(user_id: str) -> str:
        """Generate context string for the LLM"""
        profile = LearningMemory.get_user_profile(user_id)
        
        context_parts = [
            f"Student Level: {profile['level']}",
            f"Topics Studied: {', '.join(profile['topics_covered'][-5:]) if profile['topics_covered'] else 'None yet'}"
        ]
        
        return " | ".join(context_parts)

# ============================================================================
# CODE EXECUTION TOOLS
# ============================================================================
@tool
def execute_python_code(code: str) -> str:
    """
    Safely execute Python code in a restricted environment.
    Returns the output or error message.
    """
    # Security checks
    forbidden_keywords = [
        'import os', 'import sys', '__import__', 'eval', 'exec',
        'open(', 'file(', 'compile', 'subprocess', 'input('
    ]
    
    for keyword in forbidden_keywords:
        if keyword in code.lower():
            return f"❌ Error: '{keyword}' is not allowed for security reasons"
    
    if len(code) > 1000:
        return "❌ Error: Code is too long (max 1000 characters)"
    
    stdout_capture = io.StringIO()
    stderr_capture = io.StringIO()
    
    try:
        with redirect_stdout(stdout_capture), redirect_stderr(stderr_capture):
            namespace = {
                '__builtins__': {
                    'print': print, 'range': range, 'len': len,
                    'str': str, 'int': int, 'float': float,
                    'bool': bool, 'list': list, 'dict': dict,
                    'tuple': tuple, 'set': set, 'sum': sum,
                    'max': max, 'min': min, 'abs': abs,
                    'round': round, 'sorted': sorted,
                    'enumerate': enumerate, 'zip': zip,
                }
            }
            exec(code, namespace)
        
        output = stdout_capture.getvalue()
        errors = stderr_capture.getvalue()
        
        if errors:
            return f"⚠️ Warnings:\n{errors}\n\n✅ Output:\n{output}" if output else f"❌ Error:\n{errors}"
        elif output:
            return f"✅ Output:\n{output}"
        else:
            return "✅ Code executed successfully (no output)"
            
    except Exception as e:
        return f"❌ Error: {type(e).__name__}: {str(e)}"

@tool
def get_code_hints(topic: str) -> str:
    """Get hints and tips for a specific Python topic."""
    hints_db = {
        'loop': """💡 Loop Tips:
• for loops: Use when you know iterations
• while loops: Use when condition-based
• range(n): Generates 0 to n-1
• break: Exit loop early
• continue: Skip to next iteration""",
        
        'function': """💡 Function Tips:
• Define with 'def function_name():'
• Use parameters to pass data
• Use 'return' to send data back
• Functions can call other functions
• Add docstrings for documentation""",
        
        'list': """💡 List Tips:
• Create: my_list = [1, 2, 3]
• Access: my_list[0] (first item)
• Add: my_list.append(4)
• Remove: my_list.remove(2)
• Loop: for item in my_list:""",
        
        'dict': """💡 Dictionary Tips:
• Create: d = {"key": "value"}
• Access: d["key"] or d.get("key")
• Add/Update: d["new_key"] = "value"
• Loop: for k, v in d.items():
• Keys must be immutable"""
    }
    
    topic_lower = topic.lower()
    for key, hints in hints_db.items():
        if key in topic_lower:
            return hints
    
    return "💡 General Tips:\n• Write clean, readable code\n• Use meaningful names\n• Comment your code\n• Test frequently"

# ============================================================================
# HYBRID RAG RETRIEVAL TOOL
# ============================================================================
@tool(response_format="content_and_artifact")
def retrieve_context(query: str):
    """Retrieve information using hybrid search (FAISS + BM25)."""
    if not st.session_state.ensemble_retriever:
        return "No vector store found. Please process documents first.", []
    
    # Check cache first (if Redis available)
    cache_key = None
    if REDIS_AVAILABLE and redis_client:
        user_profile = LearningMemory.get_user_profile(st.session_state.user_id)
        cache_key = f"query:{user_profile['level']}:{hashlib.md5(query.encode()).hexdigest()}"
        cached = redis_client.get(cache_key)
        if cached:
            st.sidebar.success("💨 Using cached result")
            return cached, []
    
    # Retrieve using hybrid search
    retrieved_docs = st.session_state.ensemble_retriever.get_relevant_documents(query)
    
    # Format context
    user_profile = LearningMemory.get_user_profile(st.session_state.user_id)
    serialized = f"Student Level: {user_profile['level']}\n\n"
    serialized += "\n\n".join(
        (f"Content: {doc.page_content}") for doc in retrieved_docs
    )
    
    # Cache result
    if REDIS_AVAILABLE and redis_client and cache_key:
        redis_client.setex(cache_key, 3600, serialized)  # Cache for 1 hour
    
    return serialized, retrieved_docs

# ============================================================================
# DOCUMENT PROCESSING
# ============================================================================
def get_pdf_texts(pdf_file):
    """Extract text from PDF"""
    text = ""
    if pdf_file:
        pdf_reader = PdfReader(pdf_file)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text

def get_text_chunks(text):
    """Split text into chunks"""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50,
        separators=["\n\n", "\n", ".", " "]
    )
    return splitter.split_text(text)

def create_hybrid_retriever(chunks):
    """Create hybrid retriever (FAISS + BM25)"""
    documents = [Document(page_content=chunk) for chunk in chunks]
    
    # FAISS for semantic search
    vector_store = FAISS.from_documents(documents, embedding=embeddings)
    vector_store.save_local("faiss_index")
    faiss_retriever = vector_store.as_retriever(search_kwargs={"k": 3})
    
    # BM25 for keyword search
    bm25_retriever = BM25Retriever.from_documents(documents)
    bm25_retriever.k = 3
    
    # Ensemble retriever
    ensemble_retriever = EnsembleRetriever(
        retrievers=[faiss_retriever, bm25_retriever],
        weights=[0.5, 0.5]
    )
    
    st.session_state.vector_store = vector_store
    st.session_state.ensemble_retriever = ensemble_retriever
    
    return ensemble_retriever

def load_vector_store():
    """Load existing vector store"""
    if os.path.exists("faiss_index"):
        vector_store = FAISS.load_local(
            "faiss_index", 
            embeddings,
            allow_dangerous_deserialization=True
        )
        st.session_state.vector_store = vector_store
        return vector_store
    return None

# ============================================================================
# MAIN UI
# ============================================================================
st.title("Python Tutor")
st.caption("Hybrid RAG • Memory • Code Execution • Streaming • Caching")

# ============================================================================
# SIDEBAR
# ============================================================================
with st.sidebar:
    st.title("⚙️ Settings")
    
    # User Profile Section
    st.subheader("👤 User Profile")
    st.session_state.user_id = st.text_input(
        "Student ID",
        value=st.session_state.user_id,
        help="Your unique identifier"
    )
    
    profile = LearningMemory.get_user_profile(st.session_state.user_id)
    
    new_level = st.selectbox(
        "Skill Level",
        ["beginner", "intermediate", "advanced"],
        index=["beginner", "intermediate", "advanced"].index(profile["level"])
    )
    
    if new_level != profile["level"]:
        LearningMemory.update_level(st.session_state.user_id, new_level)
        st.success(f"Level updated to {new_level}!")
    
    st.divider()
    
    # Document Processing
    st.subheader("📄 Document Management")
    
    pdf_file_path = st.text_input("PDF Path", value="python-for-everybody.pdf")
    
    if st.button("🔄 Process Document"):
        if os.path.exists(pdf_file_path):
            with st.spinner("Processing PDF..."):
                with open(pdf_file_path, "rb") as pdf_file:
                    raw_text = get_pdf_texts(pdf_file)
                    chunks = get_text_chunks(raw_text)
                    create_hybrid_retriever(chunks)
                    st.success(f"✅ Processed {len(chunks)} chunks!")
        else:
            st.error("PDF file not found!")
    
    # Try auto-load
    if not st.session_state.ensemble_retriever:
        vector_store = load_vector_store()
        if vector_store and os.path.exists(pdf_file_path):
            with open(pdf_file_path, "rb") as pdf_file:
                raw_text = get_pdf_texts(pdf_file)
                chunks = get_text_chunks(raw_text)
                documents = [Document(page_content=chunk) for chunk in chunks]
                
                faiss_retriever = vector_store.as_retriever(search_kwargs={"k": 3})
                bm25_retriever = BM25Retriever.from_documents(documents)
                bm25_retriever.k = 3
                
                st.session_state.ensemble_retriever = EnsembleRetriever(
                    retrievers=[faiss_retriever, bm25_retriever],
                    weights=[0.5, 0.5]
                )
                st.sidebar.info("📂 Vector store loaded")
    
    st.divider()
    
    # Features
    st.subheader("✨ Features")
    st.session_state.stream_enabled = st.toggle(
        "Enable Streaming",
        value=st.session_state.stream_enabled,
        help="Stream responses token-by-token"
    )
    
    if REDIS_AVAILABLE:
        st.success("✅ Redis caching enabled")
        if st.button("Clear Cache"):
            try:
                keys = redis_client.keys('query:*')
                if keys:
                    redis_client.delete(*keys)
                st.success(f"Cleared {len(keys)} cache entries")
            except:
                st.error("Cache clear failed")
    else:
        st.warning("⚠️ Redis not available")
    
    st.divider()
    
    # Progress Stats
    st.subheader("📊 Your Progress")
    st.metric("Level", profile["level"].title())
    st.metric("Topics Covered", len(profile["topics_covered"]))
    
    if profile["topics_covered"]:
        st.write("**Recent Topics:**")
        for topic in profile["topics_covered"][-5:]:
            st.write(f"• {topic}")
    
    st.divider()
    
    # Quick Actions
    st.subheader("🎯 Quick Topics")
    if st.button("📚 Learn Loops"):
        st.session_state.messages.append({
            "role": "user",
            "content": "Teach me about loops in Python with examples"
        })
        st.rerun()
    
    if st.button("🔧 Learn Functions"):
        st.session_state.messages.append({
            "role": "user",
            "content": "How do I create functions in Python?"
        })
        st.rerun()
    
    if st.button("📦 Learn Data Structures"):
        st.session_state.messages.append({
            "role": "user",
            "content": "Explain Python lists and dictionaries"
        })
        st.rerun()
    
    st.divider()
    
    if st.button("🗑️ Clear Chat", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

# ============================================================================
# AGENT SETUP
# ============================================================================
system_prompt = f"""You are a helpful Python programming tutor for {profile['level']} level students.

Context: {LearningMemory.get_learning_context(st.session_state.user_id)}

You have access to these tools:
1. retrieve_context: Search the Python learning materials
2. execute_python_code: Run Python code safely
3. get_code_hints: Get tips for specific topics

Instructions:
- Always tailor explanations to the student's level
- Include code examples when relevant
- Use tools when they help answer questions
- Be encouraging and supportive
- If executing code, explain what it does first
"""

# Create agent with all tools
agent = create_agent(
    model=llm,
    tools=[retrieve_context, execute_python_code, get_code_hints],
    system_prompt=system_prompt
)

# ============================================================================
# CHAT INTERFACE
# ============================================================================
# Display chat messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Chat input
if query := st.chat_input("Ask me anything about Python..."):
    # Add user message
    st.session_state.messages.append({"role": "user", "content": query})
    LearningMemory.add_interaction(st.session_state.user_id, query)
    
    # Display user message
    with st.chat_message("user"):
        st.markdown(query)
    
    # Get assistant response
    with st.chat_message("assistant"):
        response_placeholder = st.empty()
        full_response = ""
        
        try:
            if st.session_state.stream_enabled:
                # Streaming response
                for event in agent.stream(
                    {"messages": [HumanMessage(content=query)]},
                    stream_mode="values"
                ):
                    message = event["messages"][-1]
                    if hasattr(message, 'content'):
                        chunk = str(message.content)
                        if chunk and chunk not in full_response:
                            full_response = chunk
                            response_placeholder.markdown(full_response + "▌")
                
                response_placeholder.markdown(full_response)
            else:
                # Non-streaming response
                with st.spinner("Thinking..."):
                    result = agent.invoke({"messages": [HumanMessage(content=query)]})
                    full_response = str(result["messages"][-1].content)
                    response_placeholder.markdown(full_response)
            
            # Add to chat history
            st.session_state.messages.append({
                "role": "assistant",
                "content": full_response
            })
            
            # Detect code in the response
            # (You could use more sophisticated methods here, e.g. regex or specific markers for code)
            if '```' in full_response:  # Check for markdown code block
                # Extract the code from the markdown block
                start_code = full_response.find('```') + 3
                end_code = full_response.rfind('```')
                code = full_response[start_code:end_code].strip()
                
                # Execute the code
                execution_result = execute_python_code(code)
                
                # Add the result of code execution to the chat history
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": f"💻 **Code Execution Result:**\n{execution_result}"
                })
                response_placeholder.markdown(f"💻 **Code Execution Result:**\n{execution_result}")
            
        except Exception as e:
            error_msg = f"❌ Error: {str(e)}"
            response_placeholder.error(error_msg)
            st.session_state.messages.append({
                "role": "assistant",
                "content": error_msg
            })

# ============================================================================
# FOOTER
# ============================================================================
st.divider()
st.caption("💡 Tip: Try asking 'How do I use for loops?' or 'Execute this code: for i in range(5): print(i)'")

with st.expander("📋 Example Questions"):
    st.markdown("""
    **Beginner:**
    - What are variables in Python?
    - Show me how to use for loops
    - Execute this code: print("Hello World")
    
    **Intermediate:**
    - How do list comprehensions work?
    - What's the difference between lists and tuples?
    - Give me hints about functions
    
    **Advanced:**
    - Explain generators with examples
    - How do decorators work?
    - What are context managers?
    """)

st.caption("Powered by LangChain, Ollama, FAISS, BM25 & Streamlit")
