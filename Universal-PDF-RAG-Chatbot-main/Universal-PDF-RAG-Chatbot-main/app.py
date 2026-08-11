mport streamlit as st
import os
import tempfile
import time
import logging
import datetime
import shutil
import html
from pathlib import Path
from typing import List
import textwrap

# Logging Setup
logging.basicConfig(
    filename='app.log',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

def log_event(message: str, level: str = "INFO"):
    """Log event to file and session state"""
    timestamp = datetime.datetime.now().strftime("%H:%M:%S")
    log_entry = f"[{timestamp}] {message}"
    
    if level == "INFO":
        logging.info(message)
    elif level == "WARNING":
        logging.warning(message)
    elif level == "ERROR":
        logging.error(message)
        
    if "app_logs" not in st.session_state:
        st.session_state["app_logs"] = []
    
    # Use HTML entities for emojis in logs to avoid syntax errors
    emoji = '&#128308;' if level=='ERROR' else ('&#9888;' if level=='WARNING' else '&#128313;')
    st.session_state["app_logs"].insert(0, f"{emoji} {log_entry}")
    
    if len(st.session_state["app_logs"]) > 100:
        st.session_state["app_logs"] = st.session_state["app_logs"][:100]

# LangChain imports
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader

try:
    from langchain_groq import ChatGroq
    GROQ_AVAILABLE = True
except:
    GROQ_AVAILABLE = False

try:
    from langchain_community.llms import OpenAI
    OPENAI_AVAILABLE = True
except:
    OPENAI_AVAILABLE = False

# Configuration
EMBEDDING_MODEL_NAME = "sentence-transformers/all-mpnet-base-v2"
CHUNK_SIZE = 800
CHUNK_OVERLAP = 150
TOP_K = 10

GROQ_MODELS = [
    "llama-3.3-70b-versatile",     # Latest powerful model
    "llama-3.1-70b-versatile",     # Stable 70B
    "llama-3.1-8b-instant",        # Fast & efficient
    "mixtral-8x7b-32768",          # Long context
    "gemma2-9b-it",                # Google Gemma 2
    "llama3-70b-8192",             # Original Llama 3 70B
    "llama3-8b-8192",              # Original Llama 3 8B
]

# Page Configuration
st.set_page_config(
    page_title="Universal RAG Chatbot",
    page_icon=":books:",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Yulu-style CSS
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700;800&display=swap');
    * { font-family: 'Inter', sans-serif; }
    
    .main {
        background: linear-gradient(135deg, #0f0c29 0%, #302b63 50%, #24243e 100%);
        background-attachment: fixed;
    }
    .block-container {
        background: rgba(17, 24, 39, 0.85);
        border-radius: 20px;
        padding: 2rem;
        box-shadow: 0 20px 60px rgba(0,0,0,0.7);
        backdrop-filter: blur(20px);
        border: 1px solid rgba(139, 92, 246, 0.2);
    }
    h1 {
        background: linear-gradient(135deg, #a78bfa 0%, #f472b6 50%, #fb923c 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-size: 3.5rem !important;
        font-weight: 800 !important;
        text-align: center;
        margin-bottom: 1rem;
        animation: fadeInDown 1s ease-in-out;
    }
    h2 { 
        color: #f3f4f6 !important; 
        border-bottom: 3px solid #8b5cf6; 
        padding-bottom: 0.5rem; 
        font-weight: 700 !important;
    }
    h3 { color: #e5e7eb !important; font-weight: 600 !important; }
    p, li, span, div { color: #cbd5e1; }
    
    .stTabs [data-baseweb="tab-list"] { 
        gap: 12px; 
        background-color: rgba(17, 24, 39, 0.5);
        padding: 0.5rem;
        border-radius: 12px;
    }
    .stTabs [data-baseweb="tab"] {
        background: linear-gradient(135deg, rgba(139, 92, 246, 0.1) 0%, rgba(244, 114, 182, 0.1) 100%);
        color: #a78bfa; 
        border-radius: 10px; 
        padding: 12px 24px; 
        font-weight: 600; 
        transition: all 0.3s ease; 
        border: 1px solid rgba(139, 92, 246, 0.3);
    }
    .stTabs [data-baseweb="tab"]:hover { 
        background: linear-gradient(135deg, rgba(139, 92, 246, 0.2) 0%, rgba(244, 114, 182, 0.2) 100%);
        transform: translateY(-2px); 
        box-shadow: 0 4px 12px rgba(139, 92, 246, 0.4);
    }
    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, #8b5cf6 0%, #ec4899 100%) !important; 
        color: white !important; 
        box-shadow: 0 6px 20px rgba(139, 92, 246, 0.6);
    }
    
    [data-testid="stSidebar"] { 
        background: linear-gradient(180deg, #1e1b4b 0%, #312e81 100%); 
        border-right: 1px solid rgba(139, 92, 246, 0.3);
    }
    [data-testid="stSidebar"] h1, [data-testid="stSidebar"] h2, [data-testid="stSidebar"] h3 { 
        color: white !important; 
        -webkit-text-fill-color: white !important; 
    }
    
    .stButton > button {
        background: linear-gradient(135deg, #8b5cf6 0%, #ec4899 100%); 
        color: white; 
        border-radius: 12px; 
        padding: 0.75rem 2rem; 
        font-weight: 600; 
        box-shadow: 0 4px 15px rgba(139, 92, 246, 0.4); 
        border: none;
        transition: all 0.3s ease;
    }
    .stButton > button:hover { 
        transform: translateY(-3px); 
        box-shadow: 0 8px 25px rgba(139, 92, 246, 0.6); 
    }
    
    .card {
        padding: 1rem; 
        border-radius: 16px; 
        text-align: center; 
        color: white; 
        box-shadow: 0 8px 32px rgba(0,0,0,0.4); 
        transition: all 0.4s ease; 
        border: 1px solid rgba(255,255,255,0.1);
    }
    .card:hover { 
        transform: translateY(-8px) scale(1.02); 
        box-shadow: 0 12px 40px rgba(139, 92, 246, 0.4);
    }
    
    @keyframes fadeInDown { 
        from { opacity: 0; transform: translateY(-30px); } 
        to { opacity: 1; transform: translateY(0); } 
    }
    
    .stExpander {
        background: rgba(17, 24, 39, 0.5);
        border: 1px solid rgba(139, 92, 246, 0.2);
        border-radius: 12px;
    }
    
    /* Text Area Styling */
    .stTextArea textarea {
        background-color: rgba(17, 24, 39, 0.6) !important;
        color: #e5e7eb !important;
        border: 1px solid rgba(139, 92, 246, 0.3) !important;
        border-radius: 12px !important;
    }
    .stTextArea textarea:focus {
        border-color: #8b5cf6 !important;
        box-shadow: 0 0 0 2px rgba(139, 92, 246, 0.2) !important;
    }
</style>
""", unsafe_allow_html=True)

# Header
st.markdown("""
<div style='text-align: center; padding: 1.2rem 0 0.8rem 0;'>
    <h1 style='font-size: 3rem; margin-bottom: 0;'>&#128218; Universal RAG Chatbot</h1>
</div>
""", unsafe_allow_html=True)

# Simple defaults for the upload/chat flow
has_groq = bool(os.environ.get("GROQ_API_KEY") or st.secrets.get("GROQ_API_KEY", None))
has_openai = bool(os.environ.get("OPENAI_API_KEY") or st.secrets.get("OPENAI_API_KEY", None))
use_groq = GROQ_AVAILABLE and has_groq

# How to Use Workflow
st.markdown("""
<div style='background: linear-gradient(135deg, rgba(139, 92, 246, 0.1) 0%, rgba(244, 114, 182, 0.1) 100%); 
            padding: 1.5rem; border-radius: 16px; border: 1px solid rgba(139, 92, 246, 0.3); margin-bottom: 2rem;'>
    <h3 style='text-align: center; margin-top: 0; color: #a78bfa !important; font-size: 1.2rem;'>&#128640; Quick Guide</h3>
    <div style='display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap; gap: 10px; text-align: center;'>
        <div style='flex: 1; min-width: 120px;'>
            <div style='font-size: 1.5rem; margin-bottom: 5px;'>&#128194;</div>
            <div style='font-weight: 600; color: #fff;'>1. Upload PDF</div>
            <div style='font-size: 0.8rem; color: #cbd5e1;'>Use sidebar/uploader</div>
        </div>
        <div style='font-size: 1.2rem; color: #666;'>➔</div>
        <div style='flex: 1; min-width: 120px;'>
            <div style='font-size: 1.5rem; margin-bottom: 5px;'>&#9203;</div>
            <div style='font-weight: 600; color: #fff;'>2. Wait 20-30s</div>
            <div style='font-size: 0.8rem; color: #cbd5e1;'>Processing docs</div>
        </div>
        <div style='font-size: 1.2rem; color: #666;'>➔</div>
        <div style='flex: 1; min-width: 120px;'>
            <div style='font-size: 1.5rem; margin-bottom: 5px;'>&#128172;</div>
            <div style='font-weight: 600; color: #fff;'>3. Ask Question</div>
            <div style='font-size: 0.8rem; color: #cbd5e1;'>In Chat tab</div>
        </div>
        <div style='font-size: 1.2rem; color: #666;'>➔</div>
        <div style='flex: 1; min-width: 120px;'>
            <div style='font-size: 1.5rem; margin-bottom: 5px;'>&#129302;</div>
            <div style='font-weight: 600; color: #fff;'>4. Wait 20-30s</div>
            <div style='font-size: 0.8rem; color: #cbd5e1;'>AI generating</div>
        </div>
        <div style='font-size: 1.2rem; color: #666;'>➔</div>
        <div style='flex: 1; min-width: 120px;'>
            <div style='font-size: 1.5rem; margin-bottom: 5px;'>&#128218;</div>
            <div style='font-weight: 600; color: #fff;'>5. View Sources</div>
            <div style='font-size: 0.8rem; color: #cbd5e1;'>Check citations</div>
        </div>
    </div>
    <div style='margin-top: 15px; padding-top: 10px; border-top: 1px solid rgba(255,255,255,0.1); font-size: 0.85rem; color: #a78bfa;'>
        &#128161; <strong>Tip:</strong> If answers aren't found, click <strong>"&#128296; Rebuild Index"</strong> in sidebar & re-upload.
    </div>
</div>
""", unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# Helper Functions
def save_uploaded_files(uploaded_files, dest_dir):
    os.makedirs(dest_dir, exist_ok=True)
    saved_paths = []
    for uploaded_file in uploaded_files:
        file_path = os.path.join(dest_dir, uploaded_file.name)
        with open(file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        saved_paths.append(file_path)
    return saved_paths

def load_documents_langchain(dir_path: str) -> List[Document]:
    docs = []
    pdf_files = list(Path(dir_path).glob("*.pdf"))
    
    from langchain_community.document_loaders import PyMuPDFLoader, UnstructuredPDFLoader
    
    for pdf_path in pdf_files:
        try:
            # Try PyMuPDF first (fast for text-based PDFs)
            loader = PyMuPDFLoader(str(pdf_path))
            pages = loader.load()
            
            # Check if we got meaningful text
            total_text = "".join([p.page_content for p in pages])
            
            # If very little text extracted, use OCR mode
            if len(total_text.strip()) < 100:
                log_event(f"Low text content detected in {pdf_path.name}, using OCR mode...")
                try:
                    # Use Unstructured with OCR for image-based PDFs
                    loader = UnstructuredPDFLoader(
                        str(pdf_path),
                        mode="elements",
                        strategy="hi_res"  # High resolution for better OCR
                    )
                    pages = loader.load()
                    log_event(f"OCR extraction completed for {pdf_path.name}")
                except Exception as ocr_error:
                    log_event(f"OCR failed for {pdf_path.name}: {ocr_error}", "WARNING")
                    # Continue with whatever we got from PyMuPDF
            
            # Log sample text to verify extraction
            if pages:
                sample_text = pages[0].page_content[:500].replace('\n', ' ')
                log_event(f"Extracted text sample from {pdf_path.name}: {sample_text}...")
            
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=CHUNK_SIZE,
                chunk_overlap=CHUNK_OVERLAP,
                separators=["\n\n", "\n", ".", "!", "?", ",", " ", ""],
                length_function=len
            )
            
            chunks = text_splitter.split_documents(pages)
            log_event(f"Created {len(chunks)} chunks from {pdf_path.name}")
            
            for chunk in chunks:
                docs.append(Document(
                    page_content=chunk.page_content,
                    metadata={**chunk.metadata, "source": pdf_path.name}
                ))
        except Exception as e:
            st.error(f"Failed to load {pdf_path.name}: {e}")
            log_event(f"Error loading {pdf_path.name}: {e}", "ERROR")
    
    return docs

@st.cache_resource
def get_embeddings():
    return HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME)

def build_faiss_index(docs: List[Document], index_dir: str):
    """
    Build a completely NEW FAISS index for the currently uploaded PDFs.

    We intentionally do not load a global/persistent FAISS index here.
    A global index can cause sources from an earlier PDF to appear when
    a different PDF is uploaded, especially on a deployed multi-user app.
    """
    embeddings = get_embeddings()

    # Always start with a clean index directory.
    shutil.rmtree(index_dir, ignore_errors=True)
    os.makedirs(index_dir, exist_ok=True)

    with st.spinner("&#128296; Building a fresh FAISS index for your PDF(s)..."):
        log_event(f"Building NEW FAISS index from {len(docs)} chunks")
        db = FAISS.from_documents(docs, embeddings)
        db.save_local(index_dir)
        st.success("&#9989; New FAISS index built for the uploaded PDF(s)")
        log_event("New session FAISS index built successfully")

    return db

def get_groq_llm_instance(api_key, model_name):
    return ChatGroq(
        groq_api_key=api_key,
        model_name=model_name,
        temperature=0.0,
        max_tokens=4096
    )

# File Upload
st.header("&#128193; Upload PDFs")
uploaded_files = st.file_uploader(
    "Upload one or more PDF files",
    accept_multiple_files=True,
    type=["pdf"],
    help="Upload PDF documents to create your knowledge base"
)

# Main chat area
if uploaded_files:
    if uploaded_files:
        temp_dir = tempfile.mkdtemp(prefix="rag_chatbot_")
        
        try:
            # Build a signature from filename + size + content hash.
            # This also detects a different PDF with the SAME filename.
            import hashlib

            current_file_signature = []
            for f in uploaded_files:
                file_bytes = f.getvalue()
                current_file_signature.append(
                    (f.name, len(file_bytes), hashlib.sha256(file_bytes).hexdigest())
                )

            previous_signature = st.session_state.get("processed_file_signature")

            # Only process when the uploaded document set is new or changed.
            if previous_signature != current_file_signature:
                log_event(f"Processing {len(uploaded_files)} uploaded files")

                # Remove the previous session index so old documents cannot leak
                # into the new document set.
                old_index_dir = st.session_state.get("session_index_dir")
                if old_index_dir:
                    shutil.rmtree(old_index_dir, ignore_errors=True)

                saved_paths = save_uploaded_files(uploaded_files, temp_dir)
                st.success(f"&#9989; Uploaded {len(saved_paths)} file(s)")

                with st.spinner("&#128214; Loading documents..."):
                    docs = load_documents_langchain(temp_dir)

                    if not docs:
                        st.error("&#10060; No documents loaded")
                        st.stop()

                    st.info(f"&#128196; Loaded {len(docs)} document chunks")

                # Create a separate FAISS index for THIS Streamlit session
                # and THIS exact set of uploaded PDFs.
                session_index_dir = tempfile.mkdtemp(prefix="faiss_session_")
                db = build_faiss_index(docs, session_index_dir)

                # Use ONLY this fresh index for retrieval.
                st.session_state["session_index_dir"] = session_index_dir
                st.session_state["retriever"] = db.as_retriever(
                    search_kwargs={"k": TOP_K}
                )
                st.session_state["processed_file_signature"] = current_file_signature

                # A new document set should start a new conversation.
                st.session_state["messages"] = []

            if "retriever" in st.session_state:
                retriever = st.session_state["retriever"]
                
                # Get API keys
                groq_key = os.environ.get("GROQ_API_KEY") or st.secrets.get("GROQ_API_KEY")
                openai_key = os.environ.get("OPENAI_API_KEY") or st.secrets.get("OPENAI_API_KEY")
                
                # Create enhanced prompt
                prompt = ChatPromptTemplate.from_template("""
You are an expert AI assistant helping users understand their documents.

Context from the uploaded documents:
{context}

Question: {input}

Instructions:
- First, carefully review the Context above for relevant information
- If the Context contains information about the question, provide a detailed answer based on it
- If the Context has partial information, use it and supplement with general knowledge where appropriate
- If the Context doesn't contain relevant information, provide a helpful general answer and note that specific details weren't found in the uploaded documents
- Be clear, accurate, and comprehensive in your response

Answer:
""")
                
                st.markdown("### &#128172; Chat with Your Documents")
                
                # Initialize chat history
                if "messages" not in st.session_state:
                    st.session_state["messages"] = []
                
                # Display chat history
                for role, message in st.session_state["messages"]:
                    if role == "user":
                        st.markdown(f"""
                        <div style='display: flex; gap: 12px; margin-bottom: 1rem;'>
                            <div style='width: 40px; height: 40px; border-radius: 50%; background: linear-gradient(135deg, #FF0099 0%, #493240 100%); display: flex; align-items: center; justify-content: center; font-size: 20px;'>&#128100;</div>
                            <div style='flex: 1; padding: 16px 22px; border-radius: 20px; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white;'>
                                {html.escape(message)}
                            </div>
                        </div>
                        """, unsafe_allow_html=True)
                    else:
                        # Extract debug chunks
                        debug_chunks = []
                        if "|||DEBUG_CHUNKS:" in message:
                            parts_debug = message.split("|||DEBUG_CHUNKS:")
                            message = parts_debug[0]
                            debug_info = parts_debug[1]
                            # Parse chunks
                            if "|||CHUNK" in debug_info:
                                chunk_parts = debug_info.split("|||CHUNK")
                                for chunk in chunk_parts:
                                    if ":" in chunk:
                                        chunk_num, content = chunk.split(":", 1)
                                        debug_chunks.append(f"**Chunk {chunk_num}**: {content.strip()}")

                        # Extract footer and sources
                        footer_text = ""
                        if "|||FOOTER:" in message:
                            parts_footer = message.split("|||FOOTER:")
                            message = parts_footer[0]
                            footer_text = parts_footer[1]
                        
                        parts = message.split("**&#128218; Sources:**")
                        main_answer = parts[0].strip()
                        
                        st.markdown(f"""
                        <div style='display: flex; gap: 12px; margin-bottom: 1rem;'>
                            <div style='width: 40px; height: 40px; border-radius: 50%; background: linear-gradient(135deg, #00F260 0%, #0575E6 100%); display: flex; align-items: center; justify-content: center; font-size: 20px;'>&#129302;</div>
                            <div style='flex: 1; padding: 16px 22px; border-radius: 20px; background: rgba(255, 255, 255, 0.05); border: 1px solid rgba(255, 255, 255, 0.1); color: #e0e0e0;'>
                                {html.escape(main_answer).replace(chr(10), '<br>')}
                            </div>
                        </div>
                        """, unsafe_allow_html=True)
                        
                        if len(parts) > 1:
                            sources_raw = parts[1].strip().split("\n")
                            with st.expander("&#128218; View Sources", expanded=False):
                                for line in sources_raw:
                                    line = line.strip()
                                    if not line:
                                        continue
                                    if line.startswith("**[") and "]**" in line:
                                        st.markdown(f"**{line.replace('**', '')}**")
                                    elif line.startswith(">"):
                                        st.markdown(f"> {line[1:].strip()}")
                        
                        if debug_chunks:
                            with st.expander("&#128269; Debug Context", expanded=False):
                                for chunk in debug_chunks:
                                    st.markdown(chunk)
                                    st.markdown("---")

                        if footer_text:
                            st.markdown(f"<small style='color:#888;'>&#129302; {footer_text}</small>", unsafe_allow_html=True)
                
                # Chat input
                # Chat input area
                def submit_query():
                    if st.session_state.get("query_input"):
                        st.session_state["messages"].append(("user", st.session_state["query_input"]))
                        st.session_state["query_input"] = ""

                st.text_area("Ask a question about your documents...", key="query_input")
                
                # Chat controls
                col_actions1, col_actions2 = st.columns(2)
                with col_actions1:
                    st.button("&#10140; Enter Query", on_click=submit_query, use_container_width=True)
                
                with col_actions2:
                    if st.button("&#128465; Clear Conversation", use_container_width=True):
                        st.session_state["messages"] = []
                        st.rerun()
                
                # Process latest message
                if st.session_state["messages"] and st.session_state["messages"][-1][0] == "user":
                    with st.spinner("&#129504; Thinking..."):
                        latest_query = st.session_state["messages"][-1][1]
                        log_event(f"Processing query: {latest_query[:50]}...")
                        
                        
                        success = False
                        models_to_try = []
                        all_errors = []  # Collect all errors
                        
                        if use_groq and GROQ_AVAILABLE and groq_key:
                            models_to_try.extend([(name, "groq") for name in GROQ_MODELS])
                        
                        if OPENAI_AVAILABLE and openai_key:
                            models_to_try.append(("gpt-3.5-turbo", "openai"))
                        
                        if not models_to_try:
                            st.error("&#10060; No LLM configured. Please set API keys.")
                            st.stop()
                        
                        for model_name, provider in models_to_try:
                            try:
                                if provider == "groq":
                                    current_llm = get_groq_llm_instance(groq_key, model_name)
                                else:
                                    current_llm = OpenAI(temperature=0.0, openai_api_key=openai_key)
                                
                                document_chain = create_stuff_documents_chain(current_llm, prompt)
                                qa_chain = create_retrieval_chain(retriever, document_chain)
                                
                                result = qa_chain.invoke({"input": latest_query})
                                
                                answer = result.get("answer", "No answer generated")
                                source_docs = result.get("context", [])
                                
                                # Log retrieved chunks for debugging
                                log_event(f"Retrieved {len(source_docs)} chunks for query")
                                
                                full_answer = answer
                                full_answer += f"|||FOOTER:Generated with: {model_name}"
                                
                                # Add debug info about retrieved chunks
                                if source_docs:
                                    sources_text = "\n\n**&#128218; Sources:**\n"
                                    for i, doc in enumerate(source_docs, 1):
                                        source = doc.metadata.get("source", f"Document {i}")
                                        raw_snippet = doc.page_content[:200] + "..." if len(doc.page_content) > 200 else doc.page_content
                                        snippet = " ".join(raw_snippet.split())
                                        sources_text += f"\n**[{i}]** {source}\n> {snippet}\n"
                                    full_answer += sources_text
                                    
                                    # Add full retrieved context for debugging (hidden in metadata)
                                    full_answer += f"\n\n|||DEBUG_CHUNKS:{len(source_docs)}"
                                    for i, doc in enumerate(source_docs, 1):
                                        chunk_preview = doc.page_content[:500].replace('\n', ' ')
                                        full_answer += f"\n|||CHUNK{i}:{chunk_preview}"
                                else:
                                    log_event("WARNING: No source documents retrieved!", "WARNING")
                                
                                st.session_state["messages"].append(("assistant", full_answer))
                                log_event(f"Generated answer using {model_name}")
                                st.rerun()
                                success = True
                                break
                                
                            except Exception as e:
                                error_msg = str(e)
                                all_errors.append(f"{model_name}: {error_msg}")
                                st.warning(f"&#9888; {model_name} failed: {error_msg[:100]}...")
                                log_event(f"Model {model_name} failed: {error_msg}", "WARNING")
                                continue
                        
                        if not success:
                            st.error("&#10060; All models failed. Please check your API keys or try again later.")
                            with st.expander("&#128269; View Error Details"):
                                for error in all_errors:
                                    st.code(error, language="text")
                            st.session_state["messages"].append(("assistant", "&#10060; Sorry, I couldn't generate an answer."))
                            st.rerun()
                
                # Clear chat button

        
        except Exception as e:
            st.error(f"An error occurred: {e}")
            log_event(f"App error: {e}", "ERROR")

    else:
        st.markdown("""
        <div style='text-align: center; padding: 3rem; background: rgba(139, 92, 246, 0.1); border-radius: 20px; border: 2px solid rgba(139, 92, 246, 0.3);'>
            <div style='font-size: 4rem; margin-bottom: 1rem;'>&#128193;</div>
            <h2 style='color: #a78bfa !important;'>Upload Your PDFs to Get Started</h2>
            <p style='font-size: 1.1rem; color: #cbd5e1;'>Use the file uploader above to upload one or more PDF files</p>
            <p style='color: #94a3b8;'>Limit: 200MB per file • Supports multiple PDFs</p>
        </div>
        """, unsafe_allow_html=True)
