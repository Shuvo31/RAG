import os
import warnings
from collections import defaultdict, OrderedDict
from datetime import datetime
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_openai import AzureChatOpenAI

# -----------------------------
# App / Env setup
# -----------------------------
warnings.filterwarnings("ignore")
st.set_page_config(page_title="Chat with Documents", layout="wide")
load_dotenv()

# Use Streamlit secrets (for cloud) with fallback to environment variables (for local)
AZURE_OPENAI_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT")
AZURE_OPENAI_API_KEY =  os.getenv("AZURE_OPENAI_API_KEY")
AZURE_DEPLOYMENT_NAME = os.getenv("AZURE_DEPLOYMENT_NAME")
AZURE_API_VERSION = os.getenv("AZURE_API_VERSION")

# Configuration for retrieval
MAX_SOURCES_TO_SHOW = 3  # Maximum number of source files to display
MAX_CHUNKS_PER_SOURCE = 2  # Maximum chunks to use from each source
SIMILARITY_THRESHOLD = 0.35  # Higher = stricter filtering (0.0 to 1.0)

missing = [k for k, v in {
    "AZURE_OPENAI_ENDPOINT": AZURE_OPENAI_ENDPOINT,
    "AZURE_OPENAI_API_KEY": AZURE_OPENAI_API_KEY,
    "AZURE_DEPLOYMENT_NAME": AZURE_DEPLOYMENT_NAME,
}.items() if not v]

if missing:
    st.error(f"⚠️ Missing configuration: {', '.join(missing)}")
    st.info("Please configure these values in Streamlit Cloud secrets or your .env file")
    st.stop()

FAISS_INDEX_PATH = "faiss_index"

# -----------------------------
# Cached resource loaders
# -----------------------------
@st.cache_resource
def load_embeddings():
    """Load embeddings model (cached to avoid reloading)"""
    return HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={"device": "cpu"},
    )

@st.cache_resource
def load_vectorstore():
    """Load FAISS vectorstore (cached to avoid reloading)"""
    if not os.path.exists(FAISS_INDEX_PATH):
        st.error("❌ FAISS index not found. Please ensure the 'faiss_index' folder is included in your repository.")
        st.info("Run `python create_embeddings.py --dir Entity_Resorts` locally to create the index.")
        st.stop()
    
    try:
        embeddings = load_embeddings()
        vectorstore = FAISS.load_local(
            FAISS_INDEX_PATH, embeddings, allow_dangerous_deserialization=True
        )
        return vectorstore
    except Exception as e:
        st.error(f"Error loading FAISS index: {e}")
        st.stop()

@st.cache_resource
def load_llm():
    """Load Azure OpenAI LLM (cached to avoid reloading)"""
    try:
        return AzureChatOpenAI(
            azure_endpoint=AZURE_OPENAI_ENDPOINT,
            api_key=AZURE_OPENAI_API_KEY,
            deployment_name=AZURE_DEPLOYMENT_NAME,
            api_version=AZURE_API_VERSION,
            temperature=0.2,
        )
    except Exception as e:
        st.error(f"Error initializing Azure OpenAI: {e}")
        st.stop()

# Load resources
vectorstore = load_vectorstore()
llm = load_llm()

# -----------------------------
# Helpers: de-duped citations
# -----------------------------
def compress_ranges(nums):
    """[1,2,3,5,7,8] -> '1–3, 5, 7–8'"""
    if not nums:
        return ""
    nums = sorted(set(nums))
    spans = []
    start = prev = nums[0]
    for n in nums[1:]:
        if n == prev + 1:
            prev = n
        else:
            spans.append(str(start) if start == prev else f"{start}–{prev}")
            start = prev = n
    spans.append(str(start) if start == prev else f"{start}–{prev}")
    return ", ".join(spans)

def limit_sources_intelligently(docs, max_sources=MAX_SOURCES_TO_SHOW, max_chunks_per_source=MAX_CHUNKS_PER_SOURCE):
    """
    Reduce the number of sources by:
    1. Grouping chunks by source file
    2. Taking only the top N sources
    3. Limiting chunks per source
    """
    # Group by source
    source_groups = defaultdict(list)
    for doc in docs:
        src = doc.metadata.get("source") or doc.metadata.get("file_path") or "unknown"
        source_groups[src].append(doc)
    
    # Take only max_sources files (in order of appearance)
    limited_docs = []
    for src in list(source_groups.keys())[:max_sources]:
        # Take max_chunks_per_source from each file
        limited_docs.extend(source_groups[src][:max_chunks_per_source])
    
    return limited_docs

def group_by_source_with_pages(docs):
    """
    Returns:
      ordered_sources: OrderedDict[src -> sorted_pages]
      doc_to_src_index: list[int] (per-doc 1-based source index)
    """
    grouped_pages = defaultdict(set)
    order = []
    for d in docs:
        src = d.metadata.get("source") or d.metadata.get("file_path") or "unknown"
        if src not in grouped_pages:
            order.append(src)
        page = d.metadata.get("page")
        if page is not None:
            grouped_pages[src].add(int(page) + 1)  # 0->1-based
        else:
            grouped_pages[src]  # ensure key exists

    ordered_sources = OrderedDict((src, sorted(grouped_pages[src])) for src in order)
    src_to_idx = {src: i + 1 for i, src in enumerate(ordered_sources.keys())}
    doc_to_src_index = [
        src_to_idx[(d.metadata.get("source") or d.metadata.get("file_path") or "unknown")]
        for d in docs
    ]
    return ordered_sources, doc_to_src_index

def format_sources_markdown(ordered_sources, page_limit=12, show_basename=True):
    """
    Pretty one-line per file: [n] filename.pdf (pp. 2–4, 7, …)
    """
    lines = []
    for i, (src, pages) in enumerate(ordered_sources.items(), start=1):
        name = Path(src).name if show_basename else src
        if pages:
            p = compress_ranges(pages[:page_limit])
            if len(pages) > page_limit:
                p += ", …"
            lines.append(f"[{i}] {name} (pp. {p})")
        else:
            lines.append(f"[{i}] {name}")
    return lines

# -----------------------------
# Multi-chat session management
# -----------------------------
def init_session():
    if "chats" not in st.session_state:
        st.session_state.chats = []  # [{id, title, messages:[{role, content, sources?}]}]
    if "current_chat_id" not in st.session_state:
        st.session_state.current_chat_id = None

def new_chat():
    chat_id = f"chat-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
    st.session_state.chats.append({"id": chat_id, "title": "New chat", "messages": []})
    st.session_state.current_chat_id = chat_id

def get_current_chat():
    cid = st.session_state.current_chat_id
    for c in st.session_state.chats:
        if c["id"] == cid:
            return c
    return None

def set_chat_title_if_empty(chat, first_user_msg):
    if chat and (chat.get("title") in (None, "", "New chat")):
        chat["title"] = (first_user_msg.strip()[:40] + "…") if len(first_user_msg) > 40 else first_user_msg

init_session()
if st.session_state.current_chat_id is None:
    new_chat()

# -----------------------------
# Sidebar (per-chat list, not per message)
# -----------------------------
with st.sidebar:
    st.header("Chat Options")
    if st.button("🆕 New Chat"):
        new_chat()
        st.success("Started a new chat!")

    st.subheader("💭 Chats")
    for chat in st.session_state.chats:
        label = f"• {chat['title']}"
        if st.button(label, key=chat["id"]):
            st.session_state.current_chat_id = chat["id"]
    
    # Add configuration in sidebar
    st.divider()
    st.subheader("⚙️ Settings")
    st.caption(f"Max sources: {MAX_SOURCES_TO_SHOW}")
    st.caption(f"Chunks per source: {MAX_CHUNKS_PER_SOURCE}")
    st.caption(f"Similarity threshold: {SIMILARITY_THRESHOLD}")

# -----------------------------
# Main area
# -----------------------------
current = get_current_chat()
if current is None:
    st.stop()

st.title("💬 Chat with Your Documents – Clean Citations")

# Show history
for msg in current["messages"]:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])
        if msg["role"] == "assistant" and msg.get("sources"):
            st.markdown("**Sources:**")
            for line in msg["sources"]:
                st.markdown(line)

# -----------------------------
# User input
# -----------------------------
user_query = st.chat_input("Ask something about your documents...")

if user_query:
    # Add user message
    current["messages"].append({"role": "user", "content": user_query})
    set_chat_title_if_empty(current, user_query)

    with st.chat_message("user"):
        st.write(user_query)

    # OPTIMIZED RETRIEVAL: Fewer, more relevant sources
    with st.spinner("🔍 Retrieving relevant context..."):
        try:
            # Use MMR for diversity, but with smaller k
            docs = vectorstore.max_marginal_relevance_search(
                user_query, 
                k=6,  # Reduced from 8
                fetch_k=20  # Reduced from 40
            )
            
            # Apply similarity threshold if available
            try:
                scored = vectorstore.similarity_search_with_score(user_query, k=10)
                filtered_docs = [d for d, score in scored if score >= SIMILARITY_THRESHOLD]
                if filtered_docs:
                    # Keep only docs from MMR that are also above threshold
                    filtered_sources = {
                        (d.metadata.get("source") or d.metadata.get("file_path") or "unknown") 
                        for d in filtered_docs
                    }
                    docs = [
                        d for d in docs 
                        if (d.metadata.get("source") or d.metadata.get("file_path") or "unknown") in filtered_sources
                    ]
            except:
                pass  # If scoring not available, continue with MMR results
            
            # CRITICAL: Limit to max sources and chunks
            docs = limit_sources_intelligently(docs, MAX_SOURCES_TO_SHOW, MAX_CHUNKS_PER_SOURCE)
            
        except Exception as e:
            # Fallback
            docs = vectorstore.similarity_search(user_query, k=4)
            docs = limit_sources_intelligently(docs, MAX_SOURCES_TO_SHOW, MAX_CHUNKS_PER_SOURCE)

    if not docs:
        ai_response = "I couldn't find relevant context in the document index."
        sources_md = []
    else:
        # De-dup by file, merge pages, and assign stable [n] indices
        ordered_sources, doc_src_idx = group_by_source_with_pages(docs)
        
        # Build context where each chunk is tagged with its FILE index [n]
        context_lines = [f"[{idx}] {doc.page_content}" for idx, doc in zip(doc_src_idx, docs)]
        context = "\n\n".join(context_lines)

        prompt = f"""You are a helpful assistant that answers strictly based on the provided sources.
When you state a fact, include bracket citations [1], [2], etc., where each number refers to a FILE (not a chunk).
Use the bracket numbers that appear before each chunk below.

IMPORTANT: Only cite sources that are directly relevant to your answer. Don't cite all available sources if they're not needed.

Sources (each chunk labelled with its FILE index):
{context}

Question:
{user_query}
""".strip()

        with st.spinner("🤖 Thinking..."):
            try:
                response = llm.invoke([{"role": "user", "content": prompt}])
                ai_response = response.content
            except Exception as e:
                ai_response = f"Error generating response: {e}"
                st.error(f"API Error: {e}")

        # Pretty one-line-per-file citations
        sources_md = format_sources_markdown(ordered_sources)

    # Save and render assistant message
    current["messages"].append({
        "role": "assistant",
        "content": ai_response,
        "sources": sources_md
    })

    with st.chat_message("assistant"):
        st.write(ai_response)
        if sources_md:
            st.markdown("**Sources:**")
            for line in sources_md:
                st.markdown(line)