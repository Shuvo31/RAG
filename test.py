import os
import warnings
from collections import defaultdict, OrderedDict
from datetime import datetime
from pathlib import Path
import re
import json

import streamlit as st
from dotenv import load_dotenv
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_openai import AzureChatOpenAI

# -----------------------------
# App / Env setup
# -----------------------------
warnings.filterwarnings("ignore")
st.set_page_config(page_title="Advanced RAG Chat", layout="wide")
load_dotenv()

# Configuration
MAX_SOURCES_TO_SHOW = 3
MAX_CHUNKS_PER_SOURCE = 2
SIMILARITY_THRESHOLD = 0.35
CONVERSATION_MEMORY_SIZE = 10  # Number of previous exchanges to remember
DEFAULT_LOCAL_DIR = r"Entity_Resorts"

# Azure OpenAI Configuration
AZURE_OPENAI_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT")
AZURE_OPENAI_API_KEY = os.getenv("AZURE_OPENAI_API_KEY")
AZURE_DEPLOYMENT_NAME = os.getenv("AZURE_DEPLOYMENT_NAME")
AZURE_API_VERSION = os.getenv("AZURE_API_VERSION")

# Cost tracking (approximate costs per 1K tokens)
INPUT_COST_PER_1K = 0.00025  # Adjust based on your Azure model
OUTPUT_COST_PER_1K = 0.001  # Adjust based on your Azure model

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
# Enhanced Cached Resource Loaders
# -----------------------------
@st.cache_resource
def load_embeddings():
    return HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={"device": "cpu"},
    )

@st.cache_resource
def load_vectorstore():
    if not os.path.exists(FAISS_INDEX_PATH):
        st.error(f"❌ FAISS index not found at {FAISS_INDEX_PATH}.")
        st.info(f"Please ensure the index is created from the {DEFAULT_LOCAL_DIR} directory.")
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
    try:
        return AzureChatOpenAI(
            azure_endpoint=AZURE_OPENAI_ENDPOINT,
            api_key=AZURE_OPENAI_API_KEY,
            deployment_name=AZURE_DEPLOYMENT_NAME,
            api_version=AZURE_API_VERSION or "2023-12-01-preview",
            temperature=0.1,  # Lower temperature for more consistent responses
            max_tokens=1500,  # Limit response length
        )
    except Exception as e:
        st.error(f"Error initializing Azure OpenAI: {e}")
        st.stop()

# -----------------------------
# Query Understanding & Enhancement
# -----------------------------
class QueryEnhancer:
    """Enhances vague queries and understands user intent"""
    
    @staticmethod
    def enhance_query(original_query, conversation_history=None):
        """Improve vague queries using conversation context"""
        if conversation_history is None:
            conversation_history = []
        
        # If query seems vague, add context from history
        if len(original_query.split()) < 5 and conversation_history:
            # Look for the most recent detailed question/answer
            recent_context = ""
            for msg in reversed(conversation_history[-4:]):  # Last 4 messages
                if msg["role"] == "user" and len(msg["content"].split()) > 8:
                    recent_context = f" (related to: {msg['content'][:100]}...)"
                    break
                elif msg["role"] == "assistant" and len(msg["content"].split()) > 15:
                    recent_context = f" (context: {msg['content'][:150]}...)"
                    break
            
            if recent_context:
                enhanced_query = f"{original_query}{recent_context}"
                return enhanced_query
        
        return original_query
    
    @staticmethod
    def detect_query_intent(query):
        """Detect user intent for better response formatting"""
        query_lower = query.lower()
        
        intents = {
            "table": any(word in query_lower for word in ["table", "list", "summary", "compare", "chart"]),
            "detail": any(word in query_lower for word in ["detail", "explain", "elaborate", "more about", "expand"]),
            "follow_up": any(word in query_lower for word in ["previous", "before", "earlier", "last answer", "what you said"]),
            "simple": len(query.split()) < 6
        }
        
        return intents

# -----------------------------
# Image Query Handler
# -----------------------------
class ImageQueryHandler:
    """Handle image-related queries intelligently"""
    
    @staticmethod
    def is_image_query(query):
        """Detect if query is about images/visual content"""
        query_lower = query.lower()
        image_keywords = [
            'image', 'picture', 'photo', 'diagram', 'chart', 'graph', 
            'flowchart', 'figure', 'visual', 'screenshot', 'illustration',
            'what does this image show', 'describe the image', 'what is in the picture',
            'chart shows', 'graph indicates', 'diagram illustrates'
        ]
        return any(keyword in query_lower for keyword in image_keywords)
    
    @staticmethod
    def enhance_image_response(docs, original_response):
        """Enhance response when image content is available"""
        image_docs = [doc for doc in docs if doc.metadata.get('content_type') == 'image_ocr']
        
        if not image_docs:
            return original_response + "\n\n*💡 For detailed image analysis, you can run the image processor script to extract text from images.*"
        
        # Group image texts by source
        from collections import defaultdict
        image_groups = defaultdict(list)
        
        for doc in image_docs:
            source = doc.metadata.get('file_name', 'Unknown')
            image_groups[source].append(doc)
        
        enhanced_response = original_response + "\n\n**📷 Extracted Image Content:**\n"
        
        for source, docs in list(image_groups.items())[:2]:  # Show max 2 sources
            enhanced_response += f"\n**From {source}:**\n"
            for i, doc in enumerate(docs[:2], 1):  # Show max 2 images per source
                page = doc.metadata.get('original_page', '?')
                text_preview = doc.page_content[:150] + "..." if len(doc.page_content) > 150 else doc.page_content
                enhanced_response += f"  • Image {i} (Page {page}): {text_preview}\n"
        
        if len(image_groups) > 2:
            enhanced_response += f"\n*... and {len(image_groups) - 2} more sources with image content*"
        
        return enhanced_response

# -----------------------------
# Cost Tracking
# -----------------------------
class CostTracker:
    """Track API usage costs"""
    
    def __init__(self):
        self.total_input_tokens = 0
        self.total_output_tokens = 0
        self.session_costs = 0.0
    
    def add_usage(self, input_tokens, output_tokens):
        self.total_input_tokens += input_tokens
        self.total_output_tokens += output_tokens
        
        input_cost = (input_tokens / 1000) * INPUT_COST_PER_1K
        output_cost = (output_tokens / 1000) * OUTPUT_COST_PER_1K
        cost = input_cost + output_cost
        self.session_costs += cost
        
        return cost
    
    def get_session_summary(self):
        return {
            "total_input_tokens": self.total_input_tokens,
            "total_output_tokens": self.total_output_tokens,
            "estimated_cost": self.session_costs
        }

# -----------------------------
# Enhanced Session Management with Memory
# -----------------------------
def init_session():
    if "chats" not in st.session_state:
        st.session_state.chats = []
    if "current_chat_id" not in st.session_state:
        st.session_state.current_chat_id = None
    if "cost_tracker" not in st.session_state:
        st.session_state.cost_tracker = CostTracker()
    if "query_enhancer" not in st.session_state:
        st.session_state.query_enhancer = QueryEnhancer()
    if "image_handler" not in st.session_state:
        st.session_state.image_handler = ImageQueryHandler()

def get_conversation_memory(chat, window_size=CONVERSATION_MEMORY_SIZE):
    """Extract recent conversation context for memory"""
    if not chat or "messages" not in chat:
        return []
    
    # Get last N exchanges (pairs of user-assistant messages)
    messages = chat["messages"]
    recent_exchanges = []
    
    # Start from the end and work backwards
    i = len(messages) - 1
    while i >= 0 and len(recent_exchanges) < window_size * 2:
        if messages[i]["role"] == "user":
            # Try to find the corresponding assistant response
            if i + 1 < len(messages) and messages[i + 1]["role"] == "assistant":
                recent_exchanges.insert(0, messages[i + 1])  # Assistant response
                recent_exchanges.insert(0, messages[i])      # User question
                i -= 2
            else:
                i -= 1
        else:
            i -= 1
    
    return recent_exchanges

def new_chat():
    chat_id = f"chat-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
    st.session_state.chats.append({
        "id": chat_id, 
        "title": "New chat", 
        "messages": [],
        "created_at": datetime.now().isoformat()
    })
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

# -----------------------------
# Helper Functions
# -----------------------------
def compress_ranges(nums):
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
    source_groups = defaultdict(list)
    for doc in docs:
        src = doc.metadata.get("source") or doc.metadata.get("file_path") or "unknown"
        source_groups[src].append(doc)
    
    limited_docs = []
    for src in list(source_groups.keys())[:max_sources]:
        limited_docs.extend(source_groups[src][:max_chunks_per_source])
    
    return limited_docs

def group_by_source_with_pages(docs):
    grouped_pages = defaultdict(set)
    order = []
    for d in docs:
        src = d.metadata.get("source") or d.metadata.get("file_path") or "unknown"
        if src not in grouped_pages:
            order.append(src)
        page = d.metadata.get("page")
        if page is not None:
            grouped_pages[src].add(int(page) + 1)
        else:
            grouped_pages[src]
    
    ordered_sources = OrderedDict((src, sorted(grouped_pages[src])) for src in order)
    src_to_idx = {src: i + 1 for i, src in enumerate(ordered_sources.keys())}
    doc_to_src_index = [
        src_to_idx[(d.metadata.get("source") or d.metadata.get("file_path") or "unknown")]
        for d in docs
    ]
    return ordered_sources, doc_to_src_index

def format_sources_markdown(ordered_sources, page_limit=12, show_basename=True):
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
# Initialize Resources and Session
# -----------------------------

# Load resources at module level
try:
    vectorstore = load_vectorstore()
    llm = load_llm()
except Exception as e:
    st.error(f"Failed to initialize application: {e}")
    st.stop()

init_session()
if st.session_state.current_chat_id is None:
    new_chat()

# -----------------------------
# Enhanced Sidebar
# -----------------------------
with st.sidebar:
    st.header("💭 Chat Sessions")
    
    if st.button("🆕 New Chat", use_container_width=True):
        new_chat()
        st.rerun()
    
    # Chat history with better display
    for chat in st.session_state.chats:
        is_active = chat["id"] == st.session_state.current_chat_id
        emoji = "🔵" if is_active else "⚪"
        label = f"{emoji} {chat['title']}"
        
        if st.button(label, key=chat["id"], use_container_width=True):
            st.session_state.current_chat_id = chat["id"]
            st.rerun()
    
    st.divider()
    st.subheader("📊 Usage & Costs")
    
    cost_summary = st.session_state.cost_tracker.get_session_summary()
    st.metric("Total Input Tokens", f"{cost_summary['total_input_tokens']:,}")
    st.metric("Total Output Tokens", f"{cost_summary['total_output_tokens']:,}")
    st.metric("Estimated Cost", f"${cost_summary['estimated_cost']:.4f}")
    
    st.divider()
    st.subheader("⚙️ RAG Settings")
    st.caption(f"Max sources: {MAX_SOURCES_TO_SHOW}")
    st.caption(f"Chunks per source: {MAX_CHUNKS_PER_SOURCE}")
    st.caption(f"Similarity threshold: {SIMILARITY_THRESHOLD}")
    st.caption(f"Memory size: {CONVERSATION_MEMORY_SIZE} exchanges")
    
    st.divider()
    st.subheader("🖼️ Image Support")
    st.caption("Image text extraction available via separate processor")
    if st.button("🔄 Check for Image Content", use_container_width=True):
        # Check if image content exists in the vectorstore
        image_docs = []
        try:
            # Sample query to find image content
            sample_docs = vectorstore.similarity_search("image chart diagram", k=10)
            image_docs = [doc for doc in sample_docs if doc.metadata.get('content_type') == 'image_ocr']
        except:
            pass
        
        if image_docs:
            st.success(f"✅ Found {len(image_docs)} image text chunks")
        else:
            st.info("ℹ️ No image content found. Run image_processor.py to extract text from images.")

# -----------------------------
# Main Chat Interface
# -----------------------------
current = get_current_chat()
if current is None:
    st.stop()

st.title("🧠 Advanced RAG Chat with Memory")

# Display chat history with enhanced formatting
for msg in current["messages"]:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])
        
        if msg["role"] == "assistant":
            # Show sources if available
            if msg.get("sources"):
                with st.expander("📚 View Sources", expanded=False):
                    for line in msg["sources"]:
                        st.markdown(line)
            
            # Show token usage and cost if available
            if msg.get("token_usage"):
                usage = msg["token_usage"]
                cost = msg.get("cost", 0)
                st.caption(f"Tokens: {usage['input_tokens']} in / {usage['output_tokens']} out • Cost: ${cost:.4f}")

# -----------------------------
# Enhanced User Input Processing
# -----------------------------
user_query = st.chat_input("Ask something about your documents...")

if user_query:
    # Add user message to chat
    current["messages"].append({"role": "user", "content": user_query})
    set_chat_title_if_empty(current, user_query)
    
    with st.chat_message("user"):
        st.write(user_query)
    
    # Get conversation context for memory
    conversation_context = get_conversation_memory(current)
    
    # Enhance query using conversation context and query understanding
    enhanced_query = st.session_state.query_enhancer.enhance_query(
        user_query, 
        conversation_context
    )
    
    intent = st.session_state.query_enhancer.detect_query_intent(user_query)
    
    # Check if this is an image-related query
    is_image_query = st.session_state.image_handler.is_image_query(user_query)
    
    # Retrieve relevant documents
    with st.spinner("🔍 Searching knowledge base..."):
        try:
            # Use MMR for diverse, relevant results
            docs = vectorstore.max_marginal_relevance_search(
                enhanced_query, 
                k=8,  # Slightly more for better coverage
                fetch_k=20
            )
            
            # Apply similarity threshold
            try:
                scored = vectorstore.similarity_search_with_score(enhanced_query, k=12)
                filtered_docs = [d for d, score in scored if score >= SIMILARITY_THRESHOLD]
                if filtered_docs:
                    filtered_sources = {
                        (d.metadata.get("source") or d.metadata.get("file_path") or "unknown") 
                        for d in filtered_docs
                    }
                    docs = [
                        d for d in docs 
                        if (d.metadata.get("source") or d.metadata.get("file_path") or "unknown") in filtered_sources
                    ]
            except Exception as e:
                st.warning(f"Similarity scoring not available: {e}")
                # Continue with MMR results
            
            # Limit sources intelligently
            docs = limit_sources_intelligently(docs, MAX_SOURCES_TO_SHOW, MAX_CHUNKS_PER_SOURCE)
            
        except Exception as e:
            st.error(f"Retrieval error: {e}")
            docs = []

    # Generate response with context and memory
    if not docs:
        ai_response = "I couldn't find relevant information in the knowledge base to answer your question."
        
        # Special handling for image queries
        if is_image_query:
            ai_response += "\n\nFor detailed analysis of images, charts, or diagrams, please run the image processor script to extract text from images in the documents."
        
        sources_md = []
        token_usage = {"input_tokens": 0, "output_tokens": 0}
        cost = 0
    else:
        # Prepare context with source indexing
        ordered_sources, doc_src_idx = group_by_source_with_pages(docs)
        context_lines = [f"[{idx}] {doc.page_content}" for idx, doc in zip(doc_src_idx, docs)]
        context = "\n\n".join(context_lines)
        
        # Build enhanced prompt with conversation memory
        memory_context = ""
        if conversation_context:
            memory_context = "\n\nPrevious conversation context (for reference):\n"
            for msg in conversation_context[-6:]:  # Last 3 exchanges
                role = "User" if msg["role"] == "user" else "Assistant"
                memory_context += f"{role}: {msg['content']}\n"
        
        # Intent-based response guidance
        response_guidance = ""
        if intent["table"]:
            response_guidance = "Please provide the information in a well-structured table format if appropriate."
        elif intent["detail"]:
            response_guidance = "Please provide a detailed, comprehensive explanation."
        elif intent["follow_up"]:
            response_guidance = "This appears to be a follow-up question. Please connect your response to the previous discussion."
        
        # Special guidance for image queries
        if is_image_query:
            response_guidance += " The user is asking about visual content. If you reference image-extracted text, mention that it comes from image content analysis."
        
        prompt = f"""You are a helpful assistant that answers based on the provided sources and conversation context.

SOURCES (each chunk labeled with its FILE index):
{context}
{memory_context}

USER'S QUESTION: {user_query}
{response_guidance}

INSTRUCTIONS:
1. Answer strictly based on the provided sources when possible
2. Cite relevant sources using bracket notation [1], [2], etc.
3. If this is a follow-up question, maintain continuity with the previous conversation
4. {response_guidance}
5. If the sources don't contain relevant information, say so but you can still provide a helpful response based on general knowledge

RESPONSE:"""
        
        with st.spinner("🤔 Generating response..."):
            try:
                response = llm.invoke([{"role": "user", "content": prompt}])
                ai_response = response.content
                
                # Enhance response for image queries
                if is_image_query:
                    ai_response = st.session_state.image_handler.enhance_image_response(docs, ai_response)
                
                # Extract token usage if available
                if hasattr(response, 'usage'):
                    token_usage = {
                        "input_tokens": response.usage.get('prompt_tokens', 0),
                        "output_tokens": response.usage.get('completion_tokens', 0)
                    }
                    cost = st.session_state.cost_tracker.add_usage(
                        token_usage["input_tokens"], 
                        token_usage["output_tokens"]
                    )
                else:
                    # Estimate token usage (rough approximation)
                    input_tokens_est = len(prompt.split()) * 1.3
                    output_tokens_est = len(ai_response.split()) * 1.3
                    token_usage = {
                        "input_tokens": int(input_tokens_est),
                        "output_tokens": int(output_tokens_est)
                    }
                    cost = st.session_state.cost_tracker.add_usage(
                        token_usage["input_tokens"], 
                        token_usage["output_tokens"]
                    )
                    
            except Exception as e:
                ai_response = f"Error generating response: {e}"
                st.error(f"API Error: {e}")
                token_usage = {"input_tokens": 0, "output_tokens": 0}
                cost = 0
        
        sources_md = format_sources_markdown(ordered_sources)

    # Save and display assistant response
    current["messages"].append({
        "role": "assistant",
        "content": ai_response,
        "sources": sources_md,
        "token_usage": token_usage,
        "cost": cost,
        "timestamp": datetime.now().isoformat(),
        "is_image_query": is_image_query
    })
    
    # Force a rerun to update the UI with the new message
    st.rerun()