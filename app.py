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
import msal

# -----------------------------
# App / Env setup
# -----------------------------
warnings.filterwarnings("ignore")
st.set_page_config(page_title="Club Med RAG Portal", layout="wide")
load_dotenv()

# Configuration
MAX_SOURCES_TO_SHOW = 3
MAX_CHUNKS_PER_SOURCE = 2
SIMILARITY_THRESHOLD = 0.15
CONVERSATION_MEMORY_SIZE = 10  # Number of previous exchanges to remember
DEFAULT_LOCAL_DIR = r"Entity_Resorts"

# Azure AD Configuration
AZURE_AD_CLIENT_ID = os.getenv("AZURE_AD_CLIENT_ID")
AZURE_AD_CLIENT_SECRET = os.getenv("AZURE_AD_CLIENT_SECRET")
AZURE_AD_TENANT_ID = os.getenv("AZURE_AD_TENANT_ID")
AZURE_AD_REDIRECT_URI = os.getenv("AZURE_AD_REDIRECT_URI")

# Azure OpenAI Configuration
AZURE_OPENAI_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT")
AZURE_OPENAI_API_KEY = os.getenv("AZURE_OPENAI_API_KEY")
AZURE_DEPLOYMENT_NAME = os.getenv("AZURE_DEPLOYMENT_NAME")
AZURE_API_VERSION = os.getenv("AZURE_API_VERSION")

# Cost tracking (approximate costs per 1K tokens)
INPUT_COST_PER_1K = 0.00025  # Adjust based on your Azure model
OUTPUT_COST_PER_1K = 0.001  # Adjust based on your Azure model

# Check required configurations
missing_aad = [k for k, v in {
    "AZURE_AD_CLIENT_ID": AZURE_AD_CLIENT_ID,
    "AZURE_AD_CLIENT_SECRET": AZURE_AD_CLIENT_SECRET,
    "AZURE_AD_TENANT_ID": AZURE_AD_TENANT_ID,
}.items() if not v]

missing_openai = [k for k, v in {
    "AZURE_OPENAI_ENDPOINT": AZURE_OPENAI_ENDPOINT,
    "AZURE_OPENAI_API_KEY": AZURE_OPENAI_API_KEY,
    "AZURE_DEPLOYMENT_NAME": AZURE_DEPLOYMENT_NAME,
}.items() if not v]

if missing_aad:
    st.error(f"Missing Azure AD configuration: {', '.join(missing_aad)}")
    st.info("Please configure these values in Streamlit Cloud secrets or your .env file")
    st.stop()

if missing_openai:
    st.error(f"Missing Azure OpenAI configuration: {', '.join(missing_openai)}")
    st.info("Please configure these values in Streamlit Cloud secrets or your .env file")
    st.stop()

FAISS_INDEX_PATH = "faiss_index"

# -----------------------------
# Azure AD Authentication
# -----------------------------
def init_msal_app():
    """Initialize MSAL confidential client application"""
    return msal.ConfidentialClientApplication(
        AZURE_AD_CLIENT_ID,
        authority=f"https://login.microsoftonline.com/{AZURE_AD_TENANT_ID}",
        client_credential=AZURE_AD_CLIENT_SECRET,
    )

def get_auth_url():
    """Generate Azure AD authentication URL for Streamlit Cloud"""
    msal_app = init_msal_app()
    
    # For Streamlit Cloud, we need to handle the redirect properly
    auth_url = msal_app.get_authorization_request_url(
        scopes=["User.Read"],  # Simplified scope
        redirect_uri=AZURE_AD_REDIRECT_URI,
        state="clubmed_rag_auth",  # Add state parameter for security
        prompt="select_account"  # Force account selection
    )
    return auth_url

def handle_authentication_callback():
    """Handle the authentication callback and get tokens"""
    query_params = st.query_params
    code = query_params.get("code")
    error = query_params.get("error")
    
    # Check for errors first
    if error:
        error_desc = query_params.get("error_description", "Unknown error")
        st.error(f"Authentication Error: {error}")
        st.error(f"Details: {error_desc}")
        
        if "AADSTS" in error_desc:
            st.info("This appears to be an Azure AD configuration issue. Please check:")
            st.markdown("""
            1. **Redirect URI**: Ensure it matches exactly in Azure Portal
            2. **Implicit Grant**: Enable ID tokens in Authentication settings
            3. **API Permissions**: Add User.Read permission and grant admin consent
            """)
        return
    
    if code:
        if isinstance(code, list):
            code = code[0]
            
        try:
            with st.spinner("Completing authentication..."):
                msal_app = init_msal_app()
                result = msal_app.acquire_token_by_authorization_code(
                    code,
                    scopes=["User.Read"],
                    redirect_uri=AZURE_AD_REDIRECT_URI,
                )
                
                if "access_token" in result:
                    st.session_state.auth_result = result
                    st.session_state.is_authenticated = True
                    st.session_state.user_email = result.get("id_token_claims", {}).get("preferred_username", "Unknown")
                    
                    # Clear the code from URL
                    st.query_params.clear()
                    st.success("Authentication successful! Redirecting...")
                    st.rerun()
                else:
                    error_msg = result.get("error_description", "Unknown authentication error")
                    error_code = result.get("error", "unknown_error")
                    st.error(f"Authentication failed: {error_code}")
                    st.error(f"Details: {error_msg}")
                    
                    # Provide helpful debugging info
                    with st.expander("Debug Information"):
                        st.json(result)
                    
        except Exception as e:
            st.error(f"Authentication error: {str(e)}")
            with st.expander("Error Details"):
                st.exception(e)

def login_ui():
    """Display login interface with improved styling and debugging"""
    st.title("Club Med RAG Portal")
    st.markdown("---")
    
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.image("https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcT_IRzH6PSS4CLb6lCXJyFxtAm-ZYUwRthEJg&s", width=100)
        st.subheader("Welcome to Club Med Knowledge Portal")
        st.markdown("Please sign in with your organizational account to access the knowledge base.")
        
        try:
            auth_url = get_auth_url()
            
            # Use Streamlit's native link_button for better compatibility
            st.link_button(
                "Sign in with Microsoft",
                auth_url,
                use_container_width=True,
                type="primary"
            )
            
            st.markdown("---")

        except Exception as e:
            st.error(f"Error generating login URL: {str(e)}")
            with st.expander("Error Details"):
                st.exception(e)
    
        st.caption("This portal provides AI-powered access to Club Med documentation and resources.")


def logout():
    """Clear authentication state"""
    st.session_state.clear()
    st.rerun()

def check_authentication():
    """Check if user is authenticated"""
    # Handle authentication callback if present
    if "code" in st.query_params or "error" in st.query_params:
        handle_authentication_callback()
        return False
    
    # Check if already authenticated
    if st.session_state.get("is_authenticated"):
        return True
    
    # Show login UI
    login_ui()
    return False

# Query classification for handling general chat vs knowledge queries
class QueryClassifier:
    """Distinguishes between general chat and knowledge base queries"""
    
    def __init__(self):
        self.general_chat_patterns = [
            r'^(hi|hello|hey|bonjour|salut|coucou)[\s!.?]*$',
            r'how\s+are\s+you', r'comment\s+(vas|allez)', r'ça\s+va',
            r'(i\s+am|i\'m|je\s+suis)\s+(feeling|lonely|sad|happy|tired)',
            r'(i\s+feel|je\s+me\s+sens)', r'talk\s+to\s+(me|you)',
            r'^(thanks|thank\s+you|merci)[\s!.?]*$',
            r'^(bye|goodbye|au\s+revoir)[\s!.?]*$',
            r'(what|who)\s+are\s+you',
        ]
        self.knowledge_indicators = [
            r'(comment|pourquoi|quand|où|qui|quel|combien)',
            r'(club\s+med|village|resort)',
            r'(inventaire|inventory|facture|invoice|commande|order)',
            r'(procédure|procedure|process)',
            r'(coupa|ecomat|s@fe|e[\s-]?pack)',
        ]
        self.chat_regex = [re.compile(p, re.IGNORECASE) for p in self.general_chat_patterns]
        self.knowledge_regex = [re.compile(p, re.IGNORECASE) for p in self.knowledge_indicators]
    
    def is_general_chat(self, query):
        query_lower = query.lower().strip()
        chat_matches = sum(1 for p in self.chat_regex if p.search(query_lower))
        knowledge_matches = sum(1 for p in self.knowledge_regex if p.search(query_lower))
        if chat_matches > 0 and knowledge_matches == 0:
            return True
        if knowledge_matches > 0:
            return False
        if len(query_lower.split()) <= 3:
            return True
        return False

def get_general_chat_response(query):
    """Generate appropriate response for general chat"""
    q = query.lower()
    if any(w in q for w in ['hi', 'hello', 'hey', 'bonjour', 'salut']):
        return "Hello! I'm the Club Med Knowledge Assistant. I can help you find information from Club Med documents and procedures. What would you like to know?"
    if 'how are you' in q or 'comment vas' in q or 'ça va' in q:
        return "I'm doing well, thank you! I'm here to help you with Club Med documentation. What would you like to know?"
    if any(w in q for w in ['lonely', 'sad', 'feeling']):
        return "I'm here to assist with Club Med documentation and procedures. Is there something about Club Med procedures I can help you with?"
    if 'thank' in q or 'merci' in q:
        return "You're welcome! Let me know if you need anything else."
    if 'bye' in q or 'goodbye' in q or 'au revoir' in q:
        return "Goodbye! Feel free to return anytime."
    if 'who are you' in q or 'what are you' in q:
        return "I'm the Club Med Knowledge Assistant. I help you find information from Club Med documents and procedures. How can I assist you today?"
    return "I'm the Club Med Knowledge Assistant. What would you like to know about Club Med procedures?"


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
        st.error(f"FAISS index not found at {FAISS_INDEX_PATH}.")
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

if not check_authentication():
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


# Initialize query classifier
if "query_classifier" not in st.session_state:
    st.session_state.query_classifier = QueryClassifier()

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

def is_no_answer_response(response_text):
    """
    Check if response indicates no information was found.
    Returns True if the AI says it doesn't have information.
    """
    no_answer_phrases = [
        # English phrases
        "don't have information",
        "don't find information",
        "no information about",
        "cannot find information",
        "not found in the documents",
        "outside the scope",
        "couldn't find relevant",
        
        # French phrases  
        "ne trouve pas d'information",
        "pas d'information sur",
        "ne dispose pas d'information",
        "n'ai pas d'information"
    ]
    
    response_lower = response_text.lower()
    return any(phrase in response_lower for phrase in no_answer_phrases)


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
            return original_response + "\n\n*For detailed image analysis, you can run the image processor script to extract text from images.*"
        
        # Group image texts by source
        from collections import defaultdict
        image_groups = defaultdict(list)
        
        for doc in image_docs:
            source = doc.metadata.get('file_name', 'Unknown')
            image_groups[source].append(doc)
        
        enhanced_response = original_response + "\n\n**Extracted Image Content:**\n"
        
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
    if "is_authenticated" not in st.session_state:
        st.session_state.is_authenticated = False
    if "user_email" not in st.session_state:
        st.session_state.user_email = None

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

def enhanced_document_retrieval(query, vectorstore, k=20, similarity_threshold=0.15):
    """Enhanced retrieval with multiple search strategies"""
    
    all_docs = []
    
    # Strategy 1: Standard similarity search
    try:
        docs1 = vectorstore.similarity_search(query, k=k)
        all_docs.extend(docs1)
        print(f"Standard search found {len(docs1)} docs")  # Debug
    except Exception as e:
        print(f"Standard search error: {e}")
    
    # Strategy 2: Search with score filtering
    try:
        scored_docs = vectorstore.similarity_search_with_score(query, k=k*2)
        docs2 = [doc for doc, score in scored_docs if score >= similarity_threshold]
        all_docs.extend(docs2)
        print(f"Scored search found {len(docs2)} docs")  # Debug
    except Exception as e:
        print(f"Scored search error: {e}")
    
    # Strategy 3: Try alternative query formulations
    alternative_queries = []
    
    # For French queries about scales/balances
    if any(word in query.lower() for word in ['balance', 'scale', 'pèse', 'peser', 'étalonnée']):
        alternative_queries.extend([
            "balance réception marchandises",
            "matériel nécessaire réception",
            "équipement réception",
            "balance étalonnée réception"
        ])
    
    # Add general alternatives
    alternative_queries.extend([
        query + " procédure réception",
        "réception des marchandises équipement",
        "matériel requis réception"
    ])
    
    for alt_query in alternative_queries:
        try:
            additional_docs = vectorstore.similarity_search(alt_query, k=10)
            all_docs.extend(additional_docs)
            print(f"Alt query '{alt_query}' found {len(additional_docs)} docs")  # Debug
        except Exception as e:
            print(f"Alt query error: {e}")
            continue
    
    # Remove duplicates
    seen = set()
    unique_docs = []
    for doc in all_docs:
        doc_id = (
            doc.metadata.get("source", ""),
            doc.metadata.get("page", ""),
            hash(doc.page_content[:200])  # More robust duplicate detection
        )
        if doc_id not in seen:
            seen.add(doc_id)
            unique_docs.append(doc)
    
    print(f"Total unique documents found: {len(unique_docs)}")  # Debug
    return unique_docs[:k]


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

def enhanced_retrieval_for_technical_terms(query, vectorstore, k=30, fetch_k=100):
    """
    Retrieval optimized for technical terms and French queries
    
    Key improvements:
    1. Uses MMR with high fetch_k to get more candidates
    2. Re-ranks based on exact term matching
    3. Boosts documents with technical terms
    4. No similarity threshold filtering
    """
    from collections import defaultdict
    import re
    
    # Extract potential technical terms from query
    # Technical patterns: 
    # - Words with special chars (e-pack, S@fe)
    # - English words in French context (pending, free text)
    # - Acronyms (PMH, SLV, F&B)
    
    technical_patterns = [
        r'\b[A-Z@]{2,}\b',  # Acronyms: S@fe, PMH
        r'\be-\w+\b',  # e-pack, e-mail
        r'\bpending\s+\w+\b',  # pending approval, pending receipt
        r'\bfree\s+text\b',  # free text
        r'\b[A-Z]{2,}\b',  # Uppercase: VLS, RFI
    ]
    
    technical_terms = set()
    for pattern in technical_patterns:
        matches = re.findall(pattern, query, re.IGNORECASE)
        technical_terms.update([m.lower() for m in matches])
    
    # Also extract key French terms (2+ chars, not common words)
    common_words = {'qui', 'que', 'quoi', 'est', 'dans', 'une', 'des', 'les', 'pour', 
                    'sur', 'avec', 'par', 'peux', 'expliquer', 'donner', 'peut'}
    query_words = [w.lower() for w in query.split() if len(w) > 2 and w.lower() not in common_words]
    
    print(f"Technical terms detected: {technical_terms}")
    print(f"Key query words: {query_words[:5]}")
    
    # Strategy 1: Get diverse candidates with MMR
    try:
        mmr_docs = vectorstore.max_marginal_relevance_search(
            query,
            k=k,
            fetch_k=fetch_k  # Fetch many candidates
        )
    except:
        # Fallback to similarity search
        mmr_docs = vectorstore.similarity_search(query, k=k)
    
    # Strategy 2: Also get pure similarity matches
    sim_docs = vectorstore.similarity_search_with_score(query, k=k)
    
    # Combine and deduplicate
    seen_content = set()
    combined = []
    
    # Add MMR docs first (they're already ordered by relevance + diversity)
    for doc in mmr_docs:
        content_hash = hash(doc.page_content[:200])
        if content_hash not in seen_content:
            seen_content.add(content_hash)
            combined.append((doc, 0.1))  # Low score = high relevance
    
    # Add similarity docs
    for doc, score in sim_docs:
        content_hash = hash(doc.page_content[:200])
        if content_hash not in seen_content:
            seen_content.add(content_hash)
            combined.append((doc, score))
    
    # Re-rank based on technical term matches and query word density
    scored_docs = []
    
    for doc, original_score in combined:
        content_lower = doc.page_content.lower()
        
        # Start with original score
        final_score = original_score
        
        # Heavy boost for exact technical term matches
        tech_matches = sum(1 for term in technical_terms if term in content_lower)
        if tech_matches > 0:
            # Each technical term match significantly boosts relevance
            final_score *= (0.3 ** tech_matches)  # Very strong boost
        
        # Boost for query word density
        word_matches = sum(1 for word in query_words if word in content_lower)
        if word_matches > 0:
            match_ratio = word_matches / len(query_words) if query_words else 0
            final_score *= (0.7 ** match_ratio)
        
        # Boost for exact phrase matches (if query has quoted terms or key phrases)
        if 'pending approval' in query.lower() and 'pending approval' in content_lower:
            final_score *= 0.2
        if 'pending receipt' in query.lower() and 'pending receipt' in content_lower:
            final_score *= 0.2
        if 'free text' in query.lower() and 'free text' in content_lower:
            final_score *= 0.2
        if 'e-pack' in query.lower() and 'e-pack' in content_lower:
            final_score *= 0.2
        
        scored_docs.append((doc, final_score, tech_matches, word_matches))
    
    # Sort by final score (lower is better)
    scored_docs.sort(key=lambda x: x[1])
    
    # Debug: Show top results
    print(f"\nTop 5 retrieval results:")
    for i, (doc, score, tech, words) in enumerate(scored_docs[:5], 1):
        fname = doc.metadata.get('file_name', 'Unknown')
        print(f"  {i}. Score: {score:.4f} | Tech: {tech} | Words: {words} | {fname}")
    
    # Return just the documents
    return [doc for doc, _, _, _ in scored_docs[:k]]

# -----------------------------
# Initialize Resources and Session
# -----------------------------
init_session()

# Check authentication before proceeding
if not check_authentication():
    st.stop()

# Load resources only if authenticated
try:
    vectorstore = load_vectorstore()
    llm = load_llm()
except Exception as e:
    st.error(f"Failed to initialize application: {e}")
    st.stop()

if st.session_state.current_chat_id is None:
    new_chat()

# -----------------------------
# Enhanced Sidebar with User Info
# -----------------------------
with st.sidebar:
    # Professional user profile with custom styling
    st.markdown(f"""
        <div style='padding: 0.5rem 0; border-bottom: 1px solid rgba(49, 51, 63, 0.2);'>
            <h3 style='margin: 0; font-size: 1rem; font-weight: 600;'>{st.session_state.user_email.split('@')[0].title()}</h3>
            <p style='margin: 0.25rem 0 0 0; font-size: 0.8rem; color: rgba(49, 51, 63, 0.6);'>{st.session_state.user_email}</p>
        </div>
    """, unsafe_allow_html=True)
    
    if st.button("Sign Out", use_container_width=True, type="secondary"):
        logout()
    
    st.divider()
    
    # Single, prominent button
    if st.button("+ New Conversation", use_container_width=True, type="primary"):
        new_chat()
        st.rerun()
    
    # Filter and display only active conversations
    if st.session_state.chats:
        active_chats = [chat for chat in st.session_state.chats if len(chat.get("messages", [])) > 0]
        
        if active_chats:
            st.markdown("<div style='margin-top: 1.5rem; margin-bottom: 0.5rem;'><small style='color: rgba(49, 51, 63, 0.6); text-transform: uppercase; font-weight: 600; letter-spacing: 0.05em; font-size: 0.7rem;'>Recent Conversations</small></div>", unsafe_allow_html=True)
            
            for chat in active_chats:
                is_active = chat["id"] == st.session_state.current_chat_id
                
                title = chat.get("title", "Untitled Conversation")
                if title == "New chat" and chat.get("messages"):
                    first_msg = chat["messages"][0]["content"]
                    title = (first_msg[:50] + "...") if len(first_msg) > 50 else first_msg
                
                button_type = "primary" if is_active else "secondary"
                display_title = f"{'▸ ' if is_active else '  '}{title}"
                
                if st.button(display_title, key=chat["id"], use_container_width=True, type=button_type):
                    st.session_state.current_chat_id = chat["id"]
                    st.rerun()
    st.divider()
    st.subheader("Usage & Costs")
    
    cost_summary = st.session_state.cost_tracker.get_session_summary()
    st.metric("Total Input Tokens", f"{cost_summary['total_input_tokens']:,}")
    st.metric("Total Output Tokens", f"{cost_summary['total_output_tokens']:,}")
    st.metric("Estimated Cost", f"${cost_summary['estimated_cost']:.4f}")
    
    

# Main Chat Interface
# -----------------------------
current = get_current_chat()
if current is None:
    st.stop()

st.title("Club Med Knowledge Portal")
st.caption(f"Welcome, {st.session_state.user_email} | Secure RAG Chat with Azure AD")

# Display chat history with enhanced formatting
for msg in current["messages"]:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])
        
        if msg["role"] == "assistant":
            # Show sources if available
            if not msg.get("is_general_chat", False) and msg.get("sources") and not is_no_answer_response(msg["content"]):
                with st.expander("View Sources", expanded=False):
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
user_query = st.chat_input("Ask something about Club Med documents...")

if user_query:
    # Add user message to chat
    current["messages"].append({"role": "user", "content": user_query})
    set_chat_title_if_empty(current, user_query)
    
    with st.chat_message("user"):
        st.write(user_query)
    
    is_chat = st.session_state.query_classifier.is_general_chat(user_query)

    if is_chat:
        # Handle general chat queries
        ai_response = get_general_chat_response(user_query)

        # Save assistant response (no sources, no tokens)
        current["messages"].append({
            "role": "assistant",
            "content": ai_response,
            "sources": [],
            "token_usage": {"input_tokens": 0, "output_tokens": 0},
            "cost": 0,
            "timestamp": datetime.now().isoformat(),
            "is_general_chat": True
        })
        
        st.rerun()
        
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
    with st.spinner("Searching knowledge base..."):
        try:
            # Use MMR for diverse, relevant results
            docs = enhanced_retrieval_for_technical_terms(
                enhanced_query, 
                vectorstore,
                k=30,
                fetch_k=100
            )
            
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
        
        prompt = f"""You are a Clubmed knowledge assistant that provides accurate information from clubmed document sources and conversation context.

SOURCES (each chunk labeled with its FILE index):
{context}
{memory_context}

USER'S QUESTION: {user_query}
{response_guidance}

INSTRUCTIONS:
1. **Carefully read all the provided sources above before answering**
2. **ALWAYS respond in the SAME LANGUAGE as the user's question**
3. **If the sources contain relevant information to answer the question:**
   - Provide a comprehensive answer based ONLY on the provided sources
   - Cite relevant sources using bracket notation [1], [2], etc.
   - For technical terms, be very precise and use exact terminology from the sources
   - Do not add information from your general knowledge
4. **If the sources do NOT contain relevant information to answer the question:**
   - Clearly state: "Je ne trouve pas d'informations sur [topic] dans les sources fournies." if reply for french, or "I don't find information about [topic] in the provided sources." if reply for english.
   - Do not fabricate answers or make assumptions
   - Do not provide answers from your general knowledge
5. **For technical queries (systems, processes, specific terms):**
   - Look for EXACT matches of the technical terms
   - Explain the specific meaning/purpose
   - Distinguish between similar terms if applicable
6. If this is a follow-up question, maintain continuity with the previous conversation
7. {response_guidance}

IMPORTANT: 
- Answer ONLY if the sources are ACTUALLY relevant to the question
- Be precise with technical terminology
- Don't refuse to answer if you find relevant information in the sources
- Don't use general knowledge—stick to the sources ONLY
- RESPOND IN THE SAME LANGUAGE AS THE USER'S QUESTION

RESPONSE:"""
        
        with st.spinner("Generating response..."):
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
        if is_no_answer_response(ai_response):
            sources_md = []  # Hide sources if no answer found

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