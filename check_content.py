"""
Quick diagnostic to check if specific content exists in vectorstore
"""
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings

# Initialize embeddings
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2",
    model_kwargs={"device": "cpu"},
)

# Load vectorstore
print("Loading vectorstore...")
try:
    vectorstore = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
    print(f"✓ Vectorstore loaded")
    print(f"Total vectors: {vectorstore.index.ntotal}")
except Exception as e:
    print(f"❌ Error loading vectorstore: {e}")
    exit(1)

# Test queries that are failing
failing_queries = {
    "Q2": {
        "query": "pending approval pending receipt",
        "key_terms": ["pending approval", "pending receipt", "coupa"],
        "expected": "Should explain difference between these Coupa statuses"
    },
    "Q4": {
        "query": "Free text",
        "key_terms": ["free text", "commandes", "articles non référencés"],
        "expected": "Should explain free text orders for non-catalogued items"
    },
    "Q6": {
        "query": "e-pack",
        "key_terms": ["e-pack", "hygiène", "température", "réception"],
        "expected": "Should explain e-pack hygiene system"
    },
    "Q7": {
        "query": "balance réception marchandises",
        "key_terms": ["balance", "étalonnée", "réception", "matériel"],
        "expected": "Should say balance is required and must be calibrated"
    },
    "Q8": {
        "query": "contrats S@fe",
        "key_terms": ["S@fe", "contrats"],
        "expected": "Should explain if contracts must be on S@fe"
    }
}

print("\n" + "="*80)
print("CHECKING FOR SPECIFIC CONTENT IN VECTORSTORE")
print("="*80)

for qid, info in failing_queries.items():
    print(f"\n{'='*80}")
    print(f"{qid}: {info['query']}")
    print(f"Expected: {info['expected']}")
    print(f"{'='*80}")
    
    # Search with high k to maximize chances
    results = vectorstore.similarity_search_with_score(info['query'], k=50)
    
    print(f"Retrieved {len(results)} documents")
    
    # Check each key term
    found_any = False
    for term in info['key_terms']:
        found_in_docs = []
        for i, (doc, score) in enumerate(results):
            if term.lower() in doc.page_content.lower():
                found_in_docs.append((i+1, score, doc.metadata.get('file_name', 'Unknown')))
        
        if found_in_docs:
            found_any = True
            print(f"\n  ✓ '{term}' found in {len(found_in_docs)} documents:")
            for rank, score, fname in found_in_docs[:3]:  # Show top 3
                print(f"    - Rank {rank}, Score {score:.4f}, File: {fname}")
        else:
            print(f"\n  ❌ '{term}' NOT FOUND in any of the top 50 documents")
    
    if not found_any:
        print(f"\n  ⚠️  CRITICAL: None of the key terms found! This content may not be in the vectorstore.")
    
    # Show top 3 results anyway
    print(f"\n  Top 3 retrieved chunks:")
    for i, (doc, score) in enumerate(results[:3], 1):
        fname = doc.metadata.get('file_name', 'Unknown')
        snippet = doc.page_content[:150].replace('\n', ' ')
        print(f"    {i}. Score {score:.4f} - {fname}")
        print(f"       {snippet}...")

print("\n" + "="*80)
print("DIAGNOSIS SUMMARY")
print("="*80)

print("""
If key terms are NOT FOUND:
→ The documents containing this information were NOT processed or NOT included in Entity_Resorts folder
→ Check if the source documents are in the Entity_Resorts directory
→ Verify document names contain the expected content

If key terms ARE FOUND but in low-ranked documents:
→ The retrieval/scoring is the issue
→ Try the improved retrieval strategies

NEXT STEPS:
1. Check what files are in Entity_Resorts/
2. Verify those files contain the expected answers
3. If files are missing, add them and recreate embeddings
4. If files are present, check if PDFs are text-based or scanned images
""")

# Additional check: List all unique source files
print("\n" + "="*80)
print("CHECKING AVAILABLE SOURCE FILES IN VECTORSTORE")
print("="*80)

# Get a sample of documents to see what sources we have
sample = vectorstore.similarity_search("", k=100)
unique_files = set()
for doc in sample:
    fname = doc.metadata.get('file_name', 'Unknown')
    if fname != 'Unknown':
        unique_files.add(fname)

print(f"\nFound {len(unique_files)} unique source files in vectorstore:")
for fname in sorted(unique_files):
    print(f"  - {fname}")

print("\n" + "="*80)
print("Check if the documents containing Q2, Q4, Q6, Q7, Q8 answers are in this list!")
print("="*80)