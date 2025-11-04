"""
Diagnostic script to check image content in FAISS vectorstore
Run this to verify that image content was properly added
"""

from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from pathlib import Path

FAISS_INDEX_PATH = "faiss_index"

def check_image_content():
    """Check if image content exists in the vectorstore"""
    
    print("=" * 70)
    print("VECTORSTORE IMAGE CONTENT DIAGNOSTIC")
    print("=" * 70)
    
    # Check if index exists
    if not Path(FAISS_INDEX_PATH).exists():
        print(f"❌ Error: FAISS index not found at {FAISS_INDEX_PATH}")
        return
    
    print(f"✅ Found FAISS index at {FAISS_INDEX_PATH}")
    
    # Load embeddings
    print("\n📦 Loading embeddings model...")
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={"device": "cpu"}
    )
    
    # Load vectorstore
    print("📚 Loading FAISS vectorstore...")
    try:
        vectorstore = FAISS.load_local(
            FAISS_INDEX_PATH, 
            embeddings, 
            allow_dangerous_deserialization=True
        )
        print("✅ Vectorstore loaded successfully")
    except Exception as e:
        print(f"❌ Error loading vectorstore: {e}")
        return
    
    # Get total document count
    total_docs = vectorstore.index.ntotal
    print(f"\n📊 Total documents in vectorstore: {total_docs}")
    
    # Method 1: Try multiple search queries
    print("\n" + "=" * 70)
    print("METHOD 1: Similarity Search with Various Queries")
    print("=" * 70)
    
    search_queries = [
        "image chart diagram",
        "flowchart process",
        "table data",
        "visual content",
        "figure illustration"
    ]
    
    found_any = False
    for query in search_queries:
        try:
            docs = vectorstore.similarity_search(query, k=20)  # Increased k
            image_docs = [doc for doc in docs if doc.metadata.get('content_type') == 'image_ocr']
            
            if image_docs:
                print(f"✅ Query '{query}': Found {len(image_docs)} image documents")
                found_any = True
                
                # Show sample
                if len(image_docs) > 0:
                    print(f"   Sample: {image_docs[0].metadata.get('file_name', 'Unknown')}")
                    print(f"   Text preview: {image_docs[0].page_content[:100]}...")
            else:
                print(f"⚪ Query '{query}': No image documents found")
        except Exception as e:
            print(f"❌ Query '{query}' failed: {e}")
    
    # Method 2: Sample all documents and count
    print("\n" + "=" * 70)
    print("METHOD 2: Direct Document Sampling")
    print("=" * 70)
    
    try:
        # Get a large sample
        sample_docs = vectorstore.similarity_search("document", k=min(100, total_docs))
        
        print(f"📋 Sampled {len(sample_docs)} documents")
        
        # Count by content type
        content_types = {}
        image_count = 0
        
        for doc in sample_docs:
            content_type = doc.metadata.get('content_type', 'unknown')
            content_types[content_type] = content_types.get(content_type, 0) + 1
            
            if content_type == 'image_ocr':
                image_count += 1
        
        print(f"\n📊 Content Type Distribution (from sample):")
        for ct, count in content_types.items():
            icon = "📸" if ct == "image_ocr" else "📄"
            print(f"   {icon} {ct}: {count} documents")
        
        if image_count > 0:
            print(f"\n✅ SUCCESS: Found {image_count} image documents in sample!")
            
            # Show examples
            image_examples = [doc for doc in sample_docs if doc.metadata.get('content_type') == 'image_ocr']
            print(f"\n📸 Sample Image Documents:")
            for i, doc in enumerate(image_examples[:3], 1):
                print(f"\n   {i}. File: {doc.metadata.get('file_name', 'Unknown')}")
                print(f"      Page: {doc.metadata.get('original_page', '?')}")
                print(f"      Text: {doc.page_content[:80]}...")
        else:
            print(f"\n❌ WARNING: No image documents found in sample of {len(sample_docs)} docs")
            
    except Exception as e:
        print(f"❌ Sampling failed: {e}")
    
    # Method 3: Check metadata directly
    print("\n" + "=" * 70)
    print("METHOD 3: Metadata Analysis")
    print("=" * 70)
    
    try:
        # Try to get documents with specific metadata
        all_metadata_types = set()
        
        # Sample more documents
        for query in ["text", "image", "document", "content", "data"]:
            docs = vectorstore.similarity_search(query, k=20)
            for doc in docs:
                ct = doc.metadata.get('content_type', 'none')
                all_metadata_types.add(ct)
        
        print(f"📋 Found these content_type values:")
        for ct in sorted(all_metadata_types):
            icon = "📸" if ct == "image_ocr" else "📄"
            print(f"   {icon} '{ct}'")
        
        if 'image_ocr' in all_metadata_types:
            print(f"\n✅ 'image_ocr' metadata exists in vectorstore!")
        else:
            print(f"\n❌ 'image_ocr' metadata NOT found!")
            print(f"   Found instead: {all_metadata_types}")
            
    except Exception as e:
        print(f"❌ Metadata analysis failed: {e}")
    
    # Final diagnosis
    print("\n" + "=" * 70)
    print("DIAGNOSIS")
    print("=" * 70)
    
    if found_any:
        print("✅ IMAGE CONTENT IS PRESENT in the vectorstore")
        print("✅ The UI should be able to find it")
        print("\n💡 Possible issues:")
        print("   1. UI cache needs clearing - try restarting Streamlit")
        print("   2. UI search query is too specific - it uses 'image chart diagram'")
        print("   3. Try clicking the button multiple times")
    else:
        print("❌ IMAGE CONTENT NOT FOUND in the vectorstore")
        print("\n💡 Possible causes:")
        print("   1. The script ran but didn't save properly")
        print("   2. Wrong FAISS index is being loaded")
        print("   3. Metadata field name mismatch")
        print("   4. Images were added to a different index")
        print("\n🔧 Next steps:")
        print("   1. Re-run: python create_embeddings.py --reset --include-images")
        print("   2. Make sure you see '📸 Loaded X image text chunks'")
        print("   3. Make sure you see '🎉 FAISS index created successfully!'")
    
    print("\n" + "=" * 70)

if __name__ == "__main__":
    check_image_content()