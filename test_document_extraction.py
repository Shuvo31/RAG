"""
Test if specific documents are being loaded and if they contain expected content
"""
from pathlib import Path
from langchain_community.document_loaders import PyPDFLoader

# Documents that should contain the answers
TARGET_DOCS = {
    "SC-P-WW-07 - FR": {
        "file_pattern": "SC-P-WW-07 - FR",
        "expected_terms": ["pending approval", "pending receipt", "en attente", "validation"],
        "question": "Q2 - pending approval vs pending receipt"
    },
    "SC-P-WW-07 - EN": {
        "file_pattern": "SC-P-WW-07 - EN",
        "expected_terms": ["pending approval", "pending receipt"],
        "question": "Q2 - pending approval vs pending receipt"
    },
    "SC-P-WW-01 - FR": {
        "file_pattern": "SC-P-WW-01 - FR",
        "expected_terms": ["free text", "articles non référencés", "commande"],
        "question": "Q4 - Free text definition"
    },
    "SC-P-WW-03 - FR": {
        "file_pattern": "SC-P-WW-03 - FR",
        "expected_terms": ["e-pack", "balance", "étalonnée", "réception"],
        "question": "Q6 - e-pack & Q7 - balance"
    },
}

def find_and_test_document(base_dir, file_pattern, expected_terms):
    """Find document and check if it contains expected terms"""
    p = Path(base_dir)
    
    # Find matching file
    matching_files = list(p.rglob(f"*{file_pattern}*.pdf"))
    
    if not matching_files:
        return None, f"❌ File matching '{file_pattern}' not found"
    
    if len(matching_files) > 1:
        print(f"⚠️  Multiple files match '{file_pattern}':")
        for f in matching_files:
            print(f"   - {f.name}")
    
    pdf_path = matching_files[0]
    
    try:
        # Load PDF
        loader = PyPDFLoader(str(pdf_path))
        docs = loader.load()
        
        # Combine all pages
        full_content = "\n".join([doc.page_content for doc in docs])
        content_lower = full_content.lower()
        
        # Check for terms
        found_terms = []
        missing_terms = []
        
        for term in expected_terms:
            if term.lower() in content_lower:
                found_terms.append(term)
            else:
                missing_terms.append(term)
        
        # Extract sample text
        sample = full_content[:500] if len(full_content) > 500 else full_content
        
        return {
            "file": pdf_path.name,
            "pages": len(docs),
            "total_chars": len(full_content),
            "found_terms": found_terms,
            "missing_terms": missing_terms,
            "sample": sample
        }, None
        
    except Exception as e:
        return None, f"❌ Error loading PDF: {e}"

def main():
    print("="*80)
    print("VERIFYING DOCUMENT CONTENT EXTRACTION")
    print("="*80)
    
    base_dir = "Entity_Resorts"
    
    if not Path(base_dir).exists():
        print(f"❌ Directory '{base_dir}' not found!")
        print("Please run from the directory containing Entity_Resorts/")
        return
    
    all_good = True
    
    for doc_key, doc_info in TARGET_DOCS.items():
        print(f"\n{'='*80}")
        print(f"Testing: {doc_key}")
        print(f"Question: {doc_info['question']}")
        print(f"{'='*80}")
        
        result, error = find_and_test_document(
            base_dir, 
            doc_info['file_pattern'], 
            doc_info['expected_terms']
        )
        
        if error:
            print(error)
            all_good = False
            continue
        
        print(f"✓ File: {result['file']}")
        print(f"✓ Pages: {result['pages']}")
        print(f"✓ Total characters extracted: {result['total_chars']:,}")
        
        if result['found_terms']:
            print(f"\n✓ Found terms ({len(result['found_terms'])}/{len(doc_info['expected_terms'])}):")
            for term in result['found_terms']:
                print(f"  - {term}")
        
        if result['missing_terms']:
            print(f"\n❌ Missing terms ({len(result['missing_terms'])}/{len(doc_info['expected_terms'])}):")
            for term in result['missing_terms']:
                print(f"  - {term}")
            all_good = False
        
        print(f"\nSample content (first 500 chars):")
        print("-" * 80)
        print(result['sample'])
        print("-" * 80)
    
    print("\n" + "="*80)
    print("DIAGNOSIS")
    print("="*80)
    
    if all_good:
        print("""
✓ All documents found and contain expected terms!

The issue is likely with:
1. How the content is being chunked
2. How embeddings are being created
3. The retrieval/scoring mechanism

NEXT STEPS:
1. Recreate embeddings with verbose output to see what's happening
2. Check if chunks are being created properly
3. Test retrieval with the exact terms
        """)
    else:
        print("""
❌ Some documents missing content or terms not found!

This could mean:
1. PDFs are scanned images (need OCR)
2. Terms use different wording in the documents
3. Content is in images/tables not extracted as text

NEXT STEPS:
1. Open the PDFs manually and search for the terms
2. If text is searchable → might be extraction issue
3. If text is NOT searchable → PDFs are scanned, need OCR
4. Try with --include-images flag if you have image extraction
        """)
    
    print("\n" + "="*80)
    print("QUICK TEST COMMAND")
    print("="*80)
    print("""
# Test if you can extract text from a specific PDF:
python -c "
from langchain_community.document_loaders import PyPDFLoader
loader = PyPDFLoader('Entity_Resorts/SC-P-WW-07 - FR - Procédure traitement facturation et avoirs - Mai 24.pdf')
docs = loader.load()
print(f'Pages: {len(docs)}')
print(f'First page length: {len(docs[0].page_content)}')
print(docs[0].page_content[:500])
"
    """)

if __name__ == "__main__":
    main()