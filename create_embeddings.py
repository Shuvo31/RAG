import os
import pickle
import json
import argparse
from pathlib import Path
from datetime import datetime

from langchain_community.document_loaders import PyPDFLoader, Docx2txtLoader, CSVLoader, UnstructuredExcelLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document

# Configuration
DEFAULT_LOCAL_DIR = r"Entity_Resorts"
FAISS_INDEX_PATH = "faiss_index"
METADATA_FILE = "document_metadata.pkl"

# Fast embeddings
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2",
    model_kwargs={"device": "cpu"},
)

# Text splitter
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1500,
    chunk_overlap=300,
)

def load_image_texts(image_text_file="extracted_image_texts.json"):
    """Load pre-processed image texts and convert to documents"""
    if not Path(image_text_file).exists():
        print("No image text file found. Run image_processor.py first if needed.")
        return []
    
    try:
        with open(image_text_file, 'r', encoding='utf-8') as f:
            image_data = json.load(f)
        
        documents = []
        for item in image_data:
            doc = Document(
                page_content=item['ocr_text'],
                metadata={
                    "file_path": item['pdf_path'],
                    "file_name": item['pdf_file'],
                    "content_type": "image_ocr",
                    "original_page": item['page_number'],
                    "image_index": item['image_index'],
                    "image_size": item['image_size'],
                    "extraction_method": "fast_ocr",
                    "source": f"Image from {item['pdf_file']} (Page {item['page_number']})"
                }
            )
            documents.append(doc)
        
        print(f"Loaded {len(documents)} image text chunks from {image_text_file}")
        return documents
    except Exception as e:
        print(f"Error loading image texts: {e}")
        return []

def load_regular_documents(directory=DEFAULT_LOCAL_DIR):
    """Load regular text documents"""
    p = Path(directory)
    all_docs = []
    
    loaders = {
        '.pdf': PyPDFLoader,
        '.docx': Docx2txtLoader, 
        '.csv': CSVLoader,
        '.xlsx': UnstructuredExcelLoader
    }
    
    for file_path in p.rglob("*"):
        if file_path.is_file() and file_path.suffix.lower() in loaders:
            try:
                loader_class = loaders[file_path.suffix.lower()]
                loader = loader_class(str(file_path))
                docs = loader.load()
                
                for doc in docs:
                    doc.metadata.update({
                        "file_path": str(file_path),
                        "file_name": file_path.name,
                        "content_type": "text",
                        "processed_at": datetime.now().isoformat()
                    })
                
                all_docs.extend(docs)
                print(f"{file_path.name} - {len(docs)} pages")
                
            except Exception as e:
                print(f"{file_path.name}: {e}")
    
    return all_docs

def ensure_vectorstore():
    """Load existing vectorstore if available"""
    if Path(FAISS_INDEX_PATH).exists():
        try:
            vectorstore = FAISS.load_local(
                FAISS_INDEX_PATH, embeddings, allow_dangerous_deserialization=True
            )
            return vectorstore
        except Exception as e:
            print(f"Could not load existing index: {e}")
    return None

def main():
    parser = argparse.ArgumentParser(description="Enhanced embeddings with optional image text")
    parser.add_argument("--dir", default=DEFAULT_LOCAL_DIR, 
                       help=f"Directory to process (default: {DEFAULT_LOCAL_DIR})")
    parser.add_argument("--reset", action="store_true", help="Create new index")
    parser.add_argument("--include-images", action="store_true", 
                       help="Include pre-processed image texts")
    parser.add_argument("--image-file", default="extracted_image_texts.json",
                       help="Image texts JSON file")
    
    args = parser.parse_args()

    print(f"Starting document processing from: {args.dir}")
    
    # Load regular documents
    regular_docs = load_regular_documents(args.dir)
    
    # Load image texts if requested
    image_docs = []
    if args.include_images:
        image_docs = load_image_texts(args.image_file)
    
    all_docs = regular_docs + image_docs
    
    if not all_docs:
        print("No documents found!")
        return
    
    print(f"Total documents: {len(regular_docs)} regular + {len(image_docs)} image = {len(all_docs)} total")
    
    # Split into chunks
    print("Splitting documents into chunks...")
    texts = text_splitter.split_documents(all_docs)
    print(f"Created {len(texts)} chunks")
    
    # Count image chunks for summary
    image_chunks = sum(1 for text in texts if text.metadata.get('content_type') == 'image_ocr')
    
    # Create or update vector store
    if args.reset or not Path(FAISS_INDEX_PATH).exists():
        print("Creating new FAISS index...")
        vectorstore = FAISS.from_documents(texts, embeddings)
    else:
        print("Updating existing FAISS index...")
        vectorstore = ensure_vectorstore()
        if vectorstore:
            vectorstore.add_documents(texts)
            print("Updated existing index")
        else:
            vectorstore = FAISS.from_documents(texts, embeddings)
            print("Created new index (existing one unavailable)")
    
    # Save vector store
    vectorstore.save_local(FAISS_INDEX_PATH)
    
    # Save metadata
    metadata = {
        "total_chunks": len(texts),
        "regular_chunks": len(regular_docs),
        "image_chunks": image_chunks,
        "source_directory": args.dir,
        "created_at": datetime.now().isoformat(),
        "includes_image_texts": args.include_images
    }
    with open(METADATA_FILE, "wb") as f:
        pickle.dump(metadata, f)
    
    print(f"FAISS index created successfully!")
    print(f"Summary: {len(texts)} total chunks ({image_chunks} from images)")

if __name__ == "__main__":
    main()