import os
import pickle
import argparse
from pathlib import Path
from langchain_community.document_loaders import (
    Docx2txtLoader,
    PyPDFLoader,
    CSVLoader,
    UnstructuredExcelLoader,
)
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
import certifi  # optional; fine to keep


# Defaults
DEFAULT_LOCAL_DIR = r"Entity_Resorts"
FAISS_INDEX_PATH = "faiss_index"
METADATA_FILE = "document_metadata.pkl"

# Supported file types and loaders
SUPPORTED_EXTENSIONS = {
    ".docx": Docx2txtLoader,
    ".pdf": PyPDFLoader,
    ".csv": CSVLoader,
    ".xlsx": UnstructuredExcelLoader,
}

# Embeddings model
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2",
    model_kwargs={"device": "cpu"},
)

# Splitter
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=250)

def load_docs_from_path(path: str):
    """Load either a single file or an entire folder tree."""
    p = Path(path)
    docs = []
    if p.is_file():
        ext = p.suffix.lower()
        if ext in SUPPORTED_EXTENSIONS:
            loader = SUPPORTED_EXTENSIONS[ext](str(p))
            docs.extend(loader.load())
    else:
        for root, _, files in os.walk(str(p)):
            for f in files:
                ext = os.path.splitext(f)[1].lower()
                if ext in SUPPORTED_EXTENSIONS:
                    fp = os.path.join(root, f)
                    try:
                        loader = SUPPORTED_EXTENSIONS[ext](fp)
                        docs.extend(loader.load())
                    except Exception as e:
                        print(f"Error loading file {fp}: {e}")
    return docs

def ensure_vectorstore():
    if Path(FAISS_INDEX_PATH).exists():
        try:
            vs = FAISS.load_local(
                FAISS_INDEX_PATH, embeddings, allow_dangerous_deserialization=True
            )
            return vs
        except Exception as e:
            print(f"[WARN] Could not load existing index, creating new. Reason: {e}")
    return None

def save_metadata(count_delta: int, reset: bool = False):
    meta = {"document_count": 0}
    if not reset and Path(METADATA_FILE).exists():
        try:
            with open(METADATA_FILE, "rb") as f:
                meta = pickle.load(f) or meta
        except Exception:
            pass
    meta["document_count"] = int(meta.get("document_count", 0)) + int(count_delta)
    with open(METADATA_FILE, "wb") as f:
        pickle.dump(meta, f)

def main():
    ap = argparse.ArgumentParser(description="Embeddings ingestion (incremental or batch).")
    ap.add_argument("--file", help="Ingest a single file incrementally.")
    ap.add_argument("--dir", default=DEFAULT_LOCAL_DIR, help="Ingest an entire directory.")
    ap.add_argument("--reset", action="store_true", help="Rebuild index from scratch with --dir.")
    args = ap.parse_args()

    if args.file:
        # Incremental single-file add/update
        docs = load_docs_from_path(args.file)
        if not docs:
            print(f"[INFO] No loadable docs for: {args.file}")
            return
        texts = text_splitter.split_documents(docs)
        vs = ensure_vectorstore()
        if vs is None:
            vs = FAISS.from_documents(texts, embeddings)
        else:
            vs.add_documents(texts)
        vs.save_local(FAISS_INDEX_PATH)
        save_metadata(len(texts), reset=False)
        print(f"[OK] Added/updated 1 file -> chunks: {len(texts)}")
        return

    # Batch directory mode (for full rebuilds or first-time)
    docs = load_docs_from_path(args.dir)
    texts = text_splitter.split_documents(docs)

    if args.reset or not Path(FAISS_INDEX_PATH).exists():
        vs = FAISS.from_documents(texts, embeddings)
        save_metadata(len(texts), reset=True)
    else:
        vs = ensure_vectorstore()
        if vs is None:
            vs = FAISS.from_documents(texts, embeddings)
            save_metadata(len(texts), reset=True)
        else:
            # Rebuild to keep index perfectly in sync with the folder state
            vs = FAISS.from_documents(texts, embeddings)
            save_metadata(len(texts), reset=True)

    vs.save_local(FAISS_INDEX_PATH)
    print(f"FAISS index and metadata saved to {FAISS_INDEX_PATH} and {METADATA_FILE}")

if __name__ == "__main__":
    main()
