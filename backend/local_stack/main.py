from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from pathlib import Path
import os

from . import extractor
from .rag_engine import answer_question
from .embedder import embed_text
from .db import add_chunk, init_db

BASE_DIR = Path(__file__).resolve().parents[2]
PDF_FOLDER = BASE_DIR / "data" / "sahayak_09_02" / "pdf_storage"
PDF_FOLDER.mkdir(parents=True, exist_ok=True)

app = FastAPI(title="Sahayak Local RAG")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize local vector stack on startup - FAIL FAST
print("\n" + "="*60)
print("🚀 Initializing Sahayak AI Teaching Assistant")
print("="*60)
try:
    init_db()
except RuntimeError as e:
    print(f"\n{'='*60}")
    print(f"❌ STARTUP FAILED")
    print(f"{'='*60}")
    print(f"{e}")
    print(f"{'='*60}\n")
    raise SystemExit(1)
print("="*60 + "\n")

@app.post("/upload")
async def upload_file(file: UploadFile = File(...)):
    """Upload PDF/Image → Extract text → Generate embeddings → persist locally"""
    content = await file.read()
    try:
        filename = file.filename
        save_path = PDF_FOLDER / filename
        print(f"\n📄 Processing: {filename}")
        # Save file locally
        with open(save_path, "wb") as f:
            f.write(content)
        print(f"  ✓ Saved to {save_path}")
        # Extract text based on file type
        if filename.lower().endswith(".pdf"):
            text = extractor.extract_pdf(content)
            print(f"  ✓ Extracted {len(text)} characters from PDF")
        elif filename.lower().endswith((".png", ".jpg", ".jpeg")):
            text = extractor.extract_image(content)
            print(f"  ✓ OCR extracted {len(text)} characters from image")
        else:
            return {"error": "Unsupported file type. Please upload PDF or image (PNG/JPG)"}
        if not text.strip():
            return {"error": "No text could be extracted from the file"}
        # Split into chunks (larger chunks for better context)
        chunk_size = 800
        overlap = 100
        chunks = []
        for i in range(0, len(text), chunk_size - overlap):
            chunk = text[i:i+chunk_size].strip()
            if chunk:
                chunks.append(chunk)
        print(f"  ✓ Split into {len(chunks)} chunks")
        
        # Generate embeddings and store in the local vector DB
        for idx, chunk in enumerate(chunks):
            emb = embed_text(chunk)
            add_chunk(filename, chunk, emb)
        
        print(f"  ✓ Stored {len(chunks)} chunks in the local vector DB\n")
        
        return {
            "message": f"✓ {filename} uploaded successfully!",
            "details": {
                "filename": filename,
                "text_length": len(text),
                "chunks_created": len(chunks)
            }
        }
    except Exception as e:
        print(f"  ✗ Error: {str(e)}\n")
        return {"error": f"Processing failed: {str(e)}"}
    
@app.get("/vector_store_health")
def vector_store_health():
    """Check vector DB connectivity (local FAISS in this mode)."""
    return {"vector_store": "local", "status": "local-only"}

@app.get("/ask")
def ask(question: str):
    """Query documents using RAG backed by the local FAISS index."""
    try:
        print(f"\n❓ Question: {question}")
        answer = answer_question(question)
        print(f"✓ Answer generated\n")
        return {"answer": answer}
    except Exception as e:
        print(f"✗ Error: {str(e)}\n")
        return {"error": str(e)}

@app.get("/health")
def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "message": "Sahayak API is running"}

@app.get("/")
def root():
    """Root endpoint"""
    return {
        "message": "Sahayak AI Teaching Assistant API",
        "endpoints": {
            "/upload": "POST - Upload PDF/Image files",
            "/ask": "GET - Ask questions about uploaded documents",
            "/health": "GET - Health check"
        }
    }
