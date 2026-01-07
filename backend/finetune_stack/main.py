from fastapi import FastAPI, UploadFile
from pathlib import Path

from . import extractor
from .rag_engine import answer_question
from .embedder import embed_text
from .db import add_chunk, init_db

DATA_DIR = Path(__file__).resolve().parents[2] / "data" / "ahayak"
PDF_FOLDER = DATA_DIR / "pdf_storage"
PDF_FOLDER.mkdir(parents=True, exist_ok=True)

app = FastAPI(title="Sahayak Fine-tune Helper")
init_db()
@app.post("/upload")
async def upload_file(file: UploadFile):
    content = await file.read()
    try:
        filename = file.filename
        save_path = PDF_FOLDER / filename
        # Save file locally
        with open(save_path, "wb") as f:
            f.write(content)

        # Extract text
        if filename.endswith(".pdf"):
            text = extractor.extract_pdf(content)
        elif filename.lower().endswith((".png", ".jpg")):
            text = extractor.extract_image(content)
        else:
            return {"error": "Unsupported file type"}

        # Split into chunks
        chunks = [text[i:i+500] for i in range(0, len(text), 500)]
        for c in chunks:
            emb = embed_text(c)
            add_chunk(filename, c, emb)

        return {"message": f"{filename} uploaded and processed successfully"}
    except Exception as e:
        return {"error": str(e)}

@app.get("/ask")
def ask(question: str):
    try:
        answer = answer_question(question)  
        return {"answer": answer}
    except Exception as e:
        return {"error": str(e)}
