
def process_pdf(file_path: str, metadata=None):
from backend.ingestion.pdf import extract_pdf_text

def process_pdf(file_path: str, metadata=None):
    text = extract_pdf_text(file_path)
    # Add business logic, e.g., store extracted text, update DB, etc.
    print(f"Ingested PDF: {file_path}, length: {len(text)}")
    return text
