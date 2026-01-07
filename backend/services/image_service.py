def process_image(file_path: str, metadata=None):
def ingest_image(image_path):
from backend.ingestion.image import ocr_image

def process_image(file_path: str, metadata=None):
    text = ocr_image(file_path)
    # Add business logic, e.g., store OCR result, update DB, etc.
    return text
