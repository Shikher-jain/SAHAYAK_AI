from backend.common.data_paths import get_finetune_pdf_storage_dir
from . import extractor
from backend.local_stack.app_factory import create_local_rag_app

from .db import add_chunk, init_db
from .embedder import embed_text
from .rag_engine import answer_question

PDF_FOLDER = get_finetune_pdf_storage_dir()

app = create_local_rag_app(
    title="Sahayak Fine-tune Helper",
    storage_dir=PDF_FOLDER,
    init_db=init_db,
    add_chunk=add_chunk,
    embed_text=embed_text,
    extract_pdf=extractor.extract_pdf,
    extract_image=extractor.extract_image,
    answer_question=answer_question,
)
