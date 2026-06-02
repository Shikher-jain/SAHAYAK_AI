from __future__ import annotations

from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[2]
DATA_ROOT = BASE_DIR / "data"

_PREFERRED_LOCAL_DIR = DATA_ROOT / "local"
_PREFERRED_FINETUNE_DIR = DATA_ROOT / "finetune"


def get_local_data_dir() -> Path:
    _PREFERRED_LOCAL_DIR.mkdir(parents=True, exist_ok=True)
    return _PREFERRED_LOCAL_DIR


def get_finetune_data_dir() -> Path:
    _PREFERRED_FINETUNE_DIR.mkdir(parents=True, exist_ok=True)
    return _PREFERRED_FINETUNE_DIR


def get_local_pdf_storage_dir() -> Path:
    path = get_local_data_dir() / "pdf_storage"
    path.mkdir(parents=True, exist_ok=True)
    return path


def get_finetune_pdf_storage_dir() -> Path:
    path = get_finetune_data_dir() / "pdf_storage"
    path.mkdir(parents=True, exist_ok=True)
    return path


def get_local_db_path() -> Path:
    path = get_local_data_dir() / "pdf_memory.db"
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def get_finetune_db_path() -> Path:
    path = get_finetune_data_dir() / "pdf_memory.db"
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def get_finetune_dataset_path() -> Path:
    path = get_finetune_data_dir() / "fine_tune_dataset.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    return path
