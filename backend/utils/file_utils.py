from __future__ import annotations

import os
import tempfile
import time
from pathlib import Path
from dotenv import load_dotenv
load_dotenv()
TMP_DIR = Path.home() / ".sahayak_ai" / "tmp"
TMP_DIR.mkdir(parents=True, exist_ok=True)


def save_upload_to_tmp(upload_file, suffix: str = "") -> Path:
    filename = upload_file.filename or "upload.bin"
    target = TMP_DIR / f"{filename}{suffix}"
    with open(target, "wb") as buffer:
        buffer.write(upload_file.file.read())
    upload_file.file.seek(0)
    return target


def write_bytes(path: Path, data: bytes) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)
    return path


def get_tmp_path(name: str) -> Path:
    return TMP_DIR / name


def create_named_temp_from_bytes(data: bytes, original_name: str = "upload.bin") -> Path:
    suffix = Path(original_name).suffix or ".bin"
    with tempfile.NamedTemporaryFile(dir=TMP_DIR, suffix=suffix, delete=False) as temp_handle:
        temp_handle.write(data)
        return Path(temp_handle.name)


def safe_unlink(path: str | Path) -> None:
    try:
        Path(path).unlink(missing_ok=True)
    except Exception:
        return


def cleanup_tmp_dir(max_age_seconds: int = 3600) -> int:
    cutoff = time.time() - max_age_seconds
    removed = 0
    for item in TMP_DIR.glob("*"):
        try:
            if not item.is_file():
                continue
            if os.path.getmtime(item) <= cutoff:
                item.unlink(missing_ok=True)
                removed += 1
        except Exception:
            continue
    return removed
