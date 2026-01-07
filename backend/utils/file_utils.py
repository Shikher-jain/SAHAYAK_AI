from pathlib import Path

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
