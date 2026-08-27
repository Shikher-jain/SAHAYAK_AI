from pathlib import Path
import shutil

ROOT = Path(__file__).resolve().parent

# Find only __pycache__ directories and .pyc files
pycache_dirs = [
    path
    for path in ROOT.rglob("__pycache__")
    if path.is_dir()
]

pyc_files = [
    path
    for path in ROOT.rglob("*.pyc")
    if path.is_file()
]

# Nothing to delete
if not pycache_dirs and not pyc_files:
    print("No __pycache__ directories or .pyc files found.")
    raise SystemExit(0)

# Show deletion preview
print("=" * 70)
print("PYTHON CACHE CLEANUP")
print("=" * 70)

print(f"\nProject root:")
print(f"  {ROOT}")

print(f"\n__pycache__ directories found: {len(pycache_dirs)}")

for path in pycache_dirs:
    print(f"  [DIR]  {path}")

print(f"\n.pyc files found: {len(pyc_files)}")

for path in pyc_files:
    print(f"  [FILE] {path}")

print("\n" + "=" * 70)
print("WARNING: Only the above __pycache__ directories and .pyc files")
print("will be deleted. No other file types will be touched.")
print("=" * 70)

# Human approval
answer = input("\nDo you want to DELETE these files? [y/N]: ").strip().lower()

if answer not in {"y", "yes"}:
    print("\nCleanup cancelled. Nothing was deleted.")
    raise SystemExit(0)

# Delete __pycache__ directories
deleted_dirs = 0

for path in pycache_dirs:
    try:
        shutil.rmtree(path)
        print(f"Deleted directory: {path}")
        deleted_dirs += 1
    except Exception as exc:
        print(f"FAILED to delete directory: {path}")
        print(f"Reason: {exc}")

# Delete standalone .pyc files
deleted_files = 0

for path in pyc_files:
    try:
        # It may already be gone if it was inside a deleted __pycache__
        if path.exists():
            path.unlink()
            print(f"Deleted file: {path}")
            deleted_files += 1
    except Exception as exc:
        print(f"FAILED to delete file: {path}")
        print(f"Reason: {exc}")

print("\n" + "=" * 70)
print("CLEANUP COMPLETE")
print("=" * 70)
print(f"__pycache__ directories deleted: {deleted_dirs}")
print(f".pyc files deleted:              {deleted_files}")