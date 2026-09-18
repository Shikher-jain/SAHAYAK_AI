import json
import os
from pathlib import Path

# Adjust this import path based on your exact main.py location
from backend.main import app

def export_openapi_schema():
    """Extracts the OpenAPI schema from FastAPI and writes it to disk."""
    # Define the output directory and file path
    output_dir = Path("docs")
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / "openapi.json"

    # Generate the OpenAPI schema dictionary
    openapi_schema = app.openapi()

    # Write the formatted JSON to disk
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(openapi_schema, f, indent=2)

    print(f"✅ Successfully exported OpenAPI schema to {output_file.absolute()}")

if __name__ == "__main__":
    export_openapi_schema()
