"""
CSV / Excel ingestion — pandas read, 50-row chunks with headers and numeric stats metadata.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List

import pandas as pd

SUPPORTED_EXTENSIONS = {".csv", ".xlsx", ".xls"}


def extract_table_chunks(file_path: str | Path) -> List[Dict[str, Any]]:
    path = Path(file_path)
    extension = path.suffix.lower()
    if extension not in SUPPORTED_EXTENSIONS:
        raise ValueError(f"Unsupported tabular extension: {extension}")

    if extension == ".csv":
        df = pd.read_csv(path)
    else:
        df = pd.read_excel(path)

    if df.empty:
        return []

    total_rows, total_cols = df.shape
    column_names = df.columns.tolist()
    stats = _numeric_stats(df)
    chunks: List[Dict[str, Any]] = []

    for start in range(0, total_rows, 50):
        end = min(start + 50, total_rows)
        chunk_df = df.iloc[start:end]
        # Each chunk includes column headers via pandas CSV header row.
        text = chunk_df.to_csv(index=False)
        metadata: Dict[str, Any] = {
            "filename": path.name,
            "total_rows": total_rows,
            "total_cols": total_cols,
            "column_names": column_names,
            "row_range": f"{start + 1}-{end}",
            "stats": stats,
        }
        chunks.append({"text": text, "metadata": metadata})

    return chunks


def _numeric_stats(df: pd.DataFrame) -> Dict[str, Dict[str, float]]:
    numeric_df = df.select_dtypes(include="number")
    if numeric_df.empty:
        return {}
    stats: Dict[str, Dict[str, float]] = {}
    for column in numeric_df.columns:
        series = numeric_df[column].dropna()
        if series.empty:
            continue
        stats[column] = {
            "min": float(series.min()),
            "max": float(series.max()),
            "mean": float(series.mean()),
        }
    return stats
