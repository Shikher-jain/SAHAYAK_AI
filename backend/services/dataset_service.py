from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List

DATASET_PATH = Path(__file__).resolve().parents[2] / "data" / "ahayak" / "fine_tune_dataset.jsonl"
DATASET_PATH.parent.mkdir(parents=True, exist_ok=True)


def append_example(prompt: str, completion: str, metadata: Dict[str, str] | None = None) -> None:
    metadata = metadata or {}
    record = {"prompt": prompt.strip(), "completion": completion.strip(), "metadata": metadata}
    with DATASET_PATH.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def load_examples(limit: int = 50) -> List[Dict[str, str]]:
    if not DATASET_PATH.exists():
        return []
    examples: List[Dict[str, str]] = []
    with DATASET_PATH.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            try:
                examples.append(json.loads(line))
            except json.JSONDecodeError:
                continue
            if len(examples) >= limit:
                break
    return examples


def overwrite_dataset(examples: List[Dict[str, str]]) -> None:
    with DATASET_PATH.open("w", encoding="utf-8") as handle:
        for item in examples:
            handle.write(json.dumps(item, ensure_ascii=False) + "\n")


def dataset_stats() -> Dict[str, int]:
    return {"total_examples": len(load_examples(limit=10_000))}
