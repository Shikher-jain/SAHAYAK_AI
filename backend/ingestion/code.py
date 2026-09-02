"""
Code file ingestion — AST-based Python chunks; regex boundaries for other languages.

Returns list of {text, metadata} with language, chunk_type, names, and docstrings.
"""
from __future__ import annotations

import ast
import re
from pathlib import Path
from typing import Dict, List

SUPPORTED_EXTENSIONS = {
    ".py": "python",
    ".js": "javascript",
    ".ts": "typescript",
    ".cpp": "cpp",
    ".c": "c",
    ".java": "java",
    ".go": "go",
    ".rs": "rust",
    ".rb": "ruby",
    ".php": "php",
    ".cs": "csharp",
    ".swift": "swift",
    ".kt": "kotlin",
    ".sh": "bash",
    ".html": "html",
    ".css": "css",
}

_JS_TS_FUNCTION = re.compile(r"^\s*(?:export\s+)?(?:async\s+)?function\s+(\w+)\s*\(")
_JS_TS_CLASS = re.compile(r"^\s*(?:export\s+)?class\s+(\w+)")
_GENERIC_CLASS = re.compile(r"^\s*class\s+(\w+)")
_GENERIC_FUNCTION = re.compile(
    r"^\s*(?:public|private|protected|static|inline|virtual|async|func)?\s*([\w:<>,~*&\[\]]+)\s+(\w+)\s*\("
)
_GO_FUNCTION = re.compile(r"^\s*func\s+(?:\([^)]+\)\s+)?(\w+)\s*\(")


def extract_code_chunks(file_path: str | Path) -> List[Dict[str, Dict[str, str]]]:
    path = Path(file_path)
    extension = path.suffix.lower()
    language = SUPPORTED_EXTENSIONS.get(extension, "text")
    content = path.read_text(encoding="utf-8", errors="ignore")

    if not content.strip():
        return []

    if extension == ".py":
        return _extract_python_chunks(content, language)

    # For extensions we recognise structurally, use the regex chunker;
    # for everything else, emit a single module-level chunk so the text
    # still gets stored rather than raising a ValueError.
    if extension in {".js", ".ts", ".cpp", ".c", ".java", ".go",
                     ".rs", ".rb", ".php", ".cs", ".swift", ".kt"}:
        return _extract_generic_chunks(content, language, extension)

    # Catch-all: store as a single opaque module chunk
    return [{"text": content, "metadata": {"language": language, "chunk_type": "module"}}]


def _extract_python_chunks(source: str, language: str) -> List[Dict[str, Dict[str, str]]]:
    chunks: List[Dict[str, Dict[str, str]]] = []
    if not source.strip():
        return chunks
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return [
            {
                "text": source,
                "metadata": {"language": language, "chunk_type": "module"},
            }
        ]

    lines = source.splitlines()
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            name = node.name
            start = max(node.lineno - 1, 0)
            end = node.end_lineno or node.lineno
            snippet = "\n".join(lines[start:end])
            metadata: Dict[str, str] = {
                "language": language,
                "chunk_type": "function",
                "function_name": name,
            }
            docstring = ast.get_docstring(node)
            if docstring:
                metadata["docstring"] = docstring
            chunks.append({"text": snippet, "metadata": metadata})
        elif isinstance(node, ast.ClassDef):
            name = node.name
            start = max(node.lineno - 1, 0)
            end = node.end_lineno or node.lineno
            snippet = "\n".join(lines[start:end])
            metadata = {
                "language": language,
                "chunk_type": "class",
                "class_name": name,
            }
            docstring = ast.get_docstring(node)
            if docstring:
                metadata["docstring"] = docstring
            chunks.append({"text": snippet, "metadata": metadata})

    if not chunks:
        metadata = {"language": language, "chunk_type": "module"}
        module_doc = ast.get_docstring(tree)
        if module_doc:
            metadata["docstring"] = module_doc
        chunks.append({"text": source, "metadata": metadata})
    return chunks


def _extract_generic_chunks(source: str, language: str, extension: str) -> List[Dict[str, Dict[str, str]]]:
    lines = source.splitlines()
    boundaries: List[Dict[str, str | int]] = []

    for idx, line in enumerate(lines):
        match = _JS_TS_CLASS.match(line) if extension in {".js", ".ts"} else _GENERIC_CLASS.match(line)
        if match:
            boundaries.append({"index": idx, "chunk_type": "class", "name": match.group(1)})
            continue
        if extension in {".js", ".ts"}:
            func_match = _JS_TS_FUNCTION.match(line)
            if func_match:
                boundaries.append({"index": idx, "chunk_type": "function", "name": func_match.group(1)})
                continue
        if extension == ".go":
            go_match = _GO_FUNCTION.match(line)
            if go_match:
                boundaries.append({"index": idx, "chunk_type": "function", "name": go_match.group(1)})
                continue
        func_match = _GENERIC_FUNCTION.match(line)
        if func_match:
            boundaries.append({"index": idx, "chunk_type": "function", "name": func_match.group(2)})

    if not boundaries:
        if not source.strip():
            return []
        return [
            {
                "text": source,
                "metadata": {"language": language, "chunk_type": "module"},
            }
        ]

    chunks: List[Dict[str, Dict[str, str]]] = []
    for idx, boundary in enumerate(boundaries):
        start = int(boundary["index"])
        end = int(boundaries[idx + 1]["index"]) if idx + 1 < len(boundaries) else len(lines)
        snippet = "\n".join(lines[start:end]).strip()
        if not snippet:
            continue
        metadata: Dict[str, str] = {
            "language": language,
            "chunk_type": str(boundary["chunk_type"]),
        }
        if boundary["chunk_type"] == "class":
            metadata["class_name"] = str(boundary["name"])
        else:
            metadata["function_name"] = str(boundary["name"])
        docstring = _extract_leading_comment(lines, start)
        if docstring:
            metadata["docstring"] = docstring
        chunks.append({"text": snippet, "metadata": metadata})

    return chunks


def _extract_leading_comment(lines: List[str], start_idx: int) -> str:
    if start_idx == 0:
        return ""
    comment_lines: List[str] = []
    idx = start_idx - 1
    while idx >= 0:
        line = lines[idx].rstrip()
        stripped = line.strip()
        if not stripped:
            idx -= 1
            continue
        if stripped.startswith("//"):
            comment_lines.insert(0, stripped.lstrip("/").strip())
            idx -= 1
            continue
        if stripped.endswith("*/"):
            comment_lines.insert(0, stripped.rstrip("*/").strip())
            idx -= 1
            while idx >= 0:
                stripped = lines[idx].strip()
                comment_lines.insert(0, stripped.lstrip("/*").strip())
                if stripped.startswith("/*"):
                    idx -= 1
                    break
                idx -= 1
            break
        break
    return " ".join(part for part in comment_lines if part)
