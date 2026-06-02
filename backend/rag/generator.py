from __future__ import annotations

import os
from typing import Dict, List


class Generator:
    def __init__(self, model_name: str = "google/flan-t5-base") -> None:
        self.model_name = model_name
        self._pipeline = None
        self._backend = None

    def _get_pipeline(self):
        if self._pipeline is None:
            try:
                from transformers import pipeline

                self._pipeline = pipeline("text2text-generation", model=self.model_name)
            except Exception:
                self._pipeline = None
        return self._pipeline

    def _select_backend(self) -> str:
        """
        Select LLM backend by priority:
        1) Groq if GROQ_API_KEY exists
        2) OpenAI if OPENAI_API_KEY exists
        3) HuggingFace flan-t5 fallback
        """

        if self._backend:
            return self._backend
        if os.getenv("GROQ_API_KEY"):
            self._backend = "groq"
            return self._backend
        if os.getenv("OPENAI_API_KEY"):
            self._backend = "openai"
            return self._backend
        self._backend = "hf"
        return self._backend

    def _fallback_answer(self, context: str, question: str) -> str:
        question_tokens = {token.lower() for token in question.split() if len(token) > 2}
        sentences = [segment.strip() for segment in context.replace("\n", " ").split(".") if segment.strip()]
        scored: List[tuple[int, str]] = []
        for sentence in sentences:
            sentence_tokens = {token.lower() for token in sentence.split()}
            overlap = len(question_tokens.intersection(sentence_tokens))
            scored.append((overlap, sentence))
        scored.sort(key=lambda item: item[0], reverse=True)
        if not scored:
            return "I could not find enough context to answer this question."
        best = [item[1] for item in scored[:2] if item[0] > 0]
        if not best:
            best = [scored[0][1]]
        return ". ".join(best).strip() + "."

    def _generate_groq(self, prompt: str) -> str:
        """Generate an answer using Groq chat completions."""

        try:
            from groq import Groq
        except Exception:
            return ""
        try:
            client = Groq(api_key=os.getenv("GROQ_API_KEY"))
            resp = client.chat.completions.create(
                model=os.getenv("GROQ_MODEL", "llama-3.1-8b-instant"),
                messages=[
                    {"role": "system", "content": "You are a helpful assistant that answers using provided context."},
                    {"role": "user", "content": prompt},
                ],
                temperature=0.2,
            )
            return (resp.choices[0].message.content or "").strip()
        except Exception:
            return ""

    def _generate_openai(self, prompt: str) -> str:
        """Generate an answer using OpenAI."""

        try:
            from openai import OpenAI
        except Exception:
            return ""
        try:
            client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
            model = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
            resp = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": "You are a helpful assistant that answers using provided context."},
                    {"role": "user", "content": prompt},
                ],
                temperature=0.2,
            )
            return (resp.choices[0].message.content or "").strip()
        except Exception:
            return ""

    def generate_answer(self, context: str, question: str, sources: List[Dict[str, str]] | None = None) -> Dict[str, List[Dict[str, str]] | str]:
        prompt = (
            "Answer the question using only the provided context. "
            "If context is insufficient, say so briefly.\n\n"
            f"Context:\n{context}\n\nQuestion: {question}\nAnswer:"
        )
        try:
            backend = self._select_backend()
            if backend == "groq":
                answer = self._generate_groq(prompt) or ""
                if not answer:
                    backend = "openai" if os.getenv("OPENAI_API_KEY") else "hf"
            if backend == "openai":
                answer = self._generate_openai(prompt) or ""
                if not answer:
                    backend = "hf"
            if backend == "hf":
                qa_pipeline = self._get_pipeline()
                if qa_pipeline is None:
                    answer = self._fallback_answer(context, question)
                    return self._append_sources(answer, sources)
                result = qa_pipeline(prompt, max_new_tokens=180, do_sample=False)
                answer = result[0]["generated_text"].strip()
            if not answer:
                answer = self._fallback_answer(context, question)
            return self._append_sources(answer, sources)
        except Exception:
            answer = self._fallback_answer(context, question)
            return self._append_sources(answer, sources)

    def _append_sources(self, answer: str, sources: List[Dict[str, str]] | None) -> Dict[str, List[Dict[str, str]] | str]:
        formatted_sources = self._format_sources(sources or [])
        if formatted_sources:
            answer = f"{answer}\n\nSources:\n" + "\n".join(
                f"- {item['label']}" for item in formatted_sources
            )
        return {"answer": answer, "sources": formatted_sources}

    def _format_sources(self, sources: List[Dict[str, str]]) -> List[Dict[str, str]]:
        formatted: List[Dict[str, str]] = []
        seen = set()
        for source in sources:
            label_parts = []
            origin = source.get("source") or source.get("url") or source.get("filename") or "unknown"
            label_parts.append(str(origin))
            chunk_type = source.get("chunk_type")
            if chunk_type:
                label_parts.append(str(chunk_type))
            function_name = source.get("function_name")
            if function_name:
                label_parts.append(f"fn {function_name}")
            class_name = source.get("class_name")
            if class_name:
                label_parts.append(f"class {class_name}")
            page = source.get("page")
            if page is not None and page != "":
                label_parts.append(f"page {page}")
            row_range = source.get("row_range")
            if row_range:
                label_parts.append(f"rows {row_range}")
            label = " | ".join(label_parts)
            if label in seen:
                continue
            seen.add(label)
            entry: Dict[str, str] = {"label": label, "source": str(origin)}
            for key in ("chunk_type", "page", "function_name", "class_name", "row_range"):
                value = source.get(key)
                if value is not None and value != "":
                    entry[key] = str(value)
            formatted.append(entry)
        return formatted
