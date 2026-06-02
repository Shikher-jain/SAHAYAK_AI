"""
Sahayak AI — Answer Generator.

Generates structured, learning-mode-aware answers using retrieved context.
Backend priority: Groq → OpenAI → HuggingFace fallback.

Every response includes:
- Structured answer with headings, bullets, examples
- Source citations (from retrieval metadata)
- Recommendations (related topics / next steps)
- Follow-up questions (to deepen understanding)
"""
from __future__ import annotations

import os
import re
from typing import Any, Dict, List, Optional

from backend.rag.system_prompt import build_system_prompt, build_user_prompt, build_follow_up_prompt


class Generator:
    def __init__(self) -> None:
        self._hf_model = None  # Seq2SeqGenerator from hf_models
        self._backend = None

    # ------------------------------------------------------------------
    # Backend selection
    # ------------------------------------------------------------------

    def _get_hf_model(self):
        """Load Flan-T5 via the shared singleton (AutoTokenizer + AutoModelForSeq2SeqLM)."""
        if self._hf_model is None:
            try:
                from backend.common.hf_models import HFModels
                self._hf_model = HFModels.get_text_generation()
            except Exception as e:
                print(f"Error loading HuggingFace model via HFModels: {e}")
                self._hf_model = None
        return self._hf_model

    def _select_backend(self) -> str:
        """Select LLM backend by priority: Groq → OpenAI → HuggingFace."""
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

    # ------------------------------------------------------------------
    # Fallback answer (no LLM)
    # ------------------------------------------------------------------

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

    # ------------------------------------------------------------------
    # LLM calls — now using the unified system prompt
    # ------------------------------------------------------------------

    def _generate_groq(self, system_prompt: str, user_prompt: str) -> str:
        """Generate an answer using Groq chat completions with rich system prompt."""
        try:
            from groq import Groq
        except Exception:
            return ""
        try:
            client = Groq(api_key=os.getenv("GROQ_API_KEY"))
            resp = client.chat.completions.create(
                model=os.getenv("GROQ_MODEL", "llama-3.1-8b-instant"),
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=0.3,
                max_tokens=1500,
            )
            return (resp.choices[0].message.content or "").strip()
        except Exception:
            return ""

    def _generate_openai(self, system_prompt: str, user_prompt: str) -> str:
        """Generate an answer using OpenAI with rich system prompt."""
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
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=0.3,
                max_tokens=1500,
            )
            return (resp.choices[0].message.content or "").strip()
        except Exception:
            return ""

    def _generate_follow_ups_llm(self, prompt: str) -> str:
        """Lightweight LLM call just for follow-up question generation."""
        backend = self._select_backend()
        lightweight_system = (
            "You are Sahayak AI. Generate exactly 3 follow-up questions. "
            "Output only the numbered list, nothing else."
        )
        try:
            if backend == "groq":
                from groq import Groq
                client = Groq(api_key=os.getenv("GROQ_API_KEY"))
                resp = client.chat.completions.create(
                    model=os.getenv("GROQ_MODEL", "llama-3.1-8b-instant"),
                    messages=[
                        {"role": "system", "content": lightweight_system},
                        {"role": "user", "content": prompt},
                    ],
                    temperature=0.5,
                    max_tokens=200,
                )
                return (resp.choices[0].message.content or "").strip()
            if backend == "openai":
                from openai import OpenAI
                client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
                model = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
                resp = client.chat.completions.create(
                    model=model,
                    messages=[
                        {"role": "system", "content": lightweight_system},
                        {"role": "user", "content": prompt},
                    ],
                    temperature=0.5,
                    max_tokens=200,
                )
                return (resp.choices[0].message.content or "").strip()
        except Exception:
            pass
        return ""

    # ------------------------------------------------------------------
    # Response parsing — extract structured sections
    # ------------------------------------------------------------------

    @staticmethod
    def _extract_follow_ups(raw: str) -> List[str]:
        """Parse numbered follow-up questions from LLM output."""
        if not raw:
            return []
        lines = raw.strip().split("\n")
        questions: List[str] = []
        for line in lines:
            cleaned = re.sub(r"^\d+[\.\)]\s*", "", line.strip())
            cleaned = re.sub(r"^[-*•]\s*", "", cleaned)
            if cleaned and len(cleaned) > 5:
                questions.append(cleaned)
        return questions[:5]

    @staticmethod
    def _extract_section(answer: str, header: str) -> str:
        """Extract a section (between a header and the next header or end) from LLM output."""
        pattern = rf"(?:^|\n)\s*{re.escape(header)}\s*[:\n]?\s*(.*?)(?=\n\s*(?:📚|💡|❓|Sources|Recommendations|Follow-up|##|\Z))"
        match = re.search(pattern, answer, re.IGNORECASE | re.DOTALL)
        return match.group(1).strip() if match else ""

    # ------------------------------------------------------------------
    # Main generation entry point
    # ------------------------------------------------------------------

    def generate_answer(
        self,
        context: str,
        question: str,
        sources: Optional[List[Dict[str, str]]] = None,
        learning_mode: str = "student",
        conversation_history: str = "",
        user_mode: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Generate a structured, learning-mode-aware answer.

        Parameters
        ----------
        user_mode : str, optional
            High-level mode from modes.py (student/teacher/general).

        Returns
        -------
        dict with keys:
            answer      : str  — the main answer text
            sources     : list — formatted citation dicts
            recommendations : list — follow-up recommendation strings
            follow_ups  : list — follow-up question strings
        """
        system_prompt = build_system_prompt(
            learning_mode=learning_mode,
            user_mode=user_mode,
        )
        user_prompt = build_user_prompt(
            context=context,
            question=question,
            conversation_history=conversation_history,
        )

        try:
            backend = self._select_backend()
            answer = ""
            if backend == "groq":
                answer = self._generate_groq(system_prompt, user_prompt) or ""
                if not answer:
                    backend = "openai" if os.getenv("OPENAI_API_KEY") else "hf"
            if not answer and backend == "openai":
                answer = self._generate_openai(system_prompt, user_prompt) or ""
                if not answer:
                    backend = "hf"
            if not answer and backend == "hf":
                hf_model = self._get_hf_model()
                if hf_model is None:
                    answer = self._fallback_answer(context, question)
                else:
                    # Seq2SeqGenerator wrapper — pipeline-compatible interface
                    simple_prompt = (
                        f"Context:\n{context[:1500]}\n\n"
                        f"Question: {question}\n"
                        "Answer in a clear, structured way with examples:"
                    )
                    result = hf_model(simple_prompt, max_new_tokens=300, do_sample=False)
                    answer = result[0]["generated_text"].strip()
            if not answer:
                answer = self._fallback_answer(context, question)
        except Exception:
            answer = self._fallback_answer(context, question)

        # Generate follow-up questions if the answer doesn't already contain them
        follow_ups = self._extract_follow_ups(self._extract_section(answer, "Follow-up"))
        if not follow_ups:
            follow_ups = self._extract_follow_ups(self._extract_section(answer, "❓"))
        if not follow_ups and len(answer) > 50:
            follow_up_prompt = build_follow_up_prompt(answer, question, learning_mode)
            raw_follows = self._generate_follow_ups_llm(follow_up_prompt)
            follow_ups = self._extract_follow_ups(raw_follows)

        # Extract recommendations from the answer
        recommendations_text = self._extract_section(answer, "Recommendations")
        if not recommendations_text:
            recommendations_text = self._extract_section(answer, "💡")
        rec_items = [
            line.strip().lstrip("-•* ").strip()
            for line in recommendations_text.split("\n")
            if line.strip() and len(line.strip()) > 3
        ] if recommendations_text else []

        formatted_sources = self._format_sources(sources or [])
        return {
            "answer": answer,
            "sources": formatted_sources,
            "recommendations": rec_items,
            "follow_ups": follow_ups,
        }

    # ------------------------------------------------------------------
    # Source formatting (unchanged from original)
    # ------------------------------------------------------------------

    def _append_sources(
        self, answer: str, sources: List[Dict[str, str]] | None
    ) -> Dict[str, List[Dict[str, str]] | str]:
        formatted_sources = self._format_sources(sources or [])
        if formatted_sources:
            answer = f"{answer}\n\n📚 Sources:\n" + "\n".join(
                f"- {item['label']}" for item in formatted_sources
            )
        return {"answer": answer, "sources": formatted_sources}

    def _format_sources(self, sources: List[Dict[str, str]]) -> List[Dict[str, str]]:
        formatted: List[Dict[str, str]] = []
        seen: set = set()
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
