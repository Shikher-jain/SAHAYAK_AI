"""AI Counselor service — domain-specialist academic/career guidance using LLM."""
from __future__ import annotations

import os
from typing import Any, Dict, List

# System prompt for the counselor persona
_COUNSELOR_SYSTEM_PROMPT = """You are an expert academic and career counselor for Sahayak AI.
Your role is to help students and teachers with:
- Career path guidance (STEM, Arts, Commerce, etc.)
- Subject selection advice based on goals
- Course and resource recommendations
- Study strategies and time management
- Exam preparation tips
- Industry trends and job market insights

Be encouraging, specific, and actionable. Always suggest concrete next steps."""

_DOMAIN_PROMPTS = {
    "stem": "Focus on Science, Technology, Engineering, and Mathematics career paths.",
    "arts": "Focus on creative fields: design, writing, media, performing arts.",
    "commerce": "Focus on business, finance, accounting, and entrepreneurship.",
    "medical": "Focus on healthcare, medicine, pharmacy, and allied fields.",
    "law": "Focus on legal studies, judiciary, corporate law, and policy.",
    "general": "Provide broad academic and career guidance.",
}


class CounselorService:
    """AI-powered academic and career counselor."""

    def chat(self, message: str, domain: str = "general", history: str = "") -> Dict[str, Any]:
        """Generate a counseling response using the LLM."""
        domain_hint = _DOMAIN_PROMPTS.get(domain, _DOMAIN_PROMPTS["general"])
        full_context = ""
        if history:
            full_context = f"Previous conversation:\n{history}\n\n"
        prompt = (
            f"{_COUNSELOR_SYSTEM_PROMPT}\n{domain_hint}\n\n"
            f"{full_context}Student/Teacher says: {message}\n\nCounselor response:"
        )
        # Try Groq first, then OpenAI, then HF fallback
        answer = self._call_llm(prompt)
        return {
            "answer": answer,
            "domain": domain,
            "suggestions": self._extract_suggestions(answer),
        }

    def _call_llm(self, prompt: str) -> str:
        """Call the best available LLM backend."""
        # Try Groq
        if os.getenv("GROQ_API_KEY"):
            try:
                from groq import Groq
                client = Groq(api_key=os.getenv("GROQ_API_KEY"))
                resp = client.chat.completions.create(
                    model=os.getenv("GROQ_MODEL", "llama-3.1-8b-instant"),
                    messages=[
                        {"role": "system", "content": _COUNSELOR_SYSTEM_PROMPT},
                        {"role": "user", "content": prompt},
                    ],
                    temperature=0.4,
                    max_tokens=500,
                )
                return (resp.choices[0].message.content or "").strip()
            except Exception:
                pass
        # Try OpenAI
        if os.getenv("OPENAI_API_KEY"):
            try:
                from openai import OpenAI
                client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
                resp = client.chat.completions.create(
                    model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
                    messages=[
                        {"role": "system", "content": _COUNSELOR_SYSTEM_PROMPT},
                        {"role": "user", "content": prompt},
                    ],
                    temperature=0.4,
                    max_tokens=500,
                )
                return (resp.choices[0].message.content or "").strip()
            except Exception:
                pass
        # Fallback: static counseling advice
        return self._fallback_response(prompt)

    def _fallback_response(self, prompt: str) -> str:
        return (
            "Here are some general recommendations:\n"
            "1. Identify your strengths and interests through self-assessment.\n"
            "2. Research career paths in your domain of interest.\n"
            "3. Build foundational skills through structured courses.\n"
            "4. Work on projects and internships for practical experience.\n"
            "5. Network with professionals in your field.\n"
            "6. Stay updated with industry trends.\n\n"
            "Configure GROQ_API_KEY or OPENAI_API_KEY for personalized AI counseling."
        )

    def _extract_suggestions(self, answer: str) -> List[str]:
        """Extract actionable suggestions from the response."""
        suggestions = []
        for line in answer.split("\n"):
            line = line.strip()
            if line and (line[0].isdigit() or line.startswith("-") or line.startswith("*")):
                clean = line.lstrip("0123456789.-*) ").strip()
                if clean:
                    suggestions.append(clean)
        return suggestions[:5]

    def get_domain_suggestions(self, domain: str) -> List[Dict[str, str]]:
        """Return pre-built career suggestions for a domain."""
        domain_data = {
            "stem": [
                {"role": "Data Scientist", "skills": "Python, Statistics, ML", "demand": "High"},
                {"role": "Software Engineer", "skills": "DSA, System Design, Cloud", "demand": "Very High"},
                {"role": "AI/ML Engineer", "skills": "Deep Learning, NLP, Computer Vision", "demand": "Very High"},
                {"role": "DevOps Engineer", "skills": "Docker, Kubernetes, CI/CD", "demand": "High"},
            ],
            "commerce": [
                {"role": "Financial Analyst", "skills": "Excel, Financial Modeling, CFA", "demand": "High"},
                {"role": "Chartered Accountant", "skills": "Accounting, Tax, Audit", "demand": "High"},
                {"role": "Management Consultant", "skills": "Strategy, Analytics, MBA", "demand": "High"},
            ],
            "arts": [
                {"role": "UX Designer", "skills": "Figma, User Research, Prototyping", "demand": "High"},
                {"role": "Content Creator", "skills": "Writing, Video, SEO", "demand": "Growing"},
                {"role": "Digital Marketer", "skills": "SEO, Analytics, Social Media", "demand": "High"},
            ],
            "medical": [
                {"role": "Doctor (MBBS/MD)", "skills": "NEET, Clinical Skills", "demand": "Very High"},
                {"role": "Pharmacist", "skills": "Pharmacology, Drug Interactions", "demand": "High"},
                {"role": "Biotech Researcher", "skills": "Molecular Biology, Lab Skills", "demand": "Growing"},
            ],
        }
        return domain_data.get(domain, domain_data.get("stem", []))
