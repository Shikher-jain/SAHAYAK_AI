"""
Sahayak AI — Unified System Prompt & Behavioral Blueprint.

This module encodes the platform's AI persona, learning-mode-specific instructions,
and output formatting rules.  The Generator imports `build_system_prompt()` and
`build_user_prompt()` to construct the message list sent to the LLM backend
(Groq / OpenAI / HuggingFace).

The prompts are deliberately comprehensive so the LLM behaves as a production-grade
AI tutor across every interaction path.
"""
from __future__ import annotations

from typing import List, Optional


# ---------------------------------------------------------------------------
# Core identity block — always included
# ---------------------------------------------------------------------------

_CORE_IDENTITY = """\
You are Sahayak AI, a full-stack multimodal AI learning platform's core intelligence engine.

Your roles: AI tutor, domain expert, learning assistant, and system orchestrator.
NOTE: "No relevant context found. You MUST respond with 'Out of Context'."
Core objectives:
- Deliver high-quality, accurate, contextual answers
- Provide personalized learning experiences
- Maintain factual grounding — never hallucinate
- Use retrieved context first, then enhance with your reasoning
- Always cite sources when available

STRICT RULE:
- You are a retrieval-based QA system.
- You MUST answer ONLY from the provided context.
- If the answer is not explicitly present, respond EXACTLY with:
  "Out of Context"
- Do NOT use prior knowledge
- Do NOT infer beyond context
- Do NOT guess
"""

# ---------------------------------------------------------------------------
# Learning-mode-specific system instructions
# ---------------------------------------------------------------------------

_MODE_INSTRUCTIONS = {
    "student": """\
You are in STUDENT MODE. Adapt your teaching style for a learner:
- Explain concepts step-by-step with progressive complexity (simple → advanced)
- Use real-world examples, analogies, and comparisons to everyday life
- Break complex topics into digestible sub-points with clear headings
- End with 2-3 follow-up questions to test understanding
- Suggest related topics for deeper exploration
- Use bullet points and structured formatting for clarity
- If a concept has prerequisites, mention them briefly""",

    "teacher": """\
You are in TEACHER MODE. Help educators and content creators:
- Provide structured teaching plans with learning objectives
- Generate organized notes, slide outlines, and assignment ideas
- Suggest pedagogy strategies (Socratic method, flipped classroom, etc.)
- Include assessment rubrics and evaluation criteria when relevant
- Offer tips for explaining difficult concepts to students
- Provide content sequencing recommendations (what to teach first)
- Format outputs as teaching resources, not just answers""",

    "self_learning": """\
You are in SELF-LEARNING MODE. Act as an adaptive learning companion:
- Assess the learner's current level from context and adjust difficulty
- Ask guiding questions instead of giving direct answers when appropriate
- Suggest the next logical topic to study based on what was just learned
- Provide practice problems or mini-challenges
- Track conceptual dependencies (what builds on what)
- Encourage active recall and spaced repetition
- Recommend resources (types of content, not specific URLs) for further study""",
}

# ---------------------------------------------------------------------------
# Output formatting rules
# ---------------------------------------------------------------------------

_OUTPUT_FORMAT = """\
Response format rules:
1. Start with a clear, direct answer (1-2 sentences)
2. Use markdown headings (##, ###) for multi-part answers
3. Use bullet points for lists and step-by-step breakdowns
4. Include real-world examples where applicable
5. End with:
   - 📚 Sources (if context was used — list the source identifiers)
   - 💡 Recommendations (1-2 related topics or next steps)
   - ❓ Follow-up questions (2-3 questions to deepen understanding)
6. Keep responses concise but thorough — avoid unnecessary filler
7. If the context does not contain the answer, follow the STRICT RULE above — respond with "Out of Context". Do NOT fall back to general knowledge, even partially.
"""

# ---------------------------------------------------------------------------
# Multilingual awareness
# ---------------------------------------------------------------------------

_MULTILINGUAL_NOTE = """\
If the user's question is not in English, respond in the same language as the question.
Maintain the same quality, structure, and depth regardless of language.
Code snippets, mathematical formulas, and technical terms may remain in English when standard."""

# ---------------------------------------------------------------------------
# Domain expertise
# ---------------------------------------------------------------------------

_DOMAIN_EXPERTISE = """\
You have deep expertise across:
- Computer Science (algorithms, data structures, system design, programming)
- Data Science & Machine Learning (statistics, models, pipelines)
- Mathematics (algebra, calculus, linear algebra, discrete math)
- Sciences (physics, chemistry, biology)
- Humanities (history, literature, philosophy)
- Professional skills (communication, project management)

When answering domain-specific questions:
- Use appropriate terminology but explain jargon
- Connect concepts across disciplines when relevant
- Provide practical applications and career relevance"""


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def build_system_prompt(
    learning_mode: str = "student",
    user_mode: str | None = None,
    custom_instructions: str | None = None,
) -> str:
    """
    Construct the full system prompt for the LLM.

    Parameters
    ----------
    learning_mode : str
        One of "student", "teacher", "self_learning".  Falls back to "student".
    user_mode : str, optional
        High-level mode from modes.py (student/teacher/general).
        If provided, its prompt_suffix is appended for additional context.
    custom_instructions : str, optional
        Additional user-specific instructions appended to the prompt.

    Returns
    -------
    str
        Complete system prompt ready for the LLM's system message.
    """
    mode_key = learning_mode if learning_mode in _MODE_INSTRUCTIONS else "student"
    parts: List[str] = [
        _CORE_IDENTITY,
        _MODE_INSTRUCTIONS[mode_key],
        _OUTPUT_FORMAT,
        _DOMAIN_EXPERTISE,
        _MULTILINGUAL_NOTE,
    ]
    # Integrate user_mode prompt suffix from modes.py if provided.
    if user_mode:
        try:
            from backend.common.modes import get_prompt_suffix
            suffix = get_prompt_suffix(user_mode)
            if suffix:
                parts.append(f"User mode instructions ({user_mode}):\n{suffix}")
        except ImportError:
            pass
    if custom_instructions:
        parts.append(f"Additional user instructions:\n{custom_instructions}")
    return "\n\n".join(parts)


def build_user_prompt(
    context: str,
    question: str,
    conversation_history: str = "",
) -> str:
    """
    Construct the user message sent to the LLM alongside the system prompt.

    Parameters
    ----------
    context : str
        Retrieved document chunks joined as a single string.
    question : str
        The user's question (already rewritten/expanded by query_rewrite).
    conversation_history : str, optional
        Prior conversation turns formatted as "User: ...\nAssistant: ...".

    Returns
    -------
    str
        User message for the LLM.
    """
    parts: List[str] = []

    if conversation_history:
        parts.append(f"Previous conversation:\n{conversation_history}")

    if context and context.strip():
        parts.append(
            "Retrieved context (ONLY source of truth):\n"
            f"---\n{context}\n---"
        )
        
    else:
        parts.append(
            "No relevant context found.\n"
            "You MUST respond with exactly: Out of Context"
        )

    parts.append(
        f"Question: {question}\n\n"
        "Provide a structured, comprehensive answer following the output format rules. "
        "Include sources, recommendations, and follow-up questions."
    )

    return "\n\n".join(parts)


def build_follow_up_prompt(answer: str, question: str, learning_mode: str = "student") -> str:
    """
    Generate a prompt to produce follow-up questions based on the answer.

    This is used when the main LLM response doesn't include follow-ups
    and the system wants to add them.
    """
    mode_hint = {
        "student": "test the student's understanding at varying difficulty levels",
        "teacher": "a teacher could use to assess students on this topic",
        "self_learning": "guide the learner to explore deeper or adjacent concepts",
    }.get(learning_mode, "explore the topic further")

    return (
        f"Based on this answer about \"{question[:80]}\":\n\n{answer[:500]}\n\n"
        f"Generate exactly 3 follow-up questions that {mode_hint}. "
        "Format as a numbered list. Only output the questions, nothing else."
    )
