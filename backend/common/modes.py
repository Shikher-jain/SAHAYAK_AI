"""
Sahayak AI — User Mode Configuration.

Three modes that change how the AI responds and what features are available:
  * STUDENT  — simplified explanations, guided learning, quizzes
  * TEACHER  — detailed explanations, curriculum context, content creation
  * GENERAL  — default behavior, no special adjustments

Each mode provides:
  - prompt_suffix : appended to the system prompt for LLM responses
  - ui_hints     : dict of UI labels/hints returned in API responses
  - features     : set of feature flags controlling access levels
"""
from __future__ import annotations

from typing import Dict, Set
from dotenv import load_dotenv
load_dotenv()


# ---------------------------------------------------------------------------
# Mode constants
# ---------------------------------------------------------------------------

STUDENT = "student"
TEACHER = "teacher"
GENERAL = "general"

ALL_MODES = {STUDENT, TEACHER, GENERAL}


# ---------------------------------------------------------------------------
# Per-mode configuration
# ---------------------------------------------------------------------------

_MODE_PROMPTS: Dict[str, str] = {
    STUDENT: (
        "You are helping a student. Use simple language, step-by-step explanations, "
        "real-world analogies, and examples. Break complex topics into small parts. "
        "End with a quick check-for-understanding question."
    ),
    TEACHER: (
        "You are assisting a teacher or educator. Provide detailed, structured "
        "explanations with curriculum context. Include teaching strategies, assessment "
        "ideas, and content organization tips. Use academic terminology."
    ),
    GENERAL: "",  # No special prompt adjustments
}

_MODE_UI_HINTS: Dict[str, Dict[str, str]] = {
    STUDENT: {
        "welcome": "Welcome back! Ready to learn something new today?",
        "upload_hint": "Upload your study material and I'll help you understand it.",
        "search_hint": "Ask me anything — I'll explain it step by step.",
        "mode_label": "Student Mode",
        "mode_icon": "🎓",
    },
    TEACHER: {
        "welcome": "Welcome! Let's prepare great learning content together.",
        "upload_hint": "Upload material to generate teaching resources.",
        "search_hint": "Ask for detailed explanations, lesson plans, or assessments.",
        "mode_label": "Teacher Mode",
        "mode_icon": "👩‍🏫",
    },
    GENERAL: {
        "welcome": "Welcome to Sahayak AI.",
        "upload_hint": "Upload documents to index and search.",
        "search_hint": "Ask a question about your documents.",
        "mode_label": "General Mode",
        "mode_icon": "💬",
    },
}

_MODE_FEATURES: Dict[str, Set[str]] = {
    STUDENT: {
        "quiz", "flashcards", "notes", "bookmarks", "roadmaps",
        "counselor", "progress_tracking", "simplified_explanations",
    },
    TEACHER: {
        "quiz_builder", "content_creation", "analytics", "notes",
        "bookmarks", "roadmaps", "detailed_explanations", "assessment_gen",
    },
    GENERAL: {
        "search", "upload", "notes", "bookmarks",
    },
}


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def get_prompt_suffix(mode: str) -> str:
    """Return the prompt suffix for the given mode (empty string for GENERAL)."""
    return _MODE_PROMPTS.get(mode, "")


def get_ui_hints(mode: str) -> Dict[str, str]:
    """Return UI hint strings for the given mode."""
    return dict(_MODE_UI_HINTS.get(mode, _MODE_UI_HINTS[GENERAL]))


def get_features(mode: str) -> Set[str]:
    """Return the set of feature flags available in this mode."""
    return set(_MODE_FEATURES.get(mode, _MODE_FEATURES[GENERAL]))


def has_feature(mode: str, feature: str) -> bool:
    """Check if a specific feature is available in the given mode."""
    return feature in get_features(mode)


def resolve_mode(raw: str | None) -> str:
    """Normalize a raw mode string to one of the valid constants."""
    if not raw:
        return GENERAL
    normalized = raw.strip().lower()
    # Map common aliases
    aliases = {
        "self_learning": STUDENT,
        "self-learning": STUDENT,
        "learner": STUDENT,
        "educator": TEACHER,
        "instructor": TEACHER,
        "default": GENERAL,
    }
    resolved = aliases.get(normalized, normalized)
    return resolved if resolved in ALL_MODES else GENERAL
