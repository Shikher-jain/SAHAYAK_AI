"""Help bot service — FAQ database, intent detection, guided navigation."""
from __future__ import annotations

from typing import Any, Dict, List

# FAQ database — covers platform features and navigation
_FAQ_DATABASE: List[Dict[str, Any]] = [
    {"intent": "upload", "keywords": ["upload", "ingest", "add file", "document", "pdf"], "question": "How do I upload documents?", "answer": "Use the Upload section in the sidebar. You can upload PDFs, images, audio, video, code files, CSV/Excel files, or paste a URL. Sahayak will automatically process and index your content for search and Q&A."},
    {"intent": "search", "keywords": ["search", "find", "query", "ask", "question"], "question": "How do I search or ask questions?", "answer": "Use the Search or Chat section. Type your question and Sahayak will use RAG (Retrieval-Augmented Generation) to find relevant content from your uploaded documents and generate an answer with source citations."},
    {"intent": "voice", "keywords": ["voice", "audio", "speak", "microphone", "speech"], "question": "How do I use voice features?", "answer": "The Voice section allows you to speak your question instead of typing. Sahayak transcribes your speech, processes the query through RAG, and can speak the answer back to you. Works in multiple languages."},
    {"intent": "quiz", "keywords": ["quiz", "test", "exam", "flashcard", "practice"], "question": "How do I take quizzes?", "answer": "Go to the Learn section and click 'Generate Quiz'. Sahayak creates interactive quizzes from your uploaded content. Track your scores over time in the quiz history."},
    {"intent": "roadmap", "keywords": ["roadmap", "path", "learning path", "curriculum", "plan"], "question": "How do learning roadmaps work?", "answer": "Roadmaps show you structured learning paths for subjects like Data Science, Web Dev, and ML/AI. Check off topics as you complete them and track your progress. Pre-built roadmaps are available, or create your own."},
    {"intent": "books", "keywords": ["book", "ncert", "textbook", "reading", "library"], "question": "How do I access online books?", "answer": "The Books section provides access to NCERT textbooks (classes 1-12) and other open educational resources. You can browse by subject and class, and ingest book content into your vector store for RAG queries."},
    {"intent": "counselor", "keywords": ["counselor", "career", "guidance", "advice", "advising"], "question": "How does the AI counselor work?", "answer": "The AI Counselor provides academic and career guidance. Ask about career paths, course recommendations, subject choices, and study strategies. It uses domain-specific knowledge to give personalized advice."},
    {"intent": "modes", "keywords": ["student", "teacher", "mode", "learning mode", "switch"], "question": "What are the learning modes?", "answer": "Sahayak has three modes: Student mode (guided learning, quizzes, progress tracking), Teacher mode (content creation, analytics, quiz builder), and Self-learning mode (adaptive paths based on your performance). Switch modes in your profile settings."},
    {"intent": "pricing", "keywords": ["price", "cost", "plan", "free", "pro", "enterprise", "pay"], "question": "What are the pricing plans?", "answer": "Free plan: 50 documents, 5 queries/day. Pro plan: ₹999/month for unlimited features. Enterprise: custom pricing for institutions. See the Pricing section for full details."},
    {"intent": "sync", "keywords": ["sync", "backup", "export", "import", "cloud"], "question": "How does cloud sync work?", "answer": "Use the Sync feature to export your data (documents, notes, progress) as JSON for backup. Import previously exported data to restore. Qdrant snapshots are also supported for vector store backup."},
    {"intent": "multilingual", "keywords": ["language", "hindi", "translate", "multilingual", "spanish"], "question": "Does Sahayak support multiple languages?", "answer": "Yes! Sahayak supports Hindi, English, Spanish, French, and German. Change your language in the sidebar — your queries and answers will be automatically translated."},
    {"intent": "notes", "keywords": ["notes", "note", "summary", "summarize", "generate notes"], "question": "How do I generate notes?", "answer": "After uploading a document, use the Notes feature in the Document section. Sahayak generates structured notes with key concepts, summaries, and important points. Export notes in Markdown format."},
]


class HelpBot:
    """FAQ-based help bot with keyword intent matching."""

    def __init__(self) -> None:
        self.faq_database = _FAQ_DATABASE

    def answer(self, question: str) -> Dict[str, Any]:
        """Find the best matching FAQ entry for the question."""
        question_lower = question.lower()
        best_match: Dict[str, Any] | None = None
        best_score = 0

        for entry in self.faq_database:
            score = sum(1 for kw in entry["keywords"] if kw in question_lower)
            if score > best_score:
                best_score = score
                best_match = entry

        if best_match and best_score > 0:
            return {
                "answer": best_match["answer"],
                "intent": best_match["intent"],
                "related_question": best_match["question"],
                "confidence": min(best_score / 3.0, 1.0),
            }
        return {
            "answer": (
                "I couldn't find a specific answer for that. Try asking about: "
                "uploading documents, searching/asking questions, voice features, "
                "quizzes, roadmaps, books, counseling, pricing, or cloud sync. "
                "You can also contact support@sahayak.ai for further help."
            ),
            "intent": "unknown",
            "related_question": None,
            "confidence": 0.0,
        }

    def get_all_faqs(self) -> List[Dict[str, str]]:
        return [{"question": e["question"], "answer": e["answer"]} for e in self.faq_database]
