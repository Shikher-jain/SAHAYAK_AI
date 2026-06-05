"""Sahayak AI — Unified Learning Platform Frontend."""
from __future__ import annotations

import json
import os
import uuid
from typing import Any, Dict, List, Optional, Tuple

import requests
import streamlit as st

# --- Configuration ---
PAGE_TITLE = "Sahayak - AI Learning Platform"
PAGE_ICON = "image.png"
BACKEND_URL = os.getenv("BACKEND_URL", "http://127.0.0.1:8000")
ENV_API_KEY = os.getenv("SAHAYAK_API_KEY", "").strip()

INGESTION_ROUTES: Dict[str, Tuple[str, str]] = {
    ".pdf": ("/ingest/pdf", "application/pdf"),
    ".png": ("/ingest/image", "image/png"),
    ".jpg": ("/ingest/image", "image/jpeg"),
    ".jpeg": ("/ingest/image", "image/jpeg"),
    ".wav": ("/ingest/audio", "audio/wav"),
    ".mp3": ("/ingest/audio", "audio/mpeg"),
    ".mp4": ("/ingest/video", "video/mp4"),
    ".mov": ("/ingest/video", "video/quicktime"),
    ".avi": ("/ingest/video", "video/x-msvideo"),
    ".txt": ("/ingest/text", "text/plain"),
    ".py": ("/ingest/code", "text/x-python"),
    ".js": ("/ingest/code", "application/javascript"),
    ".ts": ("/ingest/code", "application/typescript"),
    ".cpp": ("/ingest/code", "text/x-c++src"),
    ".java": ("/ingest/code", "text/x-java-source"),
    ".go": ("/ingest/code", "text/plain"),
    ".csv": ("/ingest/csv", "text/csv"),
    ".xlsx": ("/ingest/csv", "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"),
    ".xls": ("/ingest/csv", "application/vnd.ms-excel"),
}

# Multilingual UI labels (TASK 10)
_UI_LABELS = {
    "en": {"dashboard": "Dashboard", "learn": "Learn", "upload": "Upload", "search": "Search", "chat": "Chat", "roadmaps": "Roadmaps", "books": "Books", "pricing": "Pricing", "profile": "Profile", "counselor": "Counselor", "quiz": "Quiz", "help": "Help", "stories": "Stories", "knowledge": "Knowledge Graph", "progress": "Progress", "sync": "Sync"},
    "hi": {"dashboard": "डैशबोर्ड", "learn": "सीखें", "upload": "अपलोड", "search": "खोजें", "chat": "चैट", "roadmaps": "रोडमैप", "books": "किताबें", "pricing": "मूल्य", "profile": "प्रोफ़ाइल", "counselor": "परामर्शदाता", "quiz": "क्विज़", "help": "सहायता", "stories": "कहानियाँ", "knowledge": "ज्ञान ग्राफ़", "progress": "प्रगति", "sync": "सिंक"},
    "es": {"dashboard": "Panel", "learn": "Aprender", "upload": "Subir", "search": "Buscar", "chat": "Chat", "roadmaps": "Rutas", "books": "Libros", "pricing": "Precios", "profile": "Perfil", "counselor": "Consejero", "quiz": "Quiz", "help": "Ayuda", "stories": "Historias", "knowledge": "Grafo", "progress": "Progreso", "sync": "Sinc"},
    "fr": {"dashboard": "Tableau", "learn": "Apprendre", "upload": "Télécharger", "search": "Chercher", "chat": "Chat", "roadmaps": "Routes", "books": "Livres", "pricing": "Tarifs", "profile": "Profil", "counselor": "Conseiller", "quiz": "Quiz", "help": "Aide", "stories": "Histoires", "knowledge": "Graphe", "progress": "Progrès", "sync": "Sync"},
    "de": {"dashboard": "Dashboard", "learn": "Lernen", "upload": "Hochladen", "search": "Suchen", "chat": "Chat", "roadmaps": "Roadmaps", "books": "Bücher", "pricing": "Preise", "profile": "Profil", "counselor": "Berater", "quiz": "Quiz", "help": "Hilfe", "stories": "Geschichten", "knowledge": "Graph", "progress": "Fortschritt", "sync": "Sync"},
}


# --- Backend communication ---

def _init_state():
    if "backend_url" not in st.session_state:
        st.session_state.backend_url = BACKEND_URL
    if "rag_session_id" not in st.session_state:
        st.session_state.rag_session_id = str(uuid.uuid4())
    if "language" not in st.session_state:
        st.session_state.language = "en"
    if "page" not in st.session_state:
        st.session_state.page = "Dashboard"
    if "learning_mode" not in st.session_state:
        st.session_state.learning_mode = "student"
    if "user_mode" not in st.session_state:
        st.session_state.user_mode = "general"
    if "theme" not in st.session_state:
        st.session_state.theme = "dark"
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    try:
        secret_key = st.secrets.get("SAHAYAK_API_KEY", "")
    except Exception:
        secret_key = ""
    if "api_key" not in st.session_state:
        configured = (secret_key or ENV_API_KEY).strip()
        if configured:
            st.session_state.api_key = configured


def _get_api_key() -> str:
    return st.session_state.get("api_key") or ENV_API_KEY


def _get_backend_url() -> str:
    return st.session_state.get("backend_url", BACKEND_URL)


def _get_auth_headers() -> Dict[str, str]:
    headers: Dict[str, str] = {}
    api_key = _get_api_key()
    if api_key:
        headers["X-API-Key"] = api_key
    token = st.session_state.get("auth_token")
    if token:
        headers["Authorization"] = f"Bearer {token}"
    return headers


def _call_backend(method: str, path: str, **kwargs) -> Tuple[bool, Optional[Dict], str]:
    url = f"{_get_backend_url()}{path}"
    headers = dict(kwargs.pop("headers", {}) or {})
    headers.update(_get_auth_headers())
    try:
        resp = requests.request(method.upper(), url, timeout=kwargs.pop("timeout", 60), headers=headers, **kwargs)
    except requests.RequestException as exc:
        return False, None, str(exc)
    if resp.status_code >= 400:
        try:
            detail = resp.json()
        except ValueError:
            detail = resp.text
        return False, detail if isinstance(detail, dict) else None, f"HTTP {resp.status_code}: {detail}"
    try:
        payload = resp.json()
    except ValueError:
        payload = {"raw": resp.text}
    return True, payload, "ok"


def _ui(key: str) -> str:
    lang = st.session_state.get("language", "en")
    return _UI_LABELS.get(lang, _UI_LABELS["en"]).get(key, _UI_LABELS["en"].get(key, key))


# --- Theme system ---

_LIGHT_THEME_CSS = """
<style>
.stApp {
    background-color: #f5f5f5;
    color: #333333;
}
[data-testid="stSidebar"] {
    background-color: #ffffff;
    border-right: 1px solid #ddd;
}
[data-testid="stSidebar"] .stMarkdown {
    color: #333333;
}
.stButton > button {
    background-color: #4a90d9;
    color: white;
    border: none;
}
.stTextInput > div > div > input {
    background-color: #ffffff;
    color: #333333;
}
</style>
"""

_DARK_THEME_CSS = """
<style>
.stApp {
    background-color: #0e1117;
    color: #fafafa;
}
[data-testid="stSidebar"] {
    background-color: #1a1d23;
    border-right: 1px solid #333;
}
</style>
"""


def _apply_theme():
    """Inject custom CSS based on the selected theme."""
    theme = st.session_state.get("theme", "dark")
    css = _LIGHT_THEME_CSS if theme == "light" else _DARK_THEME_CSS
    st.markdown(css, unsafe_allow_html=True)


# --- Page renderers ---

def page_dashboard():
    """Home / Dashboard — hero section, stats, quick actions, recent activity."""
    # Hero section
    st.markdown(
        '<div style="text-align:center;padding:1.5rem 0 0.5rem 0;">'
        '<div style="font-size:2.6rem;font-weight:800;">🤖 Sahayak AI</div>'
        '<div style="font-size:1.2rem;opacity:0.85;">Your AI Learning Assistant</div>'
        '</div>',
        unsafe_allow_html=True,
    )
    st.caption("Ingest any document, ask questions in your language, get structured answers with sources.")
    # Live stats from backend
    ok, stats, _ = _call_backend("get", "/stats/dashboard")
    if ok and stats:
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("📄 Documents Indexed", stats.get("documents", {}).get("total_indexed", 0))
        c2.metric("❓ Queries Answered", stats.get("queries", {}).get("total", 0))
        c3.metric("👥 Users", stats.get("users", {}).get("students", 0) + stats.get("users", {}).get("teachers", 0))
        c4.metric("📚 Courses", stats.get("courses", 0))
    else:
        c1, c2 = st.columns(2)
        c1.info("Connect to backend to see live statistics.")
        c2.info(f"Session: {st.session_state.rag_session_id[:8]}…")
    st.markdown("---")
    # Quick actions
    st.subheader("⚡ Quick Actions")
    c1, c2, c3, c4 = st.columns(4)

    if c1.button("📤 Upload Document", use_container_width=True):
        st.session_state["page"] = "Upload"
        st.session_state["nav_radio"] = "Upload"
        st.rerun()
    if c2.button("💬 Ask a Question", use_container_width=True):
        st.session_state["page"] = "Search"
        st.session_state["nav_radio"] = "Search"
        st.rerun()
    if c3.button("📝 Take a Quiz", use_container_width=True):
        st.session_state["page"] = "Quiz"
        st.session_state["nav_radio"] = "Quiz"
        st.rerun()
    if c4.button("🗺️ View Roadmaps", use_container_width=True):
        st.session_state["page"] = "Roadmaps"
        st.session_state["nav_radio"] = "Roadmaps"
        st.rerun()
    
    # Recent activity / stories
    st.markdown("---")
    st.subheader("💬 What Our Users Say")
    ok, stories, _ = _call_backend("get", "/stories?limit=3")
    if ok and stories:
        for s in stories:
            with st.expander(f"{s.get('title', '')} — {s.get('username', '')} ({'⭐' * s.get('rating', 5)})"):
                st.write(s.get("content", ""))


def page_upload():
    st.header(f"{_ui('upload')} Documents")
    upload_type = st.radio("Source", ["File", "URL", "Raw Text"], horizontal=True)
    ingested_name = None
    ingested_doc_id = None
    if upload_type == "File":
        uploaded = st.file_uploader(
            "Drop file here",
            type=list(k.lstrip(".") for k in INGESTION_ROUTES.keys()),
        )
        if uploaded and st.button("Ingest File"):
            _, ext = os.path.splitext(uploaded.name)
            route = INGESTION_ROUTES.get(ext.lower())
            if route:
                endpoint, mime_type = route
                uploaded.seek(0)
                files = {"file": (uploaded.name, uploaded, mime_type)}
                ok, payload, detail = _call_backend("post", endpoint, files=files)
                if ok:
                    st.success(f"✅ Ingested {uploaded.name}")
                    ingested_name = uploaded.name
                    ingested_doc_id = None
                    if isinstance(payload, dict):
                        ingested_doc_id = payload.get("document_id") or payload.get("id") or uploaded.name
                    # Show auto-tags if available
                    tags = []
                    if isinstance(payload, dict):
                        for rec in (payload.get("records") or []):
                            meta = rec.get("metadata") or {}
                            tags.extend(meta.get("tags", []))
                    if tags:
                        st.markdown(f"🏷️ **Auto-tags:** {', '.join(set(tags[:10]))}")
                else:
                    st.error(f"Failed: {detail}")
    elif upload_type == "URL":
        url_text = st.text_input("Paste URL")
        if url_text and st.button("Ingest URL"):
            ok, payload, detail = _call_backend("post", "/ingest/url", data={"url": url_text})
            if ok:
                st.success("✅ URL processed")
                ingested_name = url_text
                ingested_doc_id = url_text
            else:
                st.error(f"Failed: {detail}")
    else:
        raw_text = st.text_area("Paste text", height=200)
        if raw_text and st.button("Ingest Text"):
            ok, payload, detail = _call_backend("post", "/ingest/text", data={"text": raw_text})
            if ok:
                st.success("✅ Text ingested")
                ingested_name = "raw-text"
                ingested_doc_id = "raw-text"
            else:
                st.error(f"Failed: {detail}")
    # Post-upload actions
    if ingested_doc_id:
        st.markdown("---")
        st.subheader("Quick Actions")
        col1, col2, col3, col4 = st.columns(4)
        if col1.button("📝 Summarize"):
            with st.spinner("Summarizing..."):
                ok2, summ, _ = _call_backend("post", "/document/summarize", json={"document_id": ingested_doc_id})
            if ok2 and summ:
                st.markdown(f"**Summary:** {summ.get('summary', '')}")
                kps = summ.get("key_points", [])
                if kps:
                    st.markdown("**Key Points:**")
                    for kp in kps[:5]:
                        st.markdown(f"- {kp}")
        if col2.button("📝 Generate Notes"):
            with st.spinner("Generating notes..."):
                ok3, notes, _ = _call_backend("post", "/document/notes", params={"document_id": ingested_doc_id})
            if ok3 and notes:
                st.markdown(notes.get("notes", "No notes generated"))
        if col3.button("❓ Ask Questions"):
            st.session_state.page = _ui("chat")
            st.rerun()
        if col4.button("📖 Explain"):
            st.info("Select a text chunk from the document and use the /document/explain endpoint.")


def page_search():
    st.header(f"{_ui('search')} & {_ui('chat')}")
    # Learning mode + user mode selectors
    col_m1, col_m2 = st.columns(2)
    with col_m1:
        mode = st.radio("Learning Mode", ["student", "teacher", "self_learning"],
                        format_func=lambda m: {"student": "🎓 Student", "teacher": "👩‍🏫 Teacher", "self_learning": "📖 Self-Learning"}.get(m, m),
                        horizontal=True, key="search_mode")
        if mode != st.session_state.learning_mode:
            st.session_state.learning_mode = mode
    with col_m2:
        umode = st.radio("User Mode", ["student", "teacher", "general"],
                         format_func=lambda m: {"student": "🎓 Student", "teacher": "👩‍🏫 Teacher", "general": "💬 General"}.get(m, m),
                         horizontal=True, key="search_user_mode")
        if umode != st.session_state.user_mode:
            st.session_state.user_mode = umode
    question = st.text_area("Ask a question about your documents", height=100)
    col1, col2 = st.columns([3, 1])
    top_k = col2.slider("Context chunks", 1, 10, 5)
    if col1.button("Get Answer") and question.strip():
        session_id = st.session_state.rag_session_id
        with st.spinner("Querying RAG..."):
            ok, payload, detail = _call_backend("post", "/search/rag", data={
                "query": question,
                "top_k": top_k,
                "session_id": session_id,
                "learning_mode": st.session_state.learning_mode,
                "user_mode": st.session_state.user_mode,
            })
        if ok and payload:
            # Store in chat history
            st.session_state.chat_history.append({"role": "user", "content": question})
            answer = payload.get("answer", "No answer")
            st.session_state.chat_history.append({"role": "assistant", "content": answer})
            # Main answer
            st.markdown(answer)
            # Source citations
            sources = payload.get("sources") or []
            if sources:
                with st.expander("📚 Sources"):
                    for idx, src in enumerate(sources, 1):
                        label = src.get("label") or src.get("source", "unknown") if isinstance(src, dict) else str(src)
                        st.markdown(f"**{idx}.** {label}")
            # Recommendations
            recs = payload.get("recommendations") or []
            if recs:
                st.markdown("---")
                st.markdown("**💡 Recommendations**")
                for r in recs[:5]:
                    if isinstance(r, dict):
                        st.caption(r.get("content", "")[:120] if r.get("content") else r.get("label", str(r)))
                    else:
                        st.caption(str(r))
            # Follow-up questions
            follow_ups = payload.get("follow_ups") or []
            if follow_ups:
                st.markdown("---")
                st.markdown("**❓ Follow-up Questions**")
                for idx, q in enumerate(follow_ups, 1):
                    if st.button(f"{idx}. {q}", key=f"fu_{idx}_{hash(q) % 10000}"):
                        st.session_state._auto_question = q
                        st.rerun()
            # Auto-submit follow-up question
            if "_auto_question" in st.session_state:
                aq = st.session_state.pop("_auto_question")
                st.info(f"Follow-up: **{aq}**")
                with st.spinner("Thinking..."):
                    ok2, p2, d2 = _call_backend("post", "/search/rag", data={
                        "query": aq, "top_k": top_k,
                        "session_id": session_id,
                        "learning_mode": st.session_state.learning_mode,
                        "user_mode": st.session_state.user_mode,
                    })
                if ok2 and p2:
                    st.session_state.chat_history.append({"role": "user", "content": aq})
                    st.session_state.chat_history.append({"role": "assistant", "content": p2.get("answer", "")})
                    st.markdown(p2.get("answer", "No answer"))
        else:
            st.error(f"Error: {detail}")
    # Chat history
    if st.session_state.chat_history:
        st.markdown("---")
        with st.expander(f"📝 Chat History ({len(st.session_state.chat_history)} messages)"):
            for msg in reversed(st.session_state.chat_history[-20:]):
                icon = "👤" if msg["role"] == "user" else "🤖"
                st.markdown(f"{icon} **{msg['role'].title()}:** {msg['content'][:200]}{'...' if len(msg['content']) > 200 else ''}")
            if st.button("Clear History"):
                st.session_state.chat_history = []
                st.rerun()


def page_learn():
    st.header(f"{_ui('learn')}")
    mode = st.radio("Learning Mode", ["Student", "Teacher", "Self-Learning"], horizontal=True)
    st.info(f"Active mode: **{mode}** — content adapts to your learning style.")
    tab_quiz, tab_notes, tab_bookmarks = st.tabs(["Quiz", "Notes", "Bookmarks"])
    with tab_quiz:
        topic = st.text_input("Quiz topic")
        num_q = st.slider("Number of questions", 3, 10, 5)
        if st.button("Generate Quiz") and topic:
            ok, payload, _ = _call_backend("post", "/quiz/generate", json={"topic": topic, "num_questions": num_q})
            if ok and payload:
                st.session_state.current_quiz = payload
                st.rerun()
        if "current_quiz" in st.session_state:
            quiz = st.session_state.current_quiz
            st.subheader(f"Quiz: {quiz.get('topic', '')}")
            answers = []
            for idx, q in enumerate(quiz.get("questions", [])):
                st.markdown(f"**Q{idx+1}.** {q.get('question', '')}")
                ans = st.radio(f"Answer {idx+1}", q.get("options", []), key=f"quiz_q_{idx}")
                answers.append(q.get("options", []).index(ans) if ans in q.get("options", []) else -1)
            if st.button("Submit Quiz"):
                ok, result, _ = _call_backend("post", "/quiz/answer", json={
                    "topic": quiz.get("topic", ""), "questions": quiz.get("questions", []), "answers": answers,
                })
                if ok and result:
                    st.success(f"Score: {result.get('correct', 0)}/{result.get('total', 0)} ({result.get('score', 0)*100:.0f}%)")
                    for r in result.get("results", []):
                        icon = "✅" if r.get("correct") else "❌"
                        st.markdown(f"{icon} {r.get('question', '')} — {r.get('explanation', '')}")
    with tab_notes:
        st.markdown("### Your Learning Notes")
        title = st.text_input("Note title")
        content = st.text_area("Note content", height=150)
        if st.button("Save Note") and title:
            ok, _, _ = _call_backend("post", "/learning/notes", json={"title": title, "content": content})
            st.success("Note saved") if ok else st.error("Failed to save")
        ok, notes, _ = _call_backend("get", "/learning/notes")
        if ok and notes:
            for n in notes:
                with st.expander(n.get("title", "")):
                    st.write(n.get("content", ""))
    with tab_bookmarks:
        doc_id = st.text_input("Document ID to bookmark")
        bm_title = st.text_input("Bookmark title")
        if st.button("Add Bookmark") and doc_id:
            ok, _, _ = _call_backend("post", "/learning/bookmarks", json={"document_id": doc_id, "title": bm_title})
            st.success("Bookmarked") if ok else st.error("Failed")
        ok, bms, _ = _call_backend("get", "/learning/bookmarks")
        if ok and bms:
            for b in bms:
                st.markdown(f"**{b.get('title', '')}** — {b.get('document_id', '')}")


def page_roadmaps():
    st.header(f"{_ui('roadmaps')}")
    ok, roadmaps, _ = _call_backend("get", "/roadmaps")
    if ok and roadmaps:
        for rm in roadmaps:
            with st.expander(f"{rm['title']} — {rm['description']}"):
                st.markdown(f"[View on roadmap.sh]({rm.get('url', '#')})")
                rm_id = rm["id"]
                ok2, detail, _ = _call_backend("get", f"/roadmaps/{rm_id}")
                if ok2 and detail and "topics" in detail:
                    for topic in detail["topics"]:
                        st.checkbox(topic["title"], key=f"rm_{rm_id}_{topic['id']}")
                        if topic.get("resources"):
                            st.caption(f"Resources: {', '.join(topic['resources'])}")


def page_books():
    st.header(f"{_ui('books')} — NCERT & Open Textbooks")
    col1, col2 = st.columns(2)
    subject = col1.selectbox("Subject", ["All", "Mathematics", "Physics", "Chemistry", "Science", "English", "Computer Science"])
    class_level = col2.selectbox("Class", ["All", 9, 10, 11, 12])
    params = {}
    if subject != "All":
        params["subject"] = subject
    if class_level != "All":
        params["class_level"] = class_level
    ok, books, _ = _call_backend("get", "/books/catalog", params=params)
    if ok and books:
        for book in books:
            with st.expander(f"{book['title']} (Class {book['class_level']})"):
                st.write(f"**Subject:** {book['subject']}")
                st.write(f"**Chapters:** {', '.join(book.get('chapters', []))}")
                if book.get("url"):
                    st.markdown(f"[Download PDF]({book['url']})")


def page_counselor():
    st.header(f"{_ui('counselor')}")
    domain = st.selectbox("Domain", ["general", "stem", "arts", "commerce", "medical", "law"])
    if "counselor_history" not in st.session_state:
        st.session_state.counselor_history = []
    for msg in st.session_state.counselor_history:
        with st.chat_message(msg["role"]):
            st.write(msg["content"])
    user_input = st.chat_input("Ask the AI counselor...")
    if user_input:
        st.session_state.counselor_history.append({"role": "user", "content": user_input})
        with st.chat_message("user"):
            st.write(user_input)
        history_str = "\n".join(f"{m['role']}: {m['content']}" for m in st.session_state.counselor_history[-6:])
        ok, payload, _ = _call_backend("post", "/counselor/chat", json={"message": user_input, "domain": domain, "history": history_str})
        if ok and payload:
            answer = payload.get("answer", "")
            st.session_state.counselor_history.append({"role": "assistant", "content": answer})
            with st.chat_message("assistant"):
                st.write(answer)
                suggestions = payload.get("suggestions", [])
                if suggestions:
                    st.markdown("**Suggestions:**")
                    for s in suggestions:
                        st.caption(f"• {s}")


def page_quiz():
    st.header(f"{_ui('quiz')}")
    topic = st.text_input("Enter a topic for quiz")
    num_q = st.slider("Number of questions", 3, 10, 5)
    if st.button("Generate Quiz") and topic:
        ok, payload, _ = _call_backend("post", "/quiz/generate", json={"topic": topic, "num_questions": num_q})
        if ok and payload:
            st.session_state.current_quiz = payload
            st.rerun()
    if "current_quiz" in st.session_state:
        quiz = st.session_state.current_quiz
        st.subheader(f"Quiz: {quiz.get('topic', '')}")
        answers = []
        for idx, q in enumerate(quiz.get("questions", [])):
            st.markdown(f"**Q{idx+1}.** {q.get('question', '')}")
            opts = q.get("options", ["A", "B", "C", "D"])
            ans = st.radio(f"Answer {idx+1}", opts, key=f"pq_{idx}")
            answers.append(opts.index(ans) if ans in opts else -1)
        if st.button("Submit"):
            ok, result, _ = _call_backend("post", "/quiz/answer", json={
                "topic": quiz.get("topic", ""), "questions": quiz.get("questions", []), "answers": answers,
            })
            if ok and result:
                st.success(f"Score: {result.get('correct', 0)}/{result.get('total', 0)}")


def page_pricing():
    st.header(f"{_ui(\'pricing\')}")
    ok, data, _ = _call_backend("get", "/pages/pricing")
    if ok and data:
        plans = data.get("plans", [])
        if not plans:
            st.info("No pricing plans available.")
            return
        cols = st.columns(len(plans))
        for idx, plan in enumerate(plans):
            with cols[idx]:
                st.markdown(f"### {plan['name']}")
                price = plan.get("price")
                price_str = "Free" if price == 0 else f"₹{price}/{plan.get('period', 'month')}" if price else "Custom"
                st.markdown(f"**{price_str}**")
                st.write(plan.get("description", ""))
                for feature in plan.get("features", []):
                    st.markdown(f"- {feature}")
                if plan["name"] != "Free":
                    if st.button(f"Choose {plan['name']}", key=f"plan_{plan['name']}"):
                        ok2, _, _ = _call_backend("post", "/commerce/cart/add", json={
                            "product_id": f"plan-{plan['name'].lower()}",
                            "product_name": f"{plan['name']} Plan",
                            "price": price or 0,
                        })
                        st.success(f"{plan['name']} added to cart!") if ok2 else None


def page_profile():
    """Profile page with login/register forms for unauthenticated users."""
    token = st.session_state.get("auth_token")
    if token:
        # --- Authenticated: show profile + logout ---
        st.header(f"{_ui('profile')}")
        user = st.session_state.get("auth_user", "Guest")
        role = st.session_state.get("auth_role", "N/A")
        st.write(f"**Username:** {user}")
        st.write(f"**Role:** {role}")
        # Fetch live profile from backend
        ok, profile, _ = _call_backend("get", "/auth/me")
        if ok and profile:
            st.markdown("---")
            c1, c2 = st.columns(2)
            c1.write(f"**Email:** {profile.get('email', '')}")
            c2.write(f"**Full Name:** {profile.get('full_name', '')}")
            st.write(f"**Member since:** {profile.get('created_at', 'N/A')}")
        st.markdown("---")
        if st.button("Logout", type="primary"):
            _call_backend("post", "/auth/logout")
            for key in ["auth_token", "auth_user", "auth_role"]:
                st.session_state.pop(key, None)
            st.rerun()
    else:
        # --- Unauthenticated: show login / register tabs ---
        st.header("Welcome to Sahayak AI")
        st.caption("Sign in or create an account to unlock all features.")
        tab_login, tab_register = st.tabs(["Login", "Register"])
        # ---- Login tab ----
        with tab_login:
            with st.form("login_form"):
                login_user = st.text_input("Username")
                login_pass = st.text_input("Password", type="password")
                submitted = st.form_submit_button("Login", type="primary")
            if submitted and login_user and login_pass:
                ok, payload, detail = _call_backend("post", "/auth/login", json={
                    "username": login_user, "password": login_pass,
                })
                if ok and payload:
                    st.session_state.auth_token = payload.get("access_token", "")
                    st.session_state.auth_user = payload.get("username", login_user)
                    st.session_state.auth_role = payload.get("role", "student")
                    st.success("Login successful!")
                    st.rerun()
                else:
                    st.error(f"Login failed: {detail}")
        # ---- Register tab ----
        with tab_register:
            with st.form("register_form"):
                reg_user = st.text_input("Username", key="reg_user")
                reg_email = st.text_input("Email", key="reg_email")
                reg_name = st.text_input("Full Name (optional)", key="reg_name")
                reg_pass = st.text_input("Password (min 6 chars)", type="password", key="reg_pass")
                reg_role = st.selectbox("Role", ["student", "teacher"], key="reg_role")
                reg_submitted = st.form_submit_button("Create Account", type="primary")
            if reg_submitted and reg_user and reg_email and reg_pass:
                ok, payload, detail = _call_backend("post", "/auth/register", json={
                    "username": reg_user,
                    "email": reg_email,
                    "password": reg_pass,
                    "role": reg_role,
                    "full_name": reg_name,
                })
                if ok and payload:
                    st.success(f"Account created for {reg_user}! Please log in.")
                else:
                    st.error(f"Registration failed: {detail}")


def page_help():
    st.header(f"{_ui('help')} Center")
    question = st.text_input("Ask about Sahayak features...")
    if question and st.button("Ask"):
        ok, payload, _ = _call_backend("post", "/help/ask", json={"question": question})
        if ok and payload:
            st.info(payload.get("answer", ""))
    st.markdown("---")
    st.subheader("FAQ")
    ok, faqs, _ = _call_backend("get", "/help/faq")
    if ok and faqs:
        for faq in faqs:
            with st.expander(faq.get("question", "")):
                st.write(faq.get("answer", ""))


def page_stories():
    st.header(f"{_ui('stories')}")
    ok, stories, _ = _call_backend("get", "/stories")
    if ok and stories:
        for s in stories:
            with st.expander(f"{s.get('title', '')} — {s.get('username', '')} ({'⭐' * s.get('rating', 5)})"):
                st.write(s.get("content", ""))
    st.markdown("---")
    st.subheader("Share Your Story")
    with st.form("story_form"):
        title = st.text_input("Title")
        content = st.text_area("Your experience", height=150)
        rating = st.slider("Rating", 1, 5, 5)
        if st.form_submit_button("Submit") and title and content:
            ok, _, _ = _call_backend("post", "/stories", json={"title": title, "content": content, "rating": rating})
            st.success("Story submitted!") if ok else st.error("Failed")


def page_progress():
    st.header(f"{_ui('progress')}")
    ok, data, _ = _call_backend("get", "/progress")
    if ok and data:
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Courses", data.get("total_courses", 0))
        c2.metric("Completed", data.get("completed_courses", 0))
        c3.metric("Study Time (min)", data.get("total_time_minutes", 0))
        c4.metric("Avg Quiz Score", f"{data.get('average_quiz_score', 0):.1f}")
        for course in data.get("courses", []):
            pct = course.get("completion_percentage", 0)
            st.progress(pct / 100)
            st.write(f"**{course.get('course_id', '')}** — {pct:.0f}% complete")


def page_knowledge():
    st.header(f"{_ui('knowledge')}")
    ok, graph, _ = _call_backend("get", "/knowledge/graph")
    if ok and graph:
        nodes = graph.get("nodes", [])
        edges = graph.get("edges", [])
        st.write(f"**{len(nodes)} entities** | **{len(edges)} relationships**")
        if nodes:
            for n in nodes[:20]:
                st.markdown(f"- **{n['name']}** ({n.get('type', 'concept')})")
    st.markdown("---")
    text = st.text_area("Extract entities from text")
    if text and st.button("Extract"):
        ok, result, _ = _call_backend("post", "/knowledge/extract", json={"text": text})
        if ok and result:
            st.write(f"Found **{len(result.get('entities', []))} entities**")
            st.write(", ".join(result.get("entities", [])))


def page_sync():
    st.header(f"{_ui('sync')}")
    if st.button("Export Data"):
        ok, payload, detail = _call_backend("post", "/sync/export")
        if ok:
            st.success(f"Exported {payload.get('total_records', 0)} records")
            st.json(payload)
        else:
            st.error(f"Export failed: {detail}")
    ok, status, _ = _call_backend("get", "/sync/status")
    if ok:
        st.json(status)


def page_contact():
    st.header("Contact Us")
    ok, data, _ = _call_backend("get", "/pages/contact")
    if ok and data:
        st.write(f"**Email:** {data.get('email', '')}")
        st.write(f"**Support:** {data.get('support_email', '')}")
        st.write(f"**Hours:** {data.get('hours', '')}")
        st.subheader("Social Media")
        socials = data.get("social_media", {})
        for platform, url in socials.items():
            st.markdown(f"- [{platform.capitalize()}]({url})")


def page_about():
    """About Sahayak AI — platform overview and feature list."""
    st.header("ℹ️ About Sahayak AI")
    st.markdown(
        """
**Sahayak AI** is a full-stack multimodal AI learning platform designed to be your 
AI-powered study companion.

### Core Capabilities
- **Multimodal RAG** — Ingest and query PDFs, images, audio, video, code, and CSV/Excel files
- **AI Tutor** — Step-by-step explanations with examples, analogies, and follow-up questions
- **Learning Modes** — Student (guided), Teacher (content creation), Self-Learning (adaptive)
- **Smart Recommendations** — Course suggestions, learning roadmaps, and topic connections
- **Knowledge Graph** — Visualize relationships between concepts across your documents
- **Multilingual** — Auto-detect language, translate, and respond in Hindi, English, Spanish, French, German
- **Voice Assistant** — Voice-to-voice Q&A with transcription and text-to-speech
- **AI Counselor** — Career guidance across STEM, Arts, Commerce, Medical, and Law domains

### Architecture
- Vector DB: Qdrant (primary) + FAISS (local fallback)
- LLM backends: Groq → OpenAI → HuggingFace (fallback chain)
- Embeddings: sentence-transformers/all-MiniLM-L6-v2 (unified singleton)
- Conversation memory: 5-turn sliding window per session

### Version
Sahayak AI Platform v2.0 — All features implemented
        """
    )


def page_settings():
    """Settings page — mode selector, language, theme, API keys."""
    st.header("⚙️ Settings")
    # User mode
    st.subheader("User Mode")
    umode = st.radio("Select your role", ["student", "teacher", "general"],
                     format_func=lambda m: {"student": "🎓 Student", "teacher": "👩‍🏫 Teacher", "general": "💬 General"}.get(m, m),
                     index=["student", "teacher", "general"].index(st.session_state.user_mode),
                     horizontal=True, key="settings_user_mode")
    if umode != st.session_state.user_mode:
        st.session_state.user_mode = umode
    st.caption("Your mode adjusts AI responses and available features.")
    # Theme
    st.markdown("---")
    st.subheader("Theme")
    theme = st.radio("Appearance", ["dark", "light"],
                     format_func=lambda t: {"dark": "🌙 Dark", "light": "☀️ Light"}.get(t, t),
                     index=["dark", "light"].index(st.session_state.theme),
                     horizontal=True, key="settings_theme")
    if theme != st.session_state.theme:
        st.session_state.theme = theme
        st.rerun()
    # Language
    st.markdown("---")
    st.subheader("Language")
    lang = st.selectbox("UI Language", ["en", "hi", "es", "fr", "de"],
                        index=["en", "hi", "es", "fr", "de"].index(st.session_state.language),
                        format_func=lambda x: {"en": "English", "hi": "हिन्दी", "es": "Español", "fr": "Français", "de": "Deutsch"}.get(x, x),
                        key="settings_lang")
    if lang != st.session_state.language:
        st.session_state.language = lang
    # API key
    st.markdown("---")
    st.subheader("API Configuration")
    new_url = st.text_input("Backend URL", value=_get_backend_url(), key="settings_url")
    if st.button("Save URL"):
        st.session_state.backend_url = new_url.strip()
        st.success("Backend URL updated")
    api_key = st.text_input("API Key", type="password", value=_get_api_key() or "", key="settings_apikey")
    if st.button("Save API Key"):
        st.session_state.api_key = api_key.strip()
        st.success("API key updated")


def _render_help_bot():
    """Floating help bot — renders a '?' button that opens a mini Q&A about Sahayak."""
    with st.sidebar:
        st.markdown("---")
        if st.button("❓ Help", key="help_bot_btn", help="How to use Sahayak AI"):
            st.session_state._show_help = not st.session_state.get("_show_help", False)
        if st.session_state.get("_show_help"):
            st.markdown("#### 🤖 Sahayak Help Bot")
            _HELP_FAQ = [
                ("How do I upload documents?", "Go to the Upload page, drop a file (PDF, image, audio, video, code, CSV), and click Ingest."),
                ("How do I ask questions?", "Go to Search & Chat, type your question, and click Get Answer. The AI uses your uploaded documents."),
                ("What are learning modes?", "Student mode gives step-by-step explanations. Teacher mode provides teaching plans. General mode is standard Q&A."),
                ("How does voice work?", "Use the /voice/voice_query endpoint: record audio → AI transcribes → answers via RAG → speaks the answer back."),
                ("Can I use my own language?", "Yes! Select your language in the sidebar. Sahayak auto-detects and translates between Hindi, English, Spanish, French, and German."),
                ("How do I track progress?", "Visit the Progress page to see courses started, topics completed, quiz scores, and study time."),
            ]
            for q, a in _HELP_FAQ:
                with st.expander(q):
                    st.write(a)
            help_q = st.text_input("Ask anything about Sahayak...", key="help_bot_q")
            if help_q and st.button("Ask", key="help_bot_ask"):
                ok, payload, _ = _call_backend("post", "/help/ask", json={"question": help_q})
                if ok and payload:
                    st.info(payload.get("answer", ""))
                else:
                    st.warning("Could not reach help bot.")

# --- Main app ---
def main():
    st.set_page_config(page_title=PAGE_TITLE, page_icon=PAGE_ICON, layout="wide")
    _init_state()

    if not st.session_state.get("auth_token"):
        st.switch_page("pages/login.py")
        return

    _apply_theme()

    with st.sidebar:
        st.markdown("### Sahayak AI")
        lang = st.selectbox("Language", ["en", "hi", "es", "fr", "de"], index=0, key="lang_selector",
                            format_func=lambda x: {"en": "English", "hi": "हिन्दी", "es": "Español", "fr": "Français", "de": "Deutsch"}.get(x, x))
        st.session_state.language = lang
        st.markdown("---")
        pages = ["Dashboard", "Upload", "Search", "Learn", "Quiz", "Chat",
                 "Roadmaps", "Books", "Counselor", "Knowledge Graph", "Progress",
                 "Stories", "Pricing", "Help", "Sync", "Contact", "Settings", "About", "Profile"]
        current = st.session_state.get("page", "Dashboard")
        if current not in pages:
            current = "Dashboard"
        
        selected = st.radio("Navigate", pages, index=pages.index(current), label_visibility="collapsed")
        if selected != st.session_state.get("page"):
            st.session_state["page"] = selected
            st.rerun()
        
        # selected = st.radio("Navigate", pages, index=pages.index(current), key="nav_radio", label_visibility="collapsed")
        # if selected != st.session_state.get("page"):
        #     st.session_state["page"] = selected
        #     st.rerun()
    
    _render_help_bot()
    page_map = {
        "Dashboard": page_dashboard, "Upload": page_upload,
        "Search": page_search, "Chat": page_search,
        "Learn": page_learn, "Quiz": page_quiz,
        "Roadmaps": page_roadmaps, "Books": page_books,
        "Counselor": page_counselor, "Knowledge Graph": page_knowledge,
        "Progress": page_progress, "Stories": page_stories,
        "Pricing": page_pricing, "Help": page_help,
        "Sync": page_sync, "Contact": page_contact,
        "Settings": page_settings, "About": page_about,
        "Profile": page_profile,
    }
    page_map.get(st.session_state.page, page_dashboard)()

main()