import json
import os
from typing import Dict, Optional, Tuple

import requests
import streamlit as st

PAGE_TITLE = "Sahayak - Unified AI Workspace"
PAGE_ICON = "image.png"
BACKEND_URL = os.getenv("BACKEND_URL", "http://127.0.0.1:8000")

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
}


def _init_backend_url() -> None:
    if "backend_url" not in st.session_state:
        st.session_state.backend_url = BACKEND_URL


def _get_backend_url() -> str:
    return st.session_state.get("backend_url", BACKEND_URL)


def _call_backend(method: str, path: str, **kwargs) -> Tuple[bool, Optional[Dict], str]:
    url = f"{_get_backend_url()}{path}"
    try:
        resp = requests.request(method.upper(), url, timeout=kwargs.pop("timeout", 60), **kwargs)
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


def _ingest_file(uploaded_file) -> Tuple[bool, Optional[Dict], str]:
    _, ext = os.path.splitext(uploaded_file.name)
    route = INGESTION_ROUTES.get(ext.lower())
    if route is None:
        return False, None, f"Unsupported file type: {ext or 'unknown'}"
    endpoint, mime_type = route
    uploaded_file.seek(0)
    files = {"file": (uploaded_file.name, uploaded_file, mime_type)}
    return _call_backend("post", endpoint, files=files)


def _ingest_url(url_text: str) -> Tuple[bool, Optional[Dict], str]:
    return _call_backend("post", "/ingest/url", data={"url": url_text})


def _rag_answer(question: str, top_k: int) -> Tuple[bool, Optional[Dict], str]:
    data = {"query": question, "top_k": top_k}
    return _call_backend("post", "/search/rag", data=data)


def _vector_search(query: str, top_k: int) -> Tuple[bool, Optional[Dict], str]:
    data = {"query": query, "top_k": top_k}
    return _call_backend("post", "/search/vector", data=data)


def _summarize_text(text: str) -> Tuple[bool, Optional[Dict], str]:
    return _call_backend("post", "/summaries/text", data={"text": text})


def _backend_health() -> Tuple[str, str]:
    ok, payload, detail = _call_backend("get", "/health", timeout=4)
    if not ok:
        return "error", detail
    status = payload.get("status") if payload else "unknown"
    if status == "healthy":
        return "success", "Backend online"
    return "warning", f"Status: {status}"


def _status_badge() -> None:
    current_url = _get_backend_url()
    status, message = _backend_health()
    if status == "success":
        st.sidebar.success(f"{message} @ {current_url}")
    elif status == "warning":
        st.sidebar.warning(f"{message} @ {current_url}")
    else:
        st.sidebar.error(f"Backend unreachable @ {current_url}\n{message}")


def _sidebar_backend_controls() -> None:
    current_url = _get_backend_url()
    st.sidebar.markdown("### Backend API")
    st.sidebar.markdown(f"[Open FastAPI docs]({current_url}/docs)")
    new_url = st.sidebar.text_input(
        "Base URL",
        value=current_url,
        key="sidebar_backend_url",
        help="Override the backend host if running remotely",
    )
    if st.sidebar.button("Apply", key="sidebar_backend_apply"):
        cleaned = new_url.strip()
        if cleaned:
            st.session_state.backend_url = cleaned
            st.sidebar.success(f"Backend set to {cleaned}")
    st.sidebar.markdown("---")


def _show_response(payload: Optional[Dict], success_msg: str) -> None:
    st.success(success_msg)
    if payload:
        with st.expander("Response payload", expanded=False):
            st.json(payload)


def hero_banner() -> None:
    st.markdown(
        """
        <style>
        .hero-title {
            font-size: 2.8rem;
            font-weight: 700;
            margin-bottom: 0.2rem;
        }
        .hero-subtitle {
            font-size: 1.1rem;
            color: #4f5969;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )
    st.markdown('<div class="hero-title">🤖 Sahayak Unified Workspace</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="hero-subtitle">Ingest anything. Ask anything. One collaborative canvas powered by your local stack.</div>',
        unsafe_allow_html=True,
    )


def quick_ingest_and_ask() -> None:
    st.subheader("⚡ Quick Capture")
    col1, col2 = st.columns([1.5, 1])

    with col1:
        uploaded = st.file_uploader(
            "Drop PDF / Image / Audio / Video",
            type=["pdf", "png", "jpg", "jpeg", "wav", "mp3", "mp4", "mov", "avi"],
            key="quick_uploader",
        )
        if uploaded and st.button("Ingest File", key="quick_ingest_btn"):
            ok, payload, detail = _ingest_file(uploaded)
            if ok:
                _show_response(payload, "File ingested into vector store")
            else:
                st.error(f"Ingestion failed: {detail}")

        url_text = st.text_input("or paste a URL", key="quick_url")
        if url_text and st.button("Ingest URL", key="quick_url_btn"):
            ok, payload, detail = _ingest_url(url_text)
            if ok:
                _show_response(payload, "URL processed")
            else:
                st.error(f"URL ingestion failed: {detail}")

    with col2:
        st.markdown("**Ask instantly**")
        question = st.text_area("Question", key="quick_question", height=120)
        top_k = st.slider("Context chunks", min_value=1, max_value=10, value=4, key="quick_topk")
        if st.button("Get Answer", key="quick_answer_btn"):
            if not question.strip():
                st.warning("Provide a question first.")
            else:
                with st.spinner("Querying RAG pipeline..."):
                    ok, payload, detail = _rag_answer(question, top_k)
                if ok and payload:
                    st.success(payload.get("answer", "No answer yet"))
                    if payload.get("sources"):
                        with st.expander("Sources", expanded=False):
                            for idx, chunk in enumerate(payload["sources"], 1):
                                text_value = chunk.get("content") or chunk.get("text", "")
                                st.markdown(f"**{idx}.** {text_value}")
                else:
                    st.error(f"RAG error: {detail}")


def workspace_tabs() -> None:
    st.subheader("🧠 Workspace")
    tabs = st.tabs(["Upload", "Search", "Recommend", "Summaries"])

    with tabs[0]:
        upload_type = st.radio("Source", ["File", "URL", "Raw Text"], horizontal=True, key="tab_upload_type")
        if upload_type == "File":
            upload_component()
        elif upload_type == "URL":
            url_value = st.text_input("Enter URL", key="tab_url")
            if st.button("Ingest URL", key="tab_url_btn") and url_value:
                ok, payload, detail = _ingest_url(url_value)
                st.success("URL processed") if ok else st.error(f"Failed: {detail}")
        else:
            raw_text = st.text_area("Paste text", key="tab_raw_text")
            if st.button("Ingest Text", key="tab_text_btn") and raw_text.strip():
                ok, payload, detail = _call_backend("post", "/ingest/text", data={"text": raw_text})
                st.success("Text ingested") if ok else st.error(f"Failed: {detail}")

    with tabs[1]:
        query = st.text_input("Semantic search", key="tab_search_query")
        top_k = st.slider("Hits", 1, 15, 5, key="tab_search_topk")
        if st.button("Search", key="tab_search_btn") and query:
            ok, payload, detail = _vector_search(query, top_k)
            if ok and payload:
                hits = payload.get("results") or []
                if not hits:
                    st.info("No matches yet. Ingest content first.")
                for idx, item in enumerate(hits, 1):
                    text_value = item.get("content") or item.get("text", "")
                    st.markdown(f"**{idx}.** {text_value}")
                    st.caption(json.dumps(item.get("metadata", {})))
            else:
                st.error(f"Search failed: {detail}")

    with tabs[2]:
        rec_query = st.text_input("Describe what you need", key="tab_rec_query")
        rec_k = st.slider("Recommendations", 1, 10, 3, key="tab_rec_k")
        if st.button("Recommend", key="tab_rec_btn") and rec_query:
            ok, payload, detail = _vector_search(rec_query, rec_k)
            if ok and payload:
                hits = payload.get("results") or []
                if not hits:
                    st.info("No recommendations yet.")
                for idx, item in enumerate(hits, 1):
                    text_value = item.get("content") or item.get("text", "")
                    st.markdown(f"**{idx}.** {text_value}")
            else:
                st.error(f"Recommendation failed: {detail}")

    with tabs[3]:
        summary_text = st.text_area("Summarize text", key="tab_summary_text")
        if st.button("Summarize", key="tab_summary_btn") and summary_text.strip():
            ok, payload, detail = _summarize_text(summary_text)
            if ok and payload:
                st.success(payload.get("summary", ""))
            else:
                st.error(f"Summary failed: {detail}")


def upload_component() -> None:
    upload = st.file_uploader(
        "Select file",
        type=["pdf", "png", "jpg", "jpeg", "wav", "mp3", "mp4", "mov", "avi", "txt"],
        key="tab_file_uploader",
    )
    if upload and st.button("Process", key="tab_file_btn"):
        ok, payload, detail = _ingest_file(upload)
        if ok:
            _show_response(payload, "File ingested")
        else:
            st.error(f"Failed: {detail}")


def mini_lab() -> None:
    with st.expander("🧪 HuggingFace-style Mini Lab"):
        file = st.file_uploader("Upload small PDF/Image", type=["pdf", "png", "jpg", "jpeg"], key="lab_file")
        if file and st.button("Lab ingest", key="lab_ingest_btn"):
            ok, _, detail = _ingest_file(file)
            st.success("Lab file ingested") if ok else st.error(f"Lab ingest failed: {detail}")

        lab_question = st.text_input("Ask", key="lab_question")
        if st.button("Lab Ask", key="lab_ask_btn") and lab_question:
            ok, payload, detail = _rag_answer(lab_question, 3)
            if ok and payload:
                st.success(payload.get("answer", "No answer yet"))
            else:
                st.error(f"Lab error: {detail}")


def main() -> None:
    _init_backend_url()
    st.set_page_config(page_title=PAGE_TITLE, page_icon=PAGE_ICON, layout="wide")
    hero_banner()
    _sidebar_backend_controls()
    _status_badge()
    quick_ingest_and_ask()
    st.divider()
    workspace_tabs()
    st.divider()
    mini_lab()


main()
