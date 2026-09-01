# 🚀 Sahayak AI — Multimodal AI Learning Platform v2.0

A production-grade, full-stack AI learning platform featuring **Multimodal RAG**, **JWT Authentication**, **Conversational Memory**, and **Multilingual Support**.

---

## 🏗️ Architecture

```text
User
  │
  ▼
Streamlit Frontend
  │
  ▼
FastAPI Backend
  │
  ├── Qdrant (Primary Vector DB)
  ├── FAISS (Fallback Vector DB)
  ├── Groq LLM
  ├── OpenAI LLM
  └── HuggingFace Models
```

---

## ✨ Key Features

### 📚 Multimodal RAG

Supports ingestion and retrieval from:

* PDF Documents
* Images
* Audio Files
* Videos
* Source Code (`.py`, `.js`, `.cpp`)
* CSV / Excel Files
* URLs
* YouTube Videos

### 🤖 Agentic AI Pipeline

```text
Query
  ↓
Query Rewrite
  ↓
Semantic Chunking
  ↓
Hybrid Retrieval
  ↓
Re-ranking
  ↓
LLM Response
```

### 🔐 Authentication & Authorization

* JWT Authentication
* User Registration & Login
* Role-Based Access Control

  * Student
  * Teacher
  * Admin
* Secure Password Hashing with bcrypt

### 🧠 Conversation Memory

* LangChain Memory Integration
* 5-Turn Sliding Window
* Session-Aware Conversations

### 🌍 Multilingual Support

* English
* Hindi
* Spanish
* French
* German

### 🎓 Learning Modes

* Student Mode
* Teacher Mode
* Self-Learning Mode

### 🚀 AI-Powered Features

* Quiz Generator
* AI Career Counselor
* Knowledge Graph Generation
* Learning Progress Tracking
* Personalized Learning Roadmaps

### 🛡️ Fault-Tolerant Design

#### Vector Database Fallback

```text
Qdrant
   ↓ (Failure)
FAISS + SQLite
```

#### LLM Fallback Chain

```text
Groq (Llama 3 70B)
        ↓
      OpenAI
        ↓
   HuggingFace
```

---

## 🛠️ Tech Stack

| Layer            | Technology                                  |
| ---------------  | ----------------------------------------    |
| Frontend (React) | React + Vite + Tailwind CSS (`sahayak-ui`)  |
| Frontend(Python) | Streamlit (`frontend/app.py`)               |
| Backend          | FastAPI + SQLAlchemy                        |
| Vector Database  | Qdrant + FAISS                              |
| Embeddings       | sentence-transformers/all-MiniLM-L6-v2      |
| LLMs             | Groq → OpenAI → HuggingFace                 |
| Authentication   | JWT + bcrypt + OAuth2                       |
| Memory           | LangChain ConversationBufferWindowMemory    |

---

## ⚡ Quick Start

### 1. Clone Repository

```bash
git clone https://github.com/your-username/SAHAYAK_AI.git
cd SAHAYAK_AI
```

### 2. Create Virtual Environment

```bash
python -m venv venv
```

### 3. Activate Environment

#### Windows

```powershell
.\venv\Scripts\activate
```

#### Linux / macOS

```bash
source venv/bin/activate
```

### 4. Install Dependencies

```bash
pip install -r requirements.txt
```

---

## ▶️ Run Backend

```bash
uvicorn backend.main:app --reload --port 8000
```

---

## ▶️ Run Frontends

You can use either (or both) of the supported frontend applications:

### Option A: React Frontend (Modern UI)

```bash
cd sahayak-ui
npm install
npm run dev
```

### Option B: Streamlit Frontend (Classic UI)

```bash
streamlit run frontend/app.py
```

---

## 🔑 Environment Variables

Create a `.env` file in the project root:

```env
QDRANT_URL=your_qdrant_url
QDRANT_API_KEY=your_qdrant_api_key
QDRANT_COLLECTION=sahayak

GROQ_API_KEY=your_groq_api_key

JWT_SECRET_KEY=your_secret_key
```

---

## 🌐 API Routes

| Endpoint           | Description                         |
| ------------------ | ----------------------------------- |
| `/auth`            | Register, Login, JWT Authentication |
| `/ingest`          | Multimodal Data Ingestion           |
| `/search/rag`      | RAG Query with Memory               |
| `/document`        | Summarization, Notes, Explanation   |
| `/quiz`            | AI Quiz Generation                  |
| `/counselor`       | AI Career Counselor                 |
| `/roadmaps`        | Learning Roadmaps                   |
| `/stats/dashboard` | Platform Analytics                  |

---

## 🐳 Docker Deployment

### Build & Start

```bash
docker compose up --build
```

### Run in Background

```bash
docker compose up -d
```

---

## 📈 Future Roadmap

* Voice-to-Voice Conversations
* Real-Time Collaborative Learning
* Advanced Analytics Dashboard
* Multi-Agent Learning Assistants
* Mobile Application Support
* LMS Integrations

---

## 🤝 Contributing

Contributions are welcome.

```bash
fork → create branch → commit → push → pull request
```

---

## 📄 Research Paper

[Research Paper (zenodo)](https://zenodo.org/records/20682334)

---

## 📄 License

MIT License

---



## 👨‍💻 Author

### Shikher Jain

Founder & Developer of **Sahayak AI**

Building the future of AI-powered education through Multimodal AI, RAG Systems, and Intelligent Learning Platforms.

* [GitHub](https://github.com/Shikher-jain)
* [LinkedIn](https://www.linkedin.com/in/shikher-jain-0bb8a8259/)

---
