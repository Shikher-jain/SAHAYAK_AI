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

## 🏗️ Architecture & Tech Stack

| Layer            | Technology                                  |
| ---------------  | ----------------------------------------    |
| Frontend (React) | React + Vite + Tailwind CSS (`sahayak-ui`)  |
| Frontend(Python) | Streamlit (`frontend/app.py`)               |
| Backend          | FastAPI (Asynchronous) + SQLAlchemy         |
| Vector Database  | Qdrant + FAISS                              |
| Embeddings       | sentence-transformers/all-MiniLM-L6-v2      |
| LLMs             | Groq → OpenAI → HuggingFace                 |
| Authentication   | JWT + bcrypt + OAuth2                       |
| Memory           | LangChain ConversationBufferWindowMemory    |

---

## 📁 Project Directory Structure

```text
SAHAYAK_AI/
│
├── backend/                  # Core FastAPI Backend
│   ├── auth_system/          # JWT Auth DB Models & Middleware
│   ├── common/               # Rate Limiting & Logging
│   ├── ingestion/            # URL, Text, PDF, Audio parsing (SSRF protected)
│   ├── routers/              # Decoupled API routes (ingest, quiz, search)
│   ├── services/             # Core business logic & AI orchestration
│   ├── vector_store/         # Qdrant & FAISS integrations
│   └── main.py               # FastAPI App Entrypoint
│
├── frontend/                 # Classic UI
│   └── app.py                # Streamlit Application
│
├── sahayak-ui/               # Modern UI
│   └── src/                  # React + Vite + Tailwind Frontend
│
├── data/                     # Persistent local databases (SQLite/FAISS)
├── docker-compose.yml        # Container orchestration
└── requirements.txt          # Python dependencies
```

---

## 🚀 Installation & Setup

### 1. Clone the Repository
```bash
git clone https://github.com/Shikher-jain/SAHAYAK_AI.git
cd SAHAYAK_AI
```

### 2. Environment Setup (Backend)
Create an isolated Python environment and install the required dependencies:
```bash
python -m venv venv

# Windows:
.\venv\Scripts\activate

# Linux/Mac:
source venv/bin/activate

pip install -r requirements.txt
```

### 3. Environment Variables
Create a `.env` file in the project root:
```env
# Vector DB Configuration
QDRANT_URL=your_qdrant_url
QDRANT_API_KEY=your_qdrant_api_key
QDRANT_COLLECTION=sahayak

# LLM Configurations
OPENAI_API_KEY=your_openai_api_key

# Security
JWT_SECRET_KEY=your_super_secret_key
ALLOWED_ORIGINS=http://localhost:5173,http://localhost:8501
```

### 4. Running the Application

**Backend API:**
```bash
uvicorn backend.main:app --reload --port 8000
```

**Frontend (React):**
```bash
cd sahayak-ui
npm install
npm run dev
```

**Frontend (Streamlit):**
```bash
streamlit run frontend/app.py
```

*(Alternatively, deploy the full stack using Docker: `docker compose up --build`)*

---

## 🌐 API Endpoints Summary

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
