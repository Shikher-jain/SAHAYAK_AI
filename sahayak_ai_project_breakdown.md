# Sahayak AI: Project Breakdown

## 1. Executive Summary (What is Sahayak AI?)
Sahayak AI is a production-grade, multimodal AI learning platform designed to make education and data interaction intelligent, secure, and highly accessible. It functions as a privacy-preserving assistant capable of ingesting and understanding a massive variety of data—ranging from text and PDFs to audio, video, and source code. Built with a robust offline-first fallback system, it ensures continuous availability and reliability even when primary cloud services fail.

## 2. The 'Why' (The Vision & Problem Solved)
In an era where AI is becoming central to learning and productivity, users often face a difficult trade-off between advanced capabilities and data privacy. Many existing platforms rely entirely on fragile cloud infrastructure, risking data exposure and suffering from downtime when third-party API services experience outages. 

Sahayak AI was built to bridge this gap by offering:
* **Privacy-Preserving AI**: Your data is processed securely with enterprise-grade JWT authentication and role-based access control.
* **Offline-First Resilience**: An intelligent fallback mechanism ensures that if primary cloud components (like external Vector Databases or LLMs) fail, the system seamlessly transitions to local, offline alternatives without dropping the user's query.
* **Holistic Learning**: Moving beyond simple text chatbots, it embraces how people naturally learn—through images, audio, and video—bringing all these modalities under one roof.

## 3. Core Features 
* **Multimodal Ingestion Pipeline**: Effortlessly processes and understands a wide array of formats:
  * **Documents & Code**: PDFs, CSVs, Excel, and source code (`.py`, `.js`, `.cpp`).
  * **Media**: Images, Audio files, and Videos (including direct YouTube URL processing).
* **Advanced Conversational Memory**: Utilizes a LangChain-powered 5-turn sliding window to maintain session-aware, contextually rich conversations.
* **Role-Based Access Control (RBAC)**: Tailored interfaces, features, and permissions for Students, Teachers, and Admins.
* **AI-Powered Educational Tools**: 
  * Automated Quiz Generator
  * AI Career Counselor
  * Dynamic Knowledge Graph Generation
  * Personalized Learning Roadmaps
* **Multilingual Support**: Breaks language barriers with native support for English, Hindi, Spanish, French, and German.
* **Dual-Interface Options**: Offers both a modern React/Tailwind frontend and a classic Python Streamlit UI to suit different deployment needs and preferences.

## 4. How It Works (The Architecture)
Sahayak AI operates on a highly decoupled, fault-tolerant architecture:

1. **User Interaction**: Users interact via the Frontend (React or Streamlit). Inputs can be standard text queries or file uploads (audio, video, images, PDFs).
2. **API Layer (FastAPI)**: The backend receives the request, instantly validating authentication via JWT.
3. **Agentic RAG Pipeline**:
   * **Ingestion & Chunking**: Uploaded media is transcribed or extracted, semantically chunked, and converted into dense embeddings using `sentence-transformers`.
   * **Vector Storage**: These embeddings are stored in **Qdrant** (Primary). If Qdrant is unreachable or network drops, the system automatically falls back to a local **FAISS + SQLite** setup.
   * **Query Processing**: User queries are rewritten for maximum clarity by an agent, then matched against the vector database using hybrid retrieval and re-ranking techniques.
4. **LLM Generation with Fallback**: The retrieved context is passed to the language model. The system attempts to use **openai/gpt-oss-20b** for ultra-fast generation. If it fails, it cascades gracefully to **OpenAI**, and finally to local **HuggingFace** models.
5. **Response**: The final, context-aware answer is delivered back to the user interface in milliseconds.

## 5. Key Benefits (Why use it?)
* **🔒 Uncompromising Security**: Built-in authentication, secure password hashing, and restricted API routes keep your sensitive data safe from unauthorized access.
* **🛡️ Unmatched Reliability**: The dual-layer fallback system (for both Vector DB and LLMs) guarantees maximum uptime and resilience against network or third-party outages.
* **🧠 True Versatility**: Whether you are querying a YouTube video, debugging a C++ file, or generating a personalized learning roadmap, Sahayak AI handles it all natively in a single platform.
* **⚡ High Performance**: Leveraging asynchronous FastAPI and ultra-fast inference engines ensures incredibly low-latency responses.
* **🌍 Inclusive Accessibility**: Multilingual capabilities make it a globally viable tool for incredibly diverse user bases.
