# 🤖 Full-Stack AI RAG Assistant

A production-ready **Retrieval-Augmented Generation (RAG)** application that enables intelligent question answering over custom datasets. Built with LangChain, FAISS, FastAPI, and React.

![RAG Assistant Demo](docs/demo.png)

---

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                        React Frontend                        │
│              (Real-time chat + Document Upload)              │
└───────────────────────────┬─────────────────────────────────┘
                            │ HTTP / WebSocket
┌───────────────────────────▼─────────────────────────────────┐
│                    FastAPI Backend                            │
│   ┌─────────────┐  ┌──────────────┐  ┌──────────────────┐  │
│   │  LangChain  │  │  FAISS Index │  │  OpenAI / LLM    │  │
│   │  Orchestr.  │  │  Vector DB   │  │  API Integration │  │
│   └─────────────┘  └──────────────┘  └──────────────────┘  │
└─────────────────────────────────────────────────────────────┘
```

## ✨ Features

- 📄 **Document Upload** — PDF, TXT, DOCX, CSV support
- 🔍 **Semantic Search** — FAISS vector similarity search
- 🤖 **RAG Pipeline** — LangChain-powered context-aware responses
- ⚡ **Real-time Streaming** — Server-sent events for live responses
- 🗂️ **Multiple Knowledge Bases** — Manage separate document collections
- 🔐 **API Key Auth** — Secure endpoints
- 📊 **Source Citations** — Every answer cites its source documents
- 🐳 **Docker Ready** — One command deployment

---

## 🚀 Quick Start

### Prerequisites
- Python 3.11+
- Node.js 18+
- Docker (optional)
- OpenAI API Key

### 1. Clone & Configure

```bash
git clone https://github.com/YOUR_USERNAME/rag-assistant.git
cd rag-assistant
cp backend/.env.example backend/.env
# Edit backend/.env and add your OPENAI_API_KEY
```

### 2. Run with Docker (Recommended)

```bash
docker-compose up --build
```

App will be available at:
- **Frontend**: http://localhost:3000
- **Backend API**: http://localhost:8000
- **API Docs**: http://localhost:8000/docs

### 3. Run Manually

**Backend:**
```bash
cd backend
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt
uvicorn app.main:app --reload --port 8000
```

**Frontend:**
```bash
cd frontend
npm install
npm start
```

---

## 📡 API Reference

| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/api/v1/documents/upload` | Upload documents |
| `GET` | `/api/v1/documents/` | List all documents |
| `DELETE` | `/api/v1/documents/{id}` | Delete a document |
| `POST` | `/api/v1/chat/query` | Ask a question |
| `POST` | `/api/v1/chat/stream` | Streaming query |
| `GET` | `/api/v1/collections/` | List knowledge bases |
| `POST` | `/api/v1/collections/` | Create knowledge base |
| `GET` | `/api/v1/health` | Health check |

### Example Query

```bash
curl -X POST http://localhost:8000/api/v1/chat/query \
  -H "Content-Type: application/json" \
  -d '{
    "question": "What is the refund policy?",
    "collection_id": "default",
    "top_k": 4
  }'
```

---

## 📁 Project Structure

```
rag-assistant/
├── backend/
│   ├── app/
│   │   ├── api/          # Route handlers
│   │   ├── core/         # Config, security, dependencies
│   │   ├── models/       # Pydantic schemas
│   │   ├── services/     # RAG engine, embeddings, LLM
│   │   └── utils/        # Helpers, document loaders
│   ├── tests/            # Pytest test suite
│   ├── requirements.txt
│   └── Dockerfile
├── frontend/
│   ├── src/
│   │   ├── components/   # Reusable UI components
│   │   ├── pages/        # Route pages
│   │   ├── hooks/        # Custom React hooks
│   │   └── services/     # API client
│   ├── package.json
│   └── Dockerfile
├── docker-compose.yml
└── README.md
```

---

## 🛠️ Tech Stack

| Layer | Technology |
|-------|-----------|
| LLM Orchestration | LangChain 0.3 |
| Vector Database | FAISS |
| Embeddings | OpenAI `text-embedding-3-small` |
| LLM | OpenAI GPT-4o-mini |
| Backend | FastAPI + Uvicorn |
| Frontend | React 18 + TypeScript |
| Styling | Tailwind CSS |
| Containerization | Docker + Docker Compose |

---

## 🧪 Running Tests

```bash
cd backend
pytest tests/ -v --cov=app
```

---

## 🤝 Contributing

1. Fork the repo
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

---

## 📄 License

MIT License — see [LICENSE](LICENSE) for details.
