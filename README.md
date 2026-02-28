# Poetry AI Assistant

An end-to-end poetry analysis, question-answering, and recommendation experience that combines a FastAPI + LangGraph backend with a React front end. The system embeds poems, classifies style, recommends similar works, and keeps multi-turn chat context to help users explore poetry.

## Features

- **Conversational Poetry Q&A** powered by retrieval-augmented generation and third-party search fallbacks.
- **Poem Classification** using an SVM model trained on labeled embeddings.
- **Personalized Recommendations** via clustering over poem embeddings.
- **Graph-Orchestrated Workflow** managed by LangGraph for flexible tool routing.
- **Modern Web UI** with React hooks and components tailored for long-form chat sessions.

## Tech Stack

- Backend: FastAPI, LangGraph, Pydantic, Uvicorn.
- Frontend: React (Vite-like setup), custom hooks, REST client.
- ML Models: SentenceTransformer embeddings, SVM classifier, K-Means clustering.
- Infra: Pinecone (vector store), Google Gemini API, DuckDuckGo search fallback.

## Repository Layout

```
poetry-ai/
├── src/
│   ├── backend/        # FastAPI app, LangGraph, services
│   └── frontend/       # React UI, hooks, services
├── models/             # Embedding, classification, clustering assets
├── data/               # Training/evaluation data
├── scripts/            # Helper scripts (start backend/frontend)
├── tests/              # Backend + frontend tests
├── requirements.txt    # Python dependencies
└── README.md           # Project documentation
```

## Prerequisites

- Python 3.10+
- Node.js 18+
- Valid Pinecone project + index
- Google Gemini API key

## Installation

1. **Clone and enter the workspace**
   ```bash
   git clone <repo-url> data-science-application
   cd data-science-application/poetry-ai
   ```

2. **Create and activate a Python environment**
   ```bash
   python -m venv .datascienceapp
   source .datascienceapp/bin/activate  # macOS/Linux
   # .\.datascienceapp\Scripts\activate  # Windows
   ```

3. **Install backend dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Install frontend dependencies**
   ```bash
   cd src/frontend
   npm install
   ```

5. **Configure environment variables**
   ```bash
   cd ../..
   cp .env.example .env
   # Fill in Pinecone + Gemini credentials
   ```

6. **Place trained models**
   - `models/embedding/`: SentenceTransformer export
   - `models/classification/svm_model.pkl`
   - `models/clustering/kmeans.pkl`

## Usage

1. **Start the backend** (from `poetry-ai/` root)
   ```bash
   uvicorn src.backend.main:app --reload --port 8000
   ```
   - API docs: http://localhost:8000/docs

2. **Start the frontend**
   ```bash
   cd src/frontend
   npm start
   ```
   - UI: http://localhost:3000 (configured to talk to the backend on port 8000)

### Available API Routes

| Method | Endpoint      | Description                    |
|--------|---------------|--------------------------------|
| GET    | `/health`     | Basic health probe             |
| POST   | `/chat`       | Conversational poetry assistant|
| POST   | `/resume`     | Resume an interrupted session  |
| POST   | `/classify`   | Predict poem genre/style       |
| POST   | `/recommend`  | Return similar poems           |

## Testing

```bash
# Backend tests
pytest tests/backend

# Frontend tests
cd src/frontend
npm test
```

## Troubleshooting

- Ensure `.env` values (Pinecone index, Gemini key) are present before starting the backend.
- If embeddings or classifiers are missing, confirm the `models/` directory mirrors the expected layout above.
- When developing the frontend alongside the backend, keep both servers running; the React dev server proxies API requests to FastAPI.

## License

This project is released under the MIT License. See `LICENSE` for details.
