# Poetry AI Assistant

An AI-powered poetry analysis, classification, and recommendation application built with FastAPI, LangGraph, and React.
## Project Structure

```
data-science-application/
├── src/
│   ├── backend/                    # FastAPI backend
│   │   ├── api/                    # API layer
│   │   │   ├── routes.py           # Route definitions
│   │   │   └── schemas.py          # Pydantic schemas
│   │   ├── core/                   # Core modules
│   │   │   ├── config.py           # Configuration settings
│   │   │   └── state.py            # Application state & ML models
│   │   ├── graph/                  # LangGraph workflow
│   │   │   └── poetry_graph.py     # Poetry assistant graph
│   │   ├── services/               # Business logic
│   │   │   ├── classification.py   # Poem classification
│   │   │   └── recommendation.py   # Poem recommendations
│   │   └── main.py                 # FastAPI application
│   │
│   └── frontend/                   # React frontend
│       ├── App/                    # Main app component
│       ├── components/             # UI components
│       │   ├── Header/
│       │   ├── InputArea/
│       │   ├── Message/
│       │   └── MessageList/
│       ├── hooks/                  # Custom React hooks
│       │   └── useChat.js
│       └── services/               # API services
│           ├── apiClient.js
│           └── chatService.js
│
├── models/                         # ML models directory
│   ├── embedding/                  # Sentence transformer model
│   ├── classification/             # SVM classifier
│   └── clustering/                 # K-means model
│
├── data/                           # Data files
│   └── poem_embedding_and_label.csv
│
├── tests/                          # Test files
│   ├── backend/
│   └── frontend/
│
├── scripts/                        # Utility scripts
├── docs/                           # Documentation
├── .env.example                    # Environment variables template
├── requirements.txt                # Python dependencies
├── package.json                    # Node.js dependencies
└── README.md
```

## Features

- **Poetry Q&A**: Ask questions about poets and poems using RAG with Pinecone
- **Poem Classification**: Classify poems by genre/style using trained SVM
- **Poem Recommendations**: Get content-based recommendations using clustering
- **Multi-turn Conversations**: Maintain context across conversation turns
- **Web Search Fallback**: DuckDuckGo search for unknown queries

## Prerequisites

- Python 3.10+
- Node.js 18+
- Pinecone account
- Google Gemini API key

## Setup

### 1. Clone and navigate to the project

```bash
cd data-science-application
```

### 2. Set up Python environment

```bash
python -m venv .datascienceapp
source .datascienceapp/bin/activate  # macOS/Linux
# or
.\.datascienceapp\Scripts\activate   # Windows

pip install -r requirements.txt
```

### 3. Configure environment variables

```bash
cp .env.example .env
# Edit .env with your API keys
```

### 4. Set up the frontend

```bash
cd src/frontend
npm install
```

### 5. Place ML models

Copy your trained models to the `models/` directory:
- `models/embedding/` - SentenceTransformer model
- `models/classification/svm_model.pkl` - SVM classifier
- `models/clustering/kmeans.pkl` - K-means model

## Running the Application

### Backend

```bash
# From project root
uvicorn src.backend.main:app --reload --port 8000
```

### Frontend

```bash
# From src/frontend
npm start
```

The application will be available at:
- Frontend: http://localhost:3000
- Backend API: http://localhost:8000
- API Docs: http://localhost:8000/docs

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/health` | Health check |
| POST | `/chat` | Send chat query |
| POST | `/resume` | Resume interrupted conversation |
| POST | `/classify` | Classify poem text |
| POST | `/recommend` | Get poem recommendations |

## Development

### Running Tests

```bash
# Backend tests
pytest tests/backend/

# Frontend tests
npm test
```

### Code Style

```bash
# Python
ruff check src/backend/
black src/backend/

# JavaScript
npm run lint
```

## Architecture

The application uses a **LangGraph** workflow to route queries:

1. **Query Classification**: Determines if query is about poet or poem
2. **Tool Selection**: Routes to appropriate handler (QA, classification, recommendation)
3. **Response Generation**: Returns formatted response to user

## License

MIT License
