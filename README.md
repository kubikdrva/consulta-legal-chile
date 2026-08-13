# Consulta Legal Chile

Self-corrective RAG agent for querying Chilean legislation in natural language. Covers labor law, consumer protection, data privacy, and migration.

## Architecture

```
Query → Hybrid Retrieve (vector + keyword) → RRF Merge → LLM Rerank → Generate → Grade
                                                                                    │
                                                                     grounded? ─── yes → Return
                                                                                   no  → Reformulate → Re-retrieve (max 2)
```

## Stack

| Layer | Technology |
|-------|-----------|
| Orchestration | LangGraph (StateGraph with conditional self-correction) |
| Vector store | PostgreSQL 17 + pgvector |
| Embeddings | OpenAI text-embedding-3-small (1536d) |
| LLM | gpt-4.1-mini |
| Retrieval | Hybrid: cosine similarity + tsvector/tsquery full-text, merged via Reciprocal Rank Fusion |
| Reranking | LLM-based relevance scoring (top 5 from 15 candidates) |
| API | FastAPI (async) |
| Frontend | Next.js + Tailwind CSS |
| Tracing | Langfuse |
| Evaluation | RAGAS + custom LLM-as-judge |

## Corpus

~6,000 chunks from 787 pages of Chilean legal documents:

- **Codigo del Trabajo** - Labor code (358 pages)
- **Guia Derechos Laborales** - Labor rights guide (326 pages)
- **Ley 21.719** - Personal data protection law (34 pages)
- **Informe Datos Personales BCN** - Data privacy analysis (16 pages)
- **Guia Consumidor SERNAC** - Consumer protection guide (44 pages)
- **Ley 21.816** - Migration law (9 pages)

PDFs are not included in the repo due to size. Place them in `docs/` and run ingestion.

## Setup

### 1. PostgreSQL + pgvector

```bash
# Docker
docker compose up -d

# Or Homebrew (macOS)
brew install postgresql@17 pgvector
createuser raguser
createdb ragdb -O raguser
psql ragdb -c "CREATE EXTENSION vector;"
```

### 2. Install dependencies

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 3. Configure environment

```bash
cp .env.example .env
# Add your OpenAI and Langfuse keys
```

### 4. Ingest documents

```bash
python -m src.ingest --dir docs/
```

### 5. Start the backend

```bash
uvicorn src.main:app --port 8000
```

### 6. Start the frontend

```bash
cd web
npm install
npm run dev
```

Open http://localhost:3000

## API

| Method | Path | Description |
|--------|------|-------------|
| POST | `/query` | Question in, answer + sources + confidence out |
| POST | `/ingest` | Upload a PDF/text file |
| GET | `/health` | Health check |

### Example

```bash
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"question": "Cuantos dias de vacaciones tiene un trabajador?"}'
```

## Evaluation

```bash
python eval/run_eval.py
```

15 golden QA pairs covering labor law, consumer rights, data protection, and migration. Metrics: faithfulness, answer relevancy, context precision/recall, plus a custom helpfulness judge.

## Tracing

All queries traced in [Langfuse](https://cloud.langfuse.com) with full observability: retrieved chunks, prompts, token usage, latency, and self-correction loop visibility.

## Demo

The frontend is rate-limited to 5 queries per session for the public demo.

## License

MIT
