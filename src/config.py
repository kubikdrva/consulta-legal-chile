from __future__ import annotations

import os

from dotenv import load_dotenv

load_dotenv()

OPENAI_API_KEY = os.environ["OPENAI_API_KEY"]
DATABASE_URL = os.environ.get(
    "DATABASE_URL",
    "postgresql+psycopg://raguser:ragpass@localhost:5432/ragdb",
)
LANGFUSE_PUBLIC_KEY = os.environ.get("LANGFUSE_PUBLIC_KEY", "")
LANGFUSE_SECRET_KEY = os.environ.get("LANGFUSE_SECRET_KEY", "")
LANGFUSE_HOST = os.environ.get("LANGFUSE_HOST", "https://cloud.langfuse.com")
ENVIRONMENT = os.environ.get("ENVIRONMENT", "development")

EMBEDDING_MODEL = "text-embedding-3-small"
EMBEDDING_DIMENSIONS = 1536
LLM_MODEL = "gpt-4.1-mini"

CHUNK_SIZE = 800
CHUNK_OVERLAP = 200

TABLE_NAME = "document_chunks"

RERANK_TOP_K = 5
RETRIEVE_TOP_K = 15
MAX_RETRIES = 2
