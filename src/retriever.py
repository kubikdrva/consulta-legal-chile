from __future__ import annotations

import json

from langchain_core.documents import Document
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_postgres import PGEngine, PGVectorStore

from src.config import (
    DATABASE_URL,
    EMBEDDING_MODEL,
    LLM_MODEL,
    RERANK_TOP_K,
    RETRIEVE_TOP_K,
    TABLE_NAME,
)
from src.prompts import RERANK_PROMPT

_engine: PGEngine | None = None
_store: PGVectorStore | None = None


async def _get_store() -> PGVectorStore:
    global _engine, _store
    if _store is None:
        _engine = PGEngine.from_connection_string(DATABASE_URL)
        embedding = OpenAIEmbeddings(model=EMBEDDING_MODEL)
        _store = await PGVectorStore.create(
            engine=_engine,
            table_name=TABLE_NAME,
            embedding_service=embedding,
        )
    return _store


async def _vector_search(query: str, k: int = RETRIEVE_TOP_K) -> list[Document]:
    store = await _get_store()
    return await store.asimilarity_search_with_score(query, k=k)


async def _keyword_search(query: str, k: int = RETRIEVE_TOP_K) -> list[Document]:
    import asyncpg

    terms = " & ".join(query.split()[:10])
    sql = f"""
        SELECT content, langchain_metadata
        FROM {TABLE_NAME}
        WHERE to_tsvector('english', content) @@ to_tsquery('english', $1)
        ORDER BY ts_rank(to_tsvector('english', content), to_tsquery('english', $1)) DESC
        LIMIT $2
    """
    conn = await asyncpg.connect(
        user="raguser", password="ragpass", database="ragdb", host="localhost"
    )
    results = []
    try:
        rows = await conn.fetch(sql, terms, k)
        for row in rows:
            metadata = json.loads(row["langchain_metadata"]) if row["langchain_metadata"] else {}
            results.append(Document(page_content=row["content"], metadata=metadata))
    finally:
        await conn.close()
    return results


def _reciprocal_rank_fusion(
    result_lists: list[list], k: int = 60
) -> list[Document]:
    scores: dict[str, float] = {}
    doc_map: dict[str, Document] = {}

    for results in result_lists:
        for rank, item in enumerate(results):
            doc = item[0] if isinstance(item, tuple) else item
            doc_id = doc.page_content[:200]
            if doc_id not in doc_map:
                doc_map[doc_id] = doc
                scores[doc_id] = 0.0
            scores[doc_id] += 1.0 / (k + rank + 1)

    sorted_ids = sorted(scores, key=lambda x: scores[x], reverse=True)
    return [doc_map[doc_id] for doc_id in sorted_ids]


async def hybrid_retrieve(query: str) -> list[Document]:
    vector_results = await _vector_search(query)
    keyword_results = await _keyword_search(query)
    fused = _reciprocal_rank_fusion([vector_results, keyword_results])
    return fused[:RETRIEVE_TOP_K]


async def rerank(query: str, documents: list[Document]) -> list[Document]:
    if not documents:
        return []

    llm = ChatOpenAI(model=LLM_MODEL, temperature=0)
    scored = []

    for doc in documents:
        prompt = RERANK_PROMPT.format(query=query, chunk=doc.page_content[:500])
        response = await llm.ainvoke(prompt)
        try:
            result = json.loads(response.content)
            score = float(result.get("score", 0))
        except (json.JSONDecodeError, ValueError):
            score = 0.0
        scored.append((doc, score))

    scored.sort(key=lambda x: x[1], reverse=True)
    return [doc for doc, _ in scored[:RERANK_TOP_K]]
