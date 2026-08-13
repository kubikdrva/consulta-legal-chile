from __future__ import annotations

import argparse
import asyncio
import os
from pathlib import Path

from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_openai import OpenAIEmbeddings
from langchain_postgres import Column, PGEngine, PGVectorStore
from langchain_text_splitters import RecursiveCharacterTextSplitter

from src.config import (
    CHUNK_OVERLAP,
    CHUNK_SIZE,
    DATABASE_URL,
    EMBEDDING_DIMENSIONS,
    EMBEDDING_MODEL,
    TABLE_NAME,
)


def _get_embedding():
    return OpenAIEmbeddings(model=EMBEDDING_MODEL)


def _get_splitter():
    return RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " ", ""],
    )


def _clean_text(text: str) -> str:
    return text.replace("—", "-").replace("–", "-").replace("‘", "'").replace("’", "'").replace("“", '"').replace("”", '"')


def _load_file(path: Path):
    ext = path.suffix.lower()
    if ext == ".pdf":
        docs = PyPDFLoader(str(path)).load()
    elif ext in (".md", ".txt", ".rst"):
        docs = TextLoader(str(path), encoding="utf-8").load()
    else:
        raise ValueError(f"Unsupported file type: {ext}")
    for doc in docs:
        doc.page_content = _clean_text(doc.page_content)
    return docs


async def _ensure_table(engine: PGEngine):
    try:
        await engine.ainit_vectorstore_table(
            table_name=TABLE_NAME,
            vector_size=EMBEDDING_DIMENSIONS,
            metadata_columns=[
                Column("source", "TEXT"),
            ],
        )
    except Exception:
        pass


async def _create_fts_index():
    import asyncpg

    conn = await asyncpg.connect(
        user="raguser", password="ragpass", database="ragdb", host="localhost"
    )
    try:
        await conn.execute(
            f"""
            CREATE INDEX IF NOT EXISTS idx_{TABLE_NAME}_fts
            ON {TABLE_NAME}
            USING gin(to_tsvector('english', content));
            """
        )
    finally:
        await conn.close()


async def ingest_directory(directory: str) -> int:
    dir_path = Path(directory)
    if not dir_path.is_dir():
        raise FileNotFoundError(f"Directory not found: {directory}")

    splitter = _get_splitter()
    embedding = _get_embedding()

    all_chunks = []
    for file_path in sorted(dir_path.iterdir()):
        if file_path.suffix.lower() not in (".pdf", ".md", ".txt", ".rst"):
            continue
        raw_docs = _load_file(file_path)
        chunks = splitter.split_documents(raw_docs)
        for chunk in chunks:
            chunk.metadata["source"] = file_path.name
        all_chunks.extend(chunks)
        print(f"  Loaded {file_path.name}: {len(raw_docs)} pages -> {len(chunks)} chunks")

    if not all_chunks:
        print("No documents found to ingest.")
        return 0

    engine = PGEngine.from_connection_string(DATABASE_URL)
    await _ensure_table(engine)

    store = await PGVectorStore.create(
        engine=engine,
        table_name=TABLE_NAME,
        embedding_service=embedding,
    )

    await store.aadd_documents(all_chunks)
    await _create_fts_index()

    print(f"Ingested {len(all_chunks)} chunks into pgvector.")
    await engine.close()
    return len(all_chunks)


async def ingest_file(file_path: str) -> int:
    path = Path(file_path)
    if not path.is_file():
        raise FileNotFoundError(f"File not found: {file_path}")

    splitter = _get_splitter()
    embedding = _get_embedding()

    raw_docs = _load_file(path)
    chunks = splitter.split_documents(raw_docs)
    for chunk in chunks:
        chunk.metadata["source"] = path.name

    engine = PGEngine.from_connection_string(DATABASE_URL)
    await _ensure_table(engine)

    store = await PGVectorStore.create(
        engine=engine,
        table_name=TABLE_NAME,
        embedding_service=embedding,
    )

    await store.aadd_documents(chunks)
    await _create_fts_index()

    print(f"Ingested {len(chunks)} chunks from {path.name}.")
    await engine.close()
    return len(chunks)


def main():
    parser = argparse.ArgumentParser(description="Ingest documents into pgvector")
    parser.add_argument("--dir", type=str, help="Directory of documents to ingest")
    parser.add_argument("--file", type=str, help="Single file to ingest")
    args = parser.parse_args()

    if args.dir:
        asyncio.run(ingest_directory(args.dir))
    elif args.file:
        asyncio.run(ingest_file(args.file))
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
