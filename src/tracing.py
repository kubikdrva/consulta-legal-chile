from __future__ import annotations

from langfuse import Langfuse
from langfuse.callback import CallbackHandler

from src.config import (
    ENVIRONMENT,
    LANGFUSE_HOST,
    LANGFUSE_PUBLIC_KEY,
    LANGFUSE_SECRET_KEY,
)

_langfuse_available = bool(LANGFUSE_PUBLIC_KEY and LANGFUSE_SECRET_KEY)

_langfuse_client: Langfuse | None = None


def _get_client() -> Langfuse | None:
    global _langfuse_client
    if not _langfuse_available:
        return None
    if _langfuse_client is None:
        _langfuse_client = Langfuse(
            public_key=LANGFUSE_PUBLIC_KEY,
            secret_key=LANGFUSE_SECRET_KEY,
            host=LANGFUSE_HOST,
        )
    return _langfuse_client


def get_langfuse_handler(
    *,
    trace_name: str = "rag-query",
    session_id: str | None = None,
    user_id: str | None = None,
    tags: list[str] | None = None,
    metadata: dict | None = None,
) -> CallbackHandler | None:
    if not _langfuse_available:
        return None

    all_tags = [ENVIRONMENT, *(tags or [])]
    all_metadata = {"environment": ENVIRONMENT, **(metadata or {})}

    return CallbackHandler(
        public_key=LANGFUSE_PUBLIC_KEY,
        secret_key=LANGFUSE_SECRET_KEY,
        host=LANGFUSE_HOST,
        session_id=session_id,
        user_id=user_id,
        trace_name=trace_name,
        tags=all_tags,
        metadata=all_metadata,
    )


def flush_langfuse():
    client = _get_client()
    if client:
        client.flush()
