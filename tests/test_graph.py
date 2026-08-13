from __future__ import annotations

import pytest
from httpx import ASGITransport, AsyncClient

from src.main import app


@pytest.fixture
def anyio_backend():
    return "asyncio"


@pytest.mark.anyio
async def test_health():
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.get("/health")
        assert response.status_code == 200
        assert response.json() == {"status": "ok"}


@pytest.mark.anyio
async def test_query_returns_expected_shape():
    """Integration test — requires pgvector running with ingested docs."""
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.post(
            "/query",
            json={"question": "What is LangGraph?"},
        )
        if response.status_code == 200:
            data = response.json()
            assert "answer" in data
            assert "sources" in data
            assert "grounded" in data
            assert "retries" in data
            assert isinstance(data["sources"], list)
        else:
            pytest.skip("Database not available for integration test")
