from __future__ import annotations

import json
from typing import TypedDict

from langchain_core.documents import Document
from langchain_openai import ChatOpenAI
from langgraph.graph import END, START, StateGraph

from src.config import LLM_MODEL, MAX_RETRIES
from src.prompts import GRADE_ANSWER_PROMPT, RAG_SYSTEM_PROMPT, REFORMULATE_PROMPT
from src.retriever import hybrid_retrieve, rerank


class AgentState(TypedDict):
    query: str
    reformulated_query: str
    documents: list[Document]
    answer: str
    is_grounded: bool
    grade_reason: str
    retry_count: int
    sources: list[dict]


async def retrieve_context(state: AgentState) -> dict:
    query = state.get("reformulated_query") or state["query"]
    documents = await hybrid_retrieve(query)
    return {"documents": documents}


async def rerank_documents(state: AgentState) -> dict:
    query = state.get("reformulated_query") or state["query"]
    documents = state["documents"]
    reranked = await rerank(query, documents)
    sources = [
        {
            "text": doc.page_content[:300],
            "source": doc.metadata.get("source", "unknown"),
        }
        for doc in reranked
    ]
    return {"documents": reranked, "sources": sources}


async def generate_answer(state: AgentState) -> dict:
    query = state["query"]
    documents = state["documents"]

    context = "\n\n---\n\n".join(
        f"[Chunk {i}] (source: {doc.metadata.get('source', 'unknown')})\n{doc.page_content}"
        for i, doc in enumerate(documents)
    )

    llm = ChatOpenAI(model=LLM_MODEL, temperature=0)
    messages = [
        {"role": "system", "content": RAG_SYSTEM_PROMPT.format(context=context)},
        {"role": "user", "content": query},
    ]
    response = await llm.ainvoke(messages)
    return {"answer": response.content}


async def grade_answer(state: AgentState) -> dict:
    documents = state["documents"]
    answer = state["answer"]

    context = "\n\n".join(doc.page_content for doc in documents)
    prompt = GRADE_ANSWER_PROMPT.format(context=context, answer=answer)

    llm = ChatOpenAI(model=LLM_MODEL, temperature=0)
    response = await llm.ainvoke(prompt)

    try:
        result = json.loads(response.content)
        grounded = result.get("grounded", False)
        reason = result.get("reason", "")
    except (json.JSONDecodeError, ValueError):
        grounded = False
        reason = "Failed to parse grading response"

    return {"is_grounded": grounded, "grade_reason": reason}


def should_retry(state: AgentState) -> str:
    if state["is_grounded"]:
        return "end"
    if state["retry_count"] >= MAX_RETRIES:
        return "end"
    return "reformulate"


async def reformulate_query(state: AgentState) -> dict:
    prompt = REFORMULATE_PROMPT.format(
        query=state["query"],
        answer=state["answer"],
        reason=state["grade_reason"],
    )
    llm = ChatOpenAI(model=LLM_MODEL, temperature=0.3)
    response = await llm.ainvoke(prompt)
    return {
        "reformulated_query": response.content.strip(),
        "retry_count": state["retry_count"] + 1,
    }


def build_graph() -> StateGraph:
    workflow = StateGraph(AgentState)

    workflow.add_node("retrieve-context", retrieve_context)
    workflow.add_node("rerank-documents", rerank_documents)
    workflow.add_node("generate-answer", generate_answer)
    workflow.add_node("grade-answer", grade_answer)
    workflow.add_node("reformulate-query", reformulate_query)

    workflow.add_edge(START, "retrieve-context")
    workflow.add_edge("retrieve-context", "rerank-documents")
    workflow.add_edge("rerank-documents", "generate-answer")
    workflow.add_edge("generate-answer", "grade-answer")

    workflow.add_conditional_edges(
        "grade-answer",
        should_retry,
        {
            "end": END,
            "reformulate": "reformulate-query",
        },
    )

    workflow.add_edge("reformulate-query", "retrieve-context")

    return workflow.compile()


graph = build_graph()
