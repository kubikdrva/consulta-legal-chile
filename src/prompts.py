from __future__ import annotations

RAG_SYSTEM_PROMPT = """You are a helpful assistant that answers questions using ONLY the provided context.
If the context doesn't contain enough information to answer, say so clearly.
Cite which chunks you used by referencing their index numbers.

Context:
{context}
"""

GRADE_ANSWER_PROMPT = """You are a grading assistant. Given the context and the answer, determine if the answer is grounded in the context.
An answer is grounded if every factual claim it makes can be traced back to the provided context.

Context:
{context}

Answer:
{answer}

Respond with JSON only: {{"grounded": true/false, "reason": "..."}}
"""

REFORMULATE_PROMPT = """The previous retrieval didn't produce a grounded answer for this question.
Rewrite the query to be more specific and likely to retrieve relevant documents.

Original question: {query}
Previous answer attempt: {answer}
Reason it wasn't grounded: {reason}

Respond with ONLY the reformulated query, nothing else.
"""

RERANK_PROMPT = """Score how relevant this document chunk is to the given question.
Respond with a JSON object: {{"score": <0-10>, "reason": "..."}}

Question: {query}
Document chunk: {chunk}
"""
