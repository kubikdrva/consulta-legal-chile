from __future__ import annotations

import asyncio
import json
from pathlib import Path

from langchain_openai import ChatOpenAI
from ragas import EvaluationDataset, SingleTurnSample, evaluate
from ragas.metrics import (
    Faithfulness,
    ResponseRelevancy,
    LLMContextPrecisionWithoutReference,
    LLMContextRecall,
)

from src.config import LLM_MODEL
from src.graph import graph


async def run_rag(question: str) -> dict:
    initial_state = {
        "query": question,
        "reformulated_query": "",
        "documents": [],
        "answer": "",
        "is_grounded": False,
        "grade_reason": "",
        "retry_count": 0,
        "sources": [],
    }
    result = await graph.ainvoke(initial_state)
    contexts = [doc.page_content for doc in result["documents"]]
    return {
        "answer": result["answer"],
        "contexts": contexts,
    }


async def build_dataset() -> EvaluationDataset:
    golden_path = Path(__file__).parent / "golden_qa.json"
    with open(golden_path) as f:
        golden_pairs = json.load(f)

    samples = []
    for pair in golden_pairs:
        print(f"  Running: {pair['question'][:60]}...")
        result = await run_rag(pair["question"])
        samples.append(
            SingleTurnSample(
                user_input=pair["question"],
                response=result["answer"],
                retrieved_contexts=result["contexts"],
                reference=pair["ground_truth"],
            )
        )

    return EvaluationDataset(samples=samples)


async def helpfulness_judge(question: str, answer: str) -> dict:
    llm = ChatOpenAI(model=LLM_MODEL, temperature=0)
    prompt = f"""Rate the helpfulness of this answer on a scale of 1-5.
1 = Not helpful at all
2 = Slightly helpful
3 = Moderately helpful
4 = Very helpful
5 = Extremely helpful

Question: {question}
Answer: {answer}

Respond with JSON: {{"score": <1-5>, "reason": "..."}}"""

    response = await llm.ainvoke(prompt)
    try:
        return json.loads(response.content)
    except json.JSONDecodeError:
        return {"score": 0, "reason": "Failed to parse"}


async def main():
    print("Building evaluation dataset (running RAG on each golden QA pair)...")
    dataset = await build_dataset()

    print("\nRunning RAGAS evaluation...")
    metrics = [
        Faithfulness(),
        ResponseRelevancy(),
        LLMContextPrecisionWithoutReference(),
        LLMContextRecall(),
    ]

    results = evaluate(dataset=dataset, metrics=metrics)
    print("\n=== RAGAS Results ===")
    print(results)

    print("\n=== Custom Helpfulness Judge ===")
    golden_path = Path(__file__).parent / "golden_qa.json"
    with open(golden_path) as f:
        golden_pairs = json.load(f)

    helpfulness_scores = []
    for i, sample in enumerate(dataset.samples):
        score = await helpfulness_judge(
            golden_pairs[i]["question"], sample.response
        )
        helpfulness_scores.append(score["score"])
        print(f"  Q: {golden_pairs[i]['question'][:50]}... -> {score['score']}/5")

    avg = sum(helpfulness_scores) / len(helpfulness_scores) if helpfulness_scores else 0
    print(f"\n  Average helpfulness: {avg:.2f}/5")

    results_dir = Path(__file__).parent / "results"
    results_dir.mkdir(exist_ok=True)
    output = {
        "ragas": results.to_pandas().to_dict() if hasattr(results, "to_pandas") else str(results),
        "helpfulness_avg": avg,
    }
    with open(results_dir / "eval_results.json", "w") as f:
        json.dump(output, f, indent=2, default=str)
    print(f"\nResults saved to eval/results/eval_results.json")


if __name__ == "__main__":
    asyncio.run(main())
