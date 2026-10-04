from __future__ import annotations

"""Module 4: RAGAS Evaluation — 4 metrics + failure analysis."""

import os, sys, json, math
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")
from dataclasses import dataclass

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import TEST_SET_PATH


@dataclass
class EvalResult:
    question: str
    answer: str
    contexts: list[str]
    ground_truth: str
    faithfulness: float
    answer_relevancy: float
    context_precision: float
    context_recall: float


def load_test_set(path: str = TEST_SET_PATH) -> list[dict]:
    """Load test set from JSON. (Đã implement sẵn)"""
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def evaluate_ragas(questions: list[str], answers: list[str],
                   contexts: list[list[str]], ground_truths: list[str]) -> dict:
    """Run RAGAS evaluation."""
    empty = {m: 0.0 for m in METRICS}
    empty["per_question"] = []
    try:
        from ragas import evaluate
        from ragas.metrics import faithfulness, answer_relevancy, context_precision, context_recall
        from datasets import Dataset

        dataset = Dataset.from_dict({
            "question": questions, "answer": answers,
            "contexts": contexts, "ground_truth": ground_truths,
        })
        from ragas.run_config import RunConfig

        result = evaluate(dataset, metrics=[faithfulness, answer_relevancy,
                                            context_precision, context_recall],
                          run_config=RunConfig(max_workers=4, timeout=180, max_retries=5))
        df = result.to_pandas()

        per_question = [
            EvalResult(
                question=row["question"],
                answer=row["answer"],
                contexts=list(row["contexts"]),
                ground_truth=row["ground_truth"],
                **{m: _to_float(row.get(m)) for m in METRICS},
            )
            for _, row in df.iterrows()
        ]
        # RAGAS yields NaN when the judge LLM fails on a row; average only the valid scores.
        aggregate = {}
        for m in METRICS:
            valid = [v for v in (getattr(r, m) for r in per_question) if not math.isnan(v)]
            aggregate[m] = sum(valid) / len(valid) if valid else 0.0
        return {**aggregate, "per_question": per_question}
    except Exception as e:
        print(f"  ⚠️  RAGAS evaluation failed: {e}")
        return empty


METRICS = ["faithfulness", "answer_relevancy", "context_precision", "context_recall"]

DIAGNOSTIC_TREE = {
    "faithfulness": ("LLM tự bịa câu trả lời ngoài tài liệu (hallucination)",
                     "Thắt chặt system prompt, giảm temperature về 0"),
    "context_recall": ("Hệ thống tìm kiếm bỏ sót đoạn văn đúng",
                       "Cải thiện bước chunking hoặc bổ sung từ khóa BM25"),
    "context_precision": ("Đoạn văn không liên quan bị xếp lên đầu",
                          "Bổ sung Cross-Encoder reranking hoặc lọc theo metadata"),
    "answer_relevancy": ("Câu trả lời bị lệch trọng tâm câu hỏi",
                         "Viết lại prompt hướng dẫn mô hình trả lời trực tiếp hơn"),
}


def _to_float(value) -> float:
    """Convert a RAGAS cell to float; unscored cells stay NaN so they are not mistaken for 0."""
    try:
        return float(value)
    except (TypeError, ValueError):
        return math.nan


def failure_analysis(eval_results: list[EvalResult], bottom_n: int = 10) -> list[dict]:
    """Analyze bottom-N worst questions using Diagnostic Tree."""
    analyzed = []
    for r in eval_results:
        scores = {m: getattr(r, m) for m in METRICS}
        valid = {m: v for m, v in scores.items() if not math.isnan(v)}
        if not valid:
            continue
        avg = sum(valid.values()) / len(valid)
        worst_metric = min(valid, key=valid.get)
        diagnosis, suggested_fix = DIAGNOSTIC_TREE[worst_metric]
        analyzed.append({
            "question": r.question,
            "answer": r.answer,
            "ground_truth": r.ground_truth,
            "avg_score": round(avg, 4),
            "worst_metric": worst_metric,
            "score": round(valid[worst_metric], 4),
            "scores": {m: (None if math.isnan(v) else round(v, 4)) for m, v in scores.items()},
            "diagnosis": diagnosis,
            "suggested_fix": suggested_fix,
        })
    analyzed.sort(key=lambda x: x["avg_score"])
    return analyzed[:bottom_n]


def save_report(results: dict, failures: list[dict], path: str = "reports/ragas_report.json"):
    """Save evaluation report to JSON. (Đã implement sẵn)"""
    parent_dir = os.path.dirname(path)
    if parent_dir:
        os.makedirs(parent_dir, exist_ok=True)
    report = {
        "aggregate": {k: v for k, v in results.items() if k != "per_question"},
        "num_questions": len(results.get("per_question", [])),
        "failures": failures,
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(f"Report saved to {path}")


if __name__ == "__main__":
    test_set = load_test_set()
    print(f"Loaded {len(test_set)} test questions")
    print("Run pipeline.py first to generate answers, then call evaluate_ragas().")
