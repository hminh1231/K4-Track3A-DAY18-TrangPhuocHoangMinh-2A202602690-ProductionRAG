from __future__ import annotations

"""
Module 5: Enrichment Pipeline
==============================
Làm giàu chunks TRƯỚC khi embed: Summarize, HyQA, Contextual Prepend, Auto Metadata.

Test: pytest tests/test_m5.py
"""

import os, sys, re, json
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")
from dataclasses import dataclass, field

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import OPENAI_API_KEY


@dataclass
class EnrichedChunk:
    """Chunk đã được làm giàu."""
    original_text: str
    enriched_text: str
    summary: str
    hypothesis_questions: list[str]
    auto_metadata: dict
    method: str  # "contextual", "summary", "hyqa", "full"


# ─── Shared LLM helpers ──────────────────────────────────

LLM_MODEL = "gpt-4o-mini"
_client = None


def _chat(system: str, user: str, max_tokens: int, json_mode: bool = False) -> str:
    """Gọi OpenAI chat completion. Raise nếu không có API key hoặc API lỗi."""
    global _client
    if not OPENAI_API_KEY:
        raise RuntimeError("OPENAI_API_KEY is not set")
    if _client is None:
        from openai import OpenAI
        _client = OpenAI(api_key=OPENAI_API_KEY)
    kwargs = {"response_format": {"type": "json_object"}} if json_mode else {}
    resp = _client.chat.completions.create(
        model=LLM_MODEL,
        messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
        max_tokens=max_tokens,
        temperature=0,
        **kwargs,
    )
    return (resp.choices[0].message.content or "").strip()


def _split_sentences(text: str) -> list[str]:
    return [s.strip() for s in re.split(r"(?<=[.!?])\s+|\n+", text) if s.strip()]


# ─── Technique 1: Chunk Summarization ────────────────────


def _summarize_fallback(text: str) -> str:
    sentences = _split_sentences(text)
    return " ".join(sentences[:2]) if sentences else text


def summarize_chunk(text: str) -> str:
    """
    Tạo summary ngắn cho chunk.
    Embed summary thay vì (hoặc cùng với) raw chunk → giảm noise.
    """
    if OPENAI_API_KEY:
        try:
            return _chat("Tóm tắt đoạn văn sau trong 2-3 câu ngắn gọn bằng tiếng Việt.", text, 150)
        except Exception as e:
            print(f"  ⚠️  OpenAI summarize failed: {e}")
    return _summarize_fallback(text)


# ─── Technique 2: Hypothesis Question-Answer (HyQA) ─────


def _hyqa_fallback(text: str, n_questions: int) -> list[str]:
    sentences = [s for s in _split_sentences(text) if len(s) > 10]
    return [f"{s.rstrip('.!?')}?" for s in sentences[:n_questions]]


def generate_hypothesis_questions(text: str, n_questions: int = 3) -> list[str]:
    """
    Generate câu hỏi mà chunk có thể trả lời.
    Index cả questions lẫn chunk → query match tốt hơn (bridge vocabulary gap).
    """
    if OPENAI_API_KEY:
        try:
            content = _chat(
                f"Dựa trên đoạn văn, tạo {n_questions} câu hỏi bằng tiếng Việt mà đoạn văn có thể trả lời. "
                "Trả về mỗi câu hỏi trên 1 dòng, không đánh số.",
                text, 200,
            )
            questions = [q.strip().lstrip("0123456789.-) ").strip() for q in content.split("\n")]
            questions = [q for q in questions if q]
            if questions:
                return questions[:n_questions]
        except Exception as e:
            print(f"  ⚠️  OpenAI HyQA failed: {e}")
    return _hyqa_fallback(text, n_questions)


# ─── Technique 3: Contextual Prepend (Anthropic style) ──


def _context_fallback(document_title: str) -> str:
    return f"Trích từ {document_title}." if document_title else ""


def contextual_prepend(text: str, document_title: str = "") -> str:
    """
    Prepend context giải thích chunk nằm ở đâu trong document.
    Anthropic benchmark: giảm 49% retrieval failure (alone).
    """
    context = ""
    if OPENAI_API_KEY:
        try:
            context = _chat(
                "Viết 1 câu ngắn bằng tiếng Việt mô tả đoạn văn này nằm ở đâu trong tài liệu "
                "và nói về chủ đề gì. Chỉ trả về 1 câu.",
                f"Tài liệu: {document_title}\n\nĐoạn văn:\n{text}", 80,
            )
        except Exception as e:
            print(f"  ⚠️  OpenAI contextual failed: {e}")
    context = context or _context_fallback(document_title)
    return f"{context}\n\n{text}" if context else text


# ─── Technique 4: Auto Metadata Extraction ──────────────

METADATA_SCHEMA = '{"topic": "...", "entities": ["..."], "category": "policy|hr|it|finance", "language": "vi|en"}'


def _metadata_fallback() -> dict:
    return {"topic": "general", "entities": [], "category": "policy", "language": "vi"}


def extract_metadata(text: str) -> dict:
    """
    LLM extract metadata tự động: topic, entities, date_range, category.
    """
    if OPENAI_API_KEY:
        try:
            content = _chat(f"Trích xuất metadata từ đoạn văn. Trả về JSON: {METADATA_SCHEMA}",
                            text, 150, json_mode=True)
            data = json.loads(content)
            if isinstance(data, dict):
                return data
        except Exception as e:
            print(f"  ⚠️  OpenAI metadata failed: {e}")
    return _metadata_fallback()


# ─── Combined Single-Call Mode ───────────────────────────

COMBINED_PROMPT = f"""Phân tích đoạn văn và trả về JSON (nội dung bằng tiếng Việt):
{{
  "summary": "tóm tắt 2-3 câu",
  "questions": ["câu hỏi 1", "câu hỏi 2", "câu hỏi 3"],
  "context": "1 câu mô tả đoạn văn nằm ở đâu trong tài liệu và nói về chủ đề gì",
  "metadata": {METADATA_SCHEMA}
}}"""


def _enrich_fallback(text: str, source: str) -> dict:
    return {
        "summary": _summarize_fallback(text),
        "questions": _hyqa_fallback(text, 3),
        "context": _context_fallback(source),
        "metadata": _metadata_fallback(),
    }


def _enrich_single_call(text: str, source: str) -> dict:
    """Single LLM call to get summary + questions + context + metadata.

    ⚠️ Cost optimization: 1 API call thay vì 4 calls riêng lẻ.
    """
    fallback = _enrich_fallback(text, source)
    if not OPENAI_API_KEY:
        return fallback
    try:
        data = json.loads(_chat(COMBINED_PROMPT, f"Tài liệu: {source}\n\nĐoạn văn:\n{text}",
                                400, json_mode=True))
    except Exception as e:
        print(f"  ⚠️  Enrichment API failed: {e}")
        return fallback
    if not isinstance(data, dict):
        return fallback

    questions = data.get("questions")
    metadata = data.get("metadata")
    return {
        "summary": str(data.get("summary") or fallback["summary"]).strip(),
        "questions": [str(q).strip() for q in questions if str(q).strip()]
                     if isinstance(questions, list) and questions else fallback["questions"],
        "context": str(data.get("context") or fallback["context"]).strip(),
        "metadata": metadata if isinstance(metadata, dict) else fallback["metadata"],
    }


# ─── Full Enrichment Pipeline ────────────────────────────


def enrich_chunks(
    chunks: list[dict],
    methods: list[str] | None = None,
) -> list[EnrichedChunk]:
    """
    Chạy enrichment pipeline trên danh sách chunks. (Đã implement sẵn — dùng functions ở trên)

    Có 2 chế độ:
    - methods cụ thể (["summary"], ["contextual"]...): gọi từng function riêng (tốt cho học/debug)
    - methods=["combined"] hoặc None: 1 API call duy nhất cho tất cả (tốt cho production)

    Args:
        chunks: List of {"text": str, "metadata": dict}
        methods: Default None → combined mode (1 call/chunk).
                 Options: "summary", "hyqa", "contextual", "metadata", "combined"
    """
    if methods is None:
        methods = ["combined"]

    use_combined = "combined" in methods

    enriched = []
    for i, chunk in enumerate(chunks):
        text = chunk["text"]
        source = chunk.get("metadata", {}).get("source", "")

        if use_combined:
            result = _enrich_single_call(text, source)
            summary = result.get("summary", "")
            questions = result.get("questions", [])
            context_line = result.get("context", "")
            enriched_text = f"{context_line}\n\n{text}" if context_line else text
            auto_meta = result.get("metadata", {})
        else:
            summary = summarize_chunk(text) if "summary" in methods else ""
            questions = generate_hypothesis_questions(text) if "hyqa" in methods else []
            enriched_text = contextual_prepend(text, source) if "contextual" in methods else text
            auto_meta = extract_metadata(text) if "metadata" in methods else {}

        enriched.append(EnrichedChunk(
            original_text=text,
            enriched_text=enriched_text,
            summary=summary,
            hypothesis_questions=questions,
            auto_metadata={**chunk.get("metadata", {}), **auto_meta},
            method="+".join(methods),
        ))

        if (i + 1) % 10 == 0 or (i + 1) == len(chunks):
            print(f"  Enriched {i + 1}/{len(chunks)} chunks...", flush=True)

    return enriched


# ─── Main ────────────────────────────────────────────────

if __name__ == "__main__":
    sample = "Nhân viên chính thức được nghỉ phép năm 12 ngày làm việc mỗi năm. Số ngày nghỉ phép tăng thêm 1 ngày cho mỗi 5 năm thâm niên công tác."

    print("=== Enrichment Pipeline Demo ===\n")
    print(f"Original: {sample}\n")

    s = summarize_chunk(sample)
    print(f"Summary: {s}\n")

    qs = generate_hypothesis_questions(sample)
    print(f"HyQA questions: {qs}\n")

    ctx = contextual_prepend(sample, "Sổ tay nhân viên VinUni 2024")
    print(f"Contextual: {ctx}\n")

    meta = extract_metadata(sample)
    print(f"Auto metadata: {meta}")
