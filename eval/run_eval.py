"""Evaluation harness: retrieval, answer and (optionally) RAGAS generation metrics.

Architectural choice: call the chain directly (via src.chain.get_chain()),
NOT the /chat HTTP endpoint. Justification:
- No HTTP overhead, no uvicorn process management in CI/eval contexts.
- Direct access to result["context"] Documents needed for the retrieval
  metrics (which require metadata and chunk text, not the API's SourceRef
  projection).
- Easier to reproduce — one Python script, no external service to start/stop.

Metrics
-------
Deterministic (primary signal, every run):
- file hit@k / file MRR — did a chunk from an ``expected_sources`` file come
  back, and how high. File-granular: any row of obd_codes.csv counts.
- passage hit@k / passage MRR — did a chunk containing one of the
  ``expected_chunk_contains`` anchors come back, and how high. This is the
  honest retrieval metric: it fails when the P0420 question retrieves six
  OBD rows that are not P0420.
- keyword recall — share of ``expected_answer_contains`` in the answer.
- refusal correctness — the out-of-corpus question gets the canned line.

LLM-as-judge (on demand, ``--ragas``):
- faithfulness — share of the answer's claims supported by the retrieved
  chunks (the hallucination check).
- answer relevance — does the answer address the question.
Judged by ``settings.ragas_judge_model`` (a different, larger model than the
one under test) against the exact chunks the answer was generated from.
"""
from __future__ import annotations

import argparse
import json
import math
import re
import sys
import time
import traceback
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml

# Ensure UTF-8 console output on Windows even when redirected.
try:
    sys.stdout.reconfigure(encoding="utf-8")  # type: ignore[attr-defined]
except Exception:
    pass

from src.chain import get_chain
from src.config import settings
from src.language import language_name, resolve_language
from src.retriever import get_retriever
from src.vectorstore import count_chunks


ROOT = Path(__file__).resolve().parent.parent
DEFAULT_QA_PATH = ROOT / "eval" / "qa_set.yaml"
DEFAULT_OUT_DIR = ROOT / "eval" / "results"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _source_matches(retrieved: str, expected_list: list[str]) -> bool:
    """endswith() match so 'obd_codes.csv' matches 'data/raw/obd_codes.csv' too."""
    if not expected_list:
        return False
    r = (retrieved or "").replace("\\", "/")
    for e in expected_list:
        e_norm = (e or "").replace("\\", "/")
        if r == e_norm or r.endswith(e_norm):
            return True
    return False


def _hit_at_k(retrieved_sources: list[str], expected_sources: list[str]) -> bool:
    """True iff any retrieved chunk's source matches any expected source.

    If expected_sources == [], hit is True iff at least one chunk was retrieved.
    """
    if not expected_sources:
        return len(retrieved_sources) > 0
    return any(_source_matches(r, expected_sources) for r in retrieved_sources)


def _mrr(retrieved_sources: list[str], expected_sources: list[str]) -> float:
    """Reciprocal rank of the first matching source (1-indexed). 0 if no match.

    If expected_sources == [], MRR = 1.0 if any retrieval occurred, else 0.
    """
    if not expected_sources:
        return 1.0 if retrieved_sources else 0.0
    for i, r in enumerate(retrieved_sources, start=1):
        if _source_matches(r, expected_sources):
            return 1.0 / i
    return 0.0


def _normalize(text: str) -> str:
    """Lowercase and collapse whitespace — PDF text breaks lines mid-phrase."""
    return re.sub(r"\s+", " ", text or "").strip().lower()


def _passage_rank(contexts: list[str], anchors: list[str]) -> int | None:
    """1-based rank of the first chunk containing any anchor; None if none does."""
    needles = [_normalize(a) for a in anchors if a]
    for i, chunk in enumerate(contexts, start=1):
        text = _normalize(chunk)
        if any(n in text for n in needles):
            return i
    return None


def _keyword_recall(
    answer: str, expected_substrings: list[str]
) -> tuple[float, list[str], list[str]]:
    """Case-insensitive substring recall. Returns (recall, hits, misses)."""
    if not expected_substrings:
        return 1.0, [], []
    answer_lc = (answer or "").lower()
    hits: list[str] = []
    misses: list[str] = []
    for s in expected_substrings:
        if s.lower() in answer_lc:
            hits.append(s)
        else:
            misses.append(s)
    return len(hits) / len(expected_substrings), hits, misses


def _safe_mean(values: list[float | None]) -> float | None:
    vals = [v for v in values if v is not None]
    if not vals:
        return None
    return sum(vals) / len(vals)


def _score_or_none(value: Any) -> float | None:
    """RAGAS reports failed judgements as NaN; store them as None."""
    try:
        f = float(value)
    except (TypeError, ValueError):
        return None
    return None if math.isnan(f) else f


# ---------------------------------------------------------------------------
# Per-question evaluation
# ---------------------------------------------------------------------------


def evaluate_question(q: dict[str, Any], retrieval_only: bool = False) -> dict[str, Any]:
    """Run one question through the chain and compute deterministic metrics.

    Mirrors the production ``/chat`` path exactly: language is resolved from
    the question text the way ``language="auto"`` (the schema default) does,
    then drives both the retrieval bias and the answer-language directive.
    Evaluating an unbiased chain would measure a code path no client uses.

    ``retrieval_only`` runs the same retriever the chain uses but skips the
    LLM: seconds instead of minutes, no Ollama needed. Answer metrics
    (keyword recall, refusal) are then None.
    """
    qid = q["id"]
    question = q["question"]
    expected_sources = q.get("expected_sources") or []
    expected_kw = q.get("expected_answer_contains") or []
    anchors = q.get("expected_chunk_contains") or []
    category = q["category"]
    language = q["language"]
    is_refusal = category == "refusal"

    resolved_lang = resolve_language(question)

    t0 = time.perf_counter()
    try:
        if retrieval_only:
            result = {"answer": None, "context": get_retriever(resolved_lang).invoke(question)}
        else:
            result = get_chain(resolved_lang).invoke(
                {"input": question, "answer_language": language_name(resolved_lang)}
            )
    except Exception as e:  # pragma: no cover - operational
        print(f"[{qid}] chain.invoke FAILED: {e}", file=sys.stderr)
        traceback.print_exc(file=sys.stderr)
        return {
            "id": qid,
            "question": question,
            "category": category,
            "language": language,
            "answer": f"<error: {e}>",
            "retrieved_sources": [],
            "retrieved_contexts": [],
            "expected_sources": expected_sources,
            "hit_at_k": None if is_refusal else False,
            "mrr": None if is_refusal else 0.0,
            "passage_rank": None,
            "passage_hit_at_k": None if is_refusal or not anchors else False,
            "passage_mrr": None if is_refusal or not anchors else 0.0,
            "keyword_recall": 0.0,
            "keyword_matches": [],
            "keyword_misses": list(expected_kw),
            "refusal_correct": False if is_refusal else None,
            "faithfulness": None,
            "answer_relevance": None,
            "latency_ms": int((time.perf_counter() - t0) * 1000),
            "error": str(e),
        }
    latency_ms = int((time.perf_counter() - t0) * 1000)

    answer = result.get("answer")
    docs = result.get("context", []) or []
    retrieved_sources = [(d.metadata or {}).get("source", "") or "" for d in docs]
    retrieved_contexts = [d.page_content or "" for d in docs]

    if retrieval_only:
        kw_recall, kw_hits, kw_misses = None, [], []
    else:
        answer = answer or ""
        kw_recall, kw_hits, kw_misses = _keyword_recall(answer, expected_kw)

    if is_refusal:
        hit = mrr = None
        p_rank = p_hit = p_mrr = None
        refusal_correct = None if retrieval_only else kw_recall >= 1.0
    else:
        hit = _hit_at_k(retrieved_sources, expected_sources)
        mrr = _mrr(retrieved_sources, expected_sources)
        refusal_correct = None
        if anchors:
            p_rank = _passage_rank(retrieved_contexts, anchors)
            p_hit = p_rank is not None
            p_mrr = 1.0 / p_rank if p_rank else 0.0
        else:
            p_rank = p_hit = p_mrr = None

    return {
        "id": qid,
        "question": question,
        "category": category,
        "language": language,
        "resolved_language": resolved_lang,
        "answer": answer,
        "retrieved_sources": retrieved_sources,
        "retrieved_contexts": retrieved_contexts,
        "expected_sources": expected_sources,
        "expected_chunk_contains": anchors,
        "hit_at_k": hit,
        "mrr": mrr,
        "passage_rank": p_rank,
        "passage_hit_at_k": p_hit,
        "passage_mrr": p_mrr,
        "keyword_recall": kw_recall,
        "keyword_matches": kw_hits,
        "keyword_misses": kw_misses,
        "refusal_correct": refusal_correct,
        "faithfulness": None,
        "answer_relevance": None,
        "latency_ms": latency_ms,
    }


# ---------------------------------------------------------------------------
# RAGAS (optional, off by default)
# ---------------------------------------------------------------------------


def _judge_available() -> bool:
    """True if the judge model is pulled in the local Ollama instance."""
    import httpx

    try:
        r = httpx.get(settings.ollama_url.rstrip("/") + "/api/tags", timeout=5.0)
        names = {m.get("name", "") for m in r.json().get("models", [])}
    except Exception as e:
        print(f"[ragas] cannot reach Ollama at {settings.ollama_url}: {e}", file=sys.stderr)
        return False
    wanted = settings.ragas_judge_model
    if wanted in names or f"{wanted}:latest" in names:
        return True
    print(
        f"[ragas] judge model '{wanted}' is not pulled. Run: ollama pull {wanted}",
        file=sys.stderr,
    )
    return False


def run_ragas(rows: list[dict[str, Any]]) -> None:
    """Fill ``faithfulness`` and ``answer_relevance`` on each scorable row, in place.

    Scores the exact chunks each answer was generated from (captured in
    ``retrieved_contexts``), never a fresh retrieval. Refusal rows and rows
    that errored are skipped. Any RAGAS failure leaves the fields as None and
    never aborts the run — the deterministic results are already on disk.
    """
    try:
        from langchain_ollama import ChatOllama
        from ragas import EvaluationDataset
        from ragas import evaluate as ragas_evaluate
        from ragas.embeddings import LangchainEmbeddingsWrapper
        from ragas.llms import LangchainLLMWrapper
        from ragas.metrics import Faithfulness, ResponseRelevancy
        from ragas.run_config import RunConfig

        from src.embeddings import get_embeddings
    except Exception as e:
        print(f"[ragas] import failed, skipping: {e}", file=sys.stderr)
        return

    if not _judge_available():
        return

    scored = [
        r for r in rows
        if r["category"] != "refusal" and not r.get("error") and r.get("retrieved_contexts")
    ]
    if not scored:
        print("[ragas] nothing to score", file=sys.stderr)
        return

    judge = ChatOllama(
        model=settings.ragas_judge_model,
        base_url=settings.ollama_url,
        temperature=0.0,
        num_ctx=settings.ragas_judge_num_ctx,
        timeout=settings.ragas_judge_timeout_s,
    )
    dataset = EvaluationDataset.from_list(
        [
            {
                "user_input": r["question"],
                "response": r["answer"],
                "retrieved_contexts": r["retrieved_contexts"],
            }
            for r in scored
        ]
    )
    # strictness=1: one generated question per answer instead of three —
    # a third of the judge calls, which matters with a 14B model on CPU.
    # max_workers=1: a local Ollama serves one request at a time; parallel
    # calls only queue up and hit the timeout.
    try:
        result = ragas_evaluate(
            dataset=dataset,
            metrics=[Faithfulness(), ResponseRelevancy(strictness=1)],
            llm=LangchainLLMWrapper(judge),
            embeddings=LangchainEmbeddingsWrapper(get_embeddings()),
            run_config=RunConfig(
                timeout=settings.ragas_judge_timeout_s, max_workers=1, max_retries=2
            ),
            raise_exceptions=False,
            show_progress=True,
        )
        df = result.to_pandas()
    except Exception as e:
        print(f"[ragas] evaluation failed: {e}", file=sys.stderr)
        traceback.print_exc(file=sys.stderr)
        return

    for r, (_, rec) in zip(scored, df.iterrows()):
        r["faithfulness"] = _score_or_none(rec.get("faithfulness"))
        r["answer_relevance"] = _score_or_none(rec.get("answer_relevancy"))


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------


def _new_bucket() -> dict[str, Any]:
    return {"hits": [], "mrrs": [], "p_hits": [], "p_mrrs": [], "kws": [], "n": 0}


def _add_to_bucket(bucket: dict[str, Any], r: dict[str, Any]) -> None:
    bucket["n"] += 1
    if r["category"] == "refusal":
        return
    bucket["kws"].append(r["keyword_recall"])
    bucket["hits"].append(1.0 if r["hit_at_k"] else 0.0)
    bucket["mrrs"].append(r["mrr"])
    if r.get("passage_hit_at_k") is not None:
        bucket["p_hits"].append(1.0 if r["passage_hit_at_k"] else 0.0)
        bucket["p_mrrs"].append(r["passage_mrr"])


def _bucket_summary(b: dict[str, Any]) -> dict[str, Any]:
    return {
        "n": b["n"],
        "hit_at_k": _safe_mean(b["hits"]),
        "mrr": _safe_mean(b["mrrs"]),
        "passage_hit_at_k": _safe_mean(b["p_hits"]),
        "passage_mrr": _safe_mean(b["p_mrrs"]),
        "keyword_recall": _safe_mean(b["kws"]),
    }


def aggregate(rows: list[dict[str, Any]], ragas_enabled: bool) -> dict[str, Any]:
    non_refusal = [r for r in rows if r["category"] != "refusal"]
    refusal = [r for r in rows if r["category"] == "refusal"]
    passage_scored = [r for r in non_refusal if r.get("passage_hit_at_k") is not None]

    by_category: dict[str, dict[str, Any]] = defaultdict(_new_bucket)
    by_lang: dict[str, dict[str, Any]] = defaultdict(_new_bucket)
    for r in rows:
        _add_to_bucket(by_category[r["category"]], r)
        _add_to_bucket(by_lang[r["language"]], r)

    agg: dict[str, Any] = {
        "n_questions": len(rows),
        "n_scored_hit_mrr": len(non_refusal),
        "n_scored_passage": len(passage_scored),
        "hit_at_k_mean": _safe_mean([1.0 if r["hit_at_k"] else 0.0 for r in non_refusal]),
        "mrr_mean": _safe_mean([r["mrr"] for r in non_refusal]),
        "passage_hit_at_k_mean": _safe_mean(
            [1.0 if r["passage_hit_at_k"] else 0.0 for r in passage_scored]
        ),
        "passage_mrr_mean": _safe_mean([r["passage_mrr"] for r in passage_scored]),
        # Non-refusal only: the refusal question scores 1.0 whenever the model
        # refuses, which inflated the old all-rows mean by ~2 points.
        "keyword_recall_mean": _safe_mean([r["keyword_recall"] for r in non_refusal]),
        # All 21 rows, kept so runs stay comparable with pre-Nov-2026 results.
        "keyword_recall_mean_all_rows": _safe_mean([r["keyword_recall"] for r in rows]),
        "refusal_correct_rate": _safe_mean(
            [None if r["refusal_correct"] is None else float(r["refusal_correct"]) for r in refusal]
        ),
        "n_refusal": len(refusal),
        "by_category": {k: _bucket_summary(v) for k, v in by_category.items()},
        "by_language": {k: _bucket_summary(v) for k, v in by_lang.items()},
        "latency_ms_mean": _safe_mean([float(r["latency_ms"]) for r in rows]),
    }

    if ragas_enabled:
        faith = [r.get("faithfulness") for r in non_refusal]
        relev = [r.get("answer_relevance") for r in non_refusal]
        agg["faithfulness_mean"] = _safe_mean(faith)
        agg["answer_relevance_mean"] = _safe_mean(relev)
        agg["n_ragas_scored"] = sum(1 for f in faith if f is not None)

    return agg


# ---------------------------------------------------------------------------
# Stdout summary
# ---------------------------------------------------------------------------


def _pct(x: float | None) -> str:
    return "n/a" if x is None else f"{x * 100:.1f}%"


def _num(x: float | None) -> str:
    return "n/a" if x is None else f"{x:.2f}"


def print_summary(rows: list[dict[str, Any]], agg: dict[str, Any], out_path: Path, collection_count: int) -> None:
    ts_utc = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    k = settings.top_k
    ragas_on = "faithfulness_mean" in agg

    print()
    print(f"=== RAG eval — {ts_utc} ===")
    print(
        f"Corpus: {collection_count} chunks  |  Model: {settings.llm_model}  |  "
        f"search: {settings.retrieval_mode}/{settings.search_type}, k={k}  |  "
        f"RAGAS: {'on' if ragas_on else 'off'}"
    )
    print()
    print(f"Retrieval ({agg['n_scored_hit_mrr']} questions, refusal excluded):")
    print(f"  passage hit@{k}  : {_pct(agg['passage_hit_at_k_mean'])}   <- the right chunk came back")
    print(f"  passage MRR     : {_num(agg['passage_mrr_mean'])}")
    print(f"  file hit@{k}     : {_pct(agg['hit_at_k_mean'])}   <- any chunk from the right file")
    print(f"  file MRR        : {_num(agg['mrr_mean'])}")
    print("Answers:")
    print(f"  keyword recall  : {_pct(agg['keyword_recall_mean'])}  (all rows: {_pct(agg['keyword_recall_mean_all_rows'])})")
    if agg["refusal_correct_rate"] is not None:
        n_ok = sum(1 for r in rows if r["category"] == "refusal" and r["refusal_correct"])
        print(f"  refusal correct : {n_ok}/{agg['n_refusal']}")
    if ragas_on:
        print(f"  faithfulness    : {_num(agg['faithfulness_mean'])}  ({agg['n_ragas_scored']} scored, judge {settings.ragas_judge_model})")
        print(f"  answer relevance: {_num(agg['answer_relevance_mean'])}")
    print()

    def _print_buckets(title: str, buckets: dict[str, Any]) -> None:
        print(title)
        for name, b in buckets.items():
            print(
                f"  {name:<22} n={b['n']:<3} passage hit={_pct(b['passage_hit_at_k']):<7} "
                f"file hit={_pct(b['hit_at_k']):<7} kw={_pct(b['keyword_recall'])}"
            )
        print()

    _print_buckets("By category:", agg["by_category"])
    _print_buckets("By language:", agg["by_language"])

    misses = [r for r in rows if r.get("passage_hit_at_k") is False]
    if misses:
        print("Passage misses (the answer passage was not retrieved):")
        for r in misses:
            print(f"  {r['id']}  [{r['category']}, {r['language']}]  \"{r['question'][:60]}\"")
        print()

    low_kw = [
        r for r in rows
        if r["category"] != "refusal" and r["keyword_recall"] is not None and r["keyword_recall"] < 0.5
    ]
    if low_kw:
        print("Low keyword recall (<50%):")
        for r in low_kw:
            print(f"  {r['id']}  recall={r['keyword_recall']:.2f}  missing={r['keyword_misses']}")
        print()

    print(f"Output: {out_path}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def _write(out_path: Path, out: dict[str, Any]) -> None:
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run the deterministic + optional RAGAS eval.")
    parser.add_argument("--ragas", action="store_true", help="Also compute RAGAS faithfulness + answer relevance (slow).")
    parser.add_argument(
        "--retrieval-only",
        action="store_true",
        help="Score retrieval only: no LLM, no Ollama, seconds instead of minutes.",
    )
    parser.add_argument("--qa-path", type=Path, default=DEFAULT_QA_PATH)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    args = parser.parse_args(argv)
    if args.ragas and args.retrieval_only:
        parser.error("--ragas needs generated answers; it cannot be combined with --retrieval-only")

    args.out_dir.mkdir(parents=True, exist_ok=True)

    # Load YAML with explicit UTF-8 (Windows defaults to cp1252 otherwise).
    with open(args.qa_path, encoding="utf-8") as f:
        qa = yaml.safe_load(f)
    if not isinstance(qa, list) or not qa:
        print(f"qa_set.yaml is empty or malformed at {args.qa_path}", file=sys.stderr)
        return 2

    print(f"Loaded {len(qa)} questions from {args.qa_path}")

    # Warm singletons (embedding models, Qdrant client, LLM). Per-question
    # chains are resolved by language inside evaluate_question.
    if args.retrieval_only:
        print("Retrieval-only mode: no LLM calls, answer metrics are skipped.")
        get_retriever()
    else:
        print("Warming chain (embeddings + vector store + Ollama)...")
        get_chain()
    collection_count = count_chunks()
    print(f"  collection={settings.collection_name} count={collection_count}")

    rows: list[dict[str, Any]] = []
    for i, q in enumerate(qa, start=1):
        print(f"  [{i:>2}/{len(qa)}] {q['id']}  ({q['language']}, {q['category']})  {q['question'][:70]}")
        row = evaluate_question(q, retrieval_only=args.retrieval_only)
        rows.append(row)
        if row["category"] == "refusal":
            tag = f"refusal_ok={row['refusal_correct']}"
        else:
            tag = f"passage_rank={row['passage_rank']} file_mrr={row['mrr']:.2f}"
        kw = "" if row["keyword_recall"] is None else f" kw={row['keyword_recall']:.2f}"
        print(f"        -> {tag}{kw} ({row['latency_ms']} ms)")

    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    out_path = args.out_dir / f"run_{stamp}.json"
    out: dict[str, Any] = {
        "timestamp_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "config": {
            "top_k": settings.top_k,
            "fetch_k": settings.fetch_k,
            "mmr_lambda": settings.mmr_lambda,
            "vector_store": "qdrant",
            "retrieval_mode": settings.retrieval_mode,
            "search_type": settings.search_type,
            "language_pref_ratio": settings.language_pref_ratio,
            "chunk_size": settings.chunk_size,
            "chunk_overlap": settings.chunk_overlap,
            "llm_model": settings.llm_model,
            "embedding_model": settings.embedding_model,
            "collection_name": settings.collection_name,
            "collection_count": collection_count,
            "retrieval_only": bool(args.retrieval_only),
            "ragas_enabled": bool(args.ragas),
            "ragas_judge_model": settings.ragas_judge_model if args.ragas else None,
        },
        "aggregates": aggregate(rows, ragas_enabled=False),
        "rows": rows,
    }
    # Save before RAGAS: a slow or failing judge run must not lose the
    # generation results it took ~7 minutes to produce.
    _write(out_path, out)

    if args.ragas:
        print("\nRunning RAGAS faithfulness + answer relevance (slow, on-demand)...")
        run_ragas(rows)
        out["aggregates"] = aggregate(rows, ragas_enabled=True)
        _write(out_path, out)

    print_summary(rows, out["aggregates"], out_path, collection_count)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
