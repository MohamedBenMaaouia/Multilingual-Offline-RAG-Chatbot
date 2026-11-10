"""Day 5 baseline eval harness.

Architectural choice: call the chain directly (via src.chain.get_chain()),
NOT the /chat HTTP endpoint. Justification:
- No HTTP overhead, no uvicorn process management in CI/eval contexts.
- Direct access to result["context"] Documents needed for hit@k and MRR
  (which require metadata, not just the API's SourceRef projection).
- RAGAS integration needs the same chain object; HTTP would add a JSON
  deserialization round-trip for no gain.
- Easier to reproduce — one Python script, no external service to start/stop.

Per ARCHITECTURE.md §6:
- Deterministic metrics (hit@k, MRR, keyword recall) are the primary signal.
- RAGAS faithfulness + answer_relevance are on-demand via --ragas because
  they cost LLM calls and are non-deterministic (LLM-as-judge).
"""
from __future__ import annotations

import argparse
import json
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
from src.vectorstore import get_collection


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


def _safe_mean(values: list[float]) -> float | None:
    vals = [v for v in values if v is not None]
    if not vals:
        return None
    return sum(vals) / len(vals)


# ---------------------------------------------------------------------------
# Per-question evaluation
# ---------------------------------------------------------------------------


def evaluate_question(q: dict[str, Any]) -> dict[str, Any]:
    """Run one question through the chain and compute deterministic metrics.

    Mirrors the production ``/chat`` path exactly: language is resolved from
    the question text the way ``language="auto"`` (the schema default) does,
    then drives both the retrieval bias and the answer-language directive.
    Evaluating an unbiased chain would measure a code path no client uses.
    """
    qid = q["id"]
    question = q["question"]
    expected_sources = q.get("expected_sources") or []
    expected_kw = q.get("expected_answer_contains") or []
    category = q["category"]
    language = q["language"]

    resolved_lang = resolve_language(question)
    chain = get_chain(resolved_lang)

    t0 = time.perf_counter()
    try:
        result = chain.invoke(
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
            "expected_sources": expected_sources,
            "hit_at_k": None if category == "refusal" else False,
            "mrr": None if category == "refusal" else 0.0,
            "keyword_recall": 0.0,
            "keyword_matches": [],
            "keyword_misses": list(expected_kw),
            "refusal_correct": False if category == "refusal" else None,
            "faithfulness": None,
            "answer_relevance": None,
            "latency_ms": int((time.perf_counter() - t0) * 1000),
            "error": str(e),
        }
    latency_ms = int((time.perf_counter() - t0) * 1000)

    answer = result.get("answer", "") or ""
    docs = result.get("context", []) or []
    retrieved_sources = [
        (d.metadata.get("source") if hasattr(d, "metadata") else "") or ""
        for d in docs
    ]

    kw_recall, kw_hits, kw_misses = _keyword_recall(answer, expected_kw)

    is_refusal = category == "refusal"
    if is_refusal:
        hit = None
        mrr = None
        refusal_correct = kw_recall >= 1.0
    else:
        hit = _hit_at_k(retrieved_sources, expected_sources)
        mrr = _mrr(retrieved_sources, expected_sources)
        refusal_correct = None

    return {
        "id": qid,
        "question": question,
        "category": category,
        "language": language,
        "resolved_language": resolved_lang,
        "answer": answer,
        "retrieved_sources": retrieved_sources,
        "expected_sources": expected_sources,
        "hit_at_k": hit,
        "mrr": mrr,
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


def run_ragas(rows: list[dict[str, Any]]) -> None:
    """Mutate rows in place: fill faithfulness + answer_relevance per question.

    Wrapped in try/except — RAGAS' API churns. On any failure, leave the field
    as None and log to stderr. NEVER abort the eval run on RAGAS errors.
    """
    try:
        from ragas import evaluate as ragas_evaluate
        from ragas.metrics import Faithfulness, AnswerRelevancy
        from ragas.llms import LangchainLLMWrapper
        from ragas.embeddings import LangchainEmbeddingsWrapper
        from datasets import Dataset

        from src.llm import get_llm
        from src.embeddings import get_embeddings
    except Exception as e:
        print(f"[ragas] import failed, skipping: {e}", file=sys.stderr)
        return

    # Build the dataset RAGAS expects.
    try:
        records = []
        for r in rows:
            if r["category"] == "refusal":
                # Refusal questions are not meaningfully scorable by RAGAS
                # (the model is supposed to refuse — faithfulness is moot).
                continue
            # We don't have a separate "reference" answer; RAGAS faithfulness
            # only needs (question, answer, contexts). answer_relevance only
            # needs (question, answer). We supply both.
            records.append(
                {
                    "question": r["question"],
                    "answer": r["answer"],
                    "contexts": [],  # filled below
                    "id": r["id"],
                }
            )
    except Exception as e:
        print(f"[ragas] dataset build failed: {e}", file=sys.stderr)
        return

    # We need the original retrieved Documents' text for contexts, which we
    # didn't capture in the row (only the source names). Re-running the chain
    # would double the runtime. Instead, fetch contexts opportunistically by
    # storing them when we run RAGAS — but we lost them after evaluate_question.
    # Compromise: do per-row RAGAS calls so we can use the chain we already have.
    try:
        llm_wrapper = LangchainLLMWrapper(__import__("src.llm", fromlist=["get_llm"]).get_llm())
        emb_wrapper = LangchainEmbeddingsWrapper(
            __import__("src.embeddings", fromlist=["get_embeddings"]).get_embeddings()
        )
    except Exception as e:
        print(f"[ragas] wrapper init failed: {e}", file=sys.stderr)
        return

    # Per-row scoring. We re-invoke the chain (cost: doubles total LLM calls
    # for refusal+non-refusal). This is the trade-off for keeping the main
    # loop simple and avoiding context capture refactoring.
    try:
        chain = get_chain()
    except Exception as e:
        print(f"[ragas] chain init failed: {e}", file=sys.stderr)
        return

    for r in rows:
        if r["category"] == "refusal":
            continue
        try:
            # Re-fetch contexts by invoking the retriever portion only.
            # We reuse the cached singletons so this is just MMR retrieval.
            from src.retriever import get_retriever

            docs = get_retriever().invoke(r["question"])
            contexts = [d.page_content for d in docs]

            ds = Dataset.from_list(
                [
                    {
                        "question": r["question"],
                        "answer": r["answer"],
                        "contexts": contexts,
                    }
                ]
            )
            scores = ragas_evaluate(
                dataset=ds,
                metrics=[Faithfulness(), AnswerRelevancy()],
                llm=llm_wrapper,
                embeddings=emb_wrapper,
                raise_exceptions=False,
            )
            df = scores.to_pandas()
            r["faithfulness"] = float(df["faithfulness"].iloc[0]) if "faithfulness" in df.columns else None
            r["answer_relevance"] = (
                float(df["answer_relevancy"].iloc[0]) if "answer_relevancy" in df.columns else None
            )
        except Exception as e:
            print(f"[ragas] {r['id']} failed: {e}", file=sys.stderr)
            r["faithfulness"] = None
            r["answer_relevance"] = None


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------


def aggregate(rows: list[dict[str, Any]], ragas_enabled: bool) -> dict[str, Any]:
    non_refusal = [r for r in rows if r["category"] != "refusal"]
    refusal = [r for r in rows if r["category"] == "refusal"]

    hit_vals = [1.0 if r["hit_at_k"] else 0.0 for r in non_refusal]
    mrr_vals = [r["mrr"] for r in non_refusal]
    kw_all = [r["keyword_recall"] for r in rows]  # includes refusal
    refusal_vals = [1.0 if r["refusal_correct"] else 0.0 for r in refusal]

    by_category: dict[str, dict[str, Any]] = defaultdict(lambda: {"hits": [], "mrrs": [], "kws": [], "n": 0})
    for r in rows:
        bucket = by_category[r["category"]]
        bucket["n"] += 1
        bucket["kws"].append(r["keyword_recall"])
        if r["category"] != "refusal":
            bucket["hits"].append(1.0 if r["hit_at_k"] else 0.0)
            bucket["mrrs"].append(r["mrr"])

    by_lang: dict[str, dict[str, Any]] = defaultdict(lambda: {"hits": [], "mrrs": [], "kws": [], "n": 0})
    for r in rows:
        bucket = by_lang[r["language"]]
        bucket["n"] += 1
        bucket["kws"].append(r["keyword_recall"])
        if r["category"] != "refusal":
            bucket["hits"].append(1.0 if r["hit_at_k"] else 0.0)
            bucket["mrrs"].append(r["mrr"])

    def _bucket_summary(b: dict[str, Any]) -> dict[str, Any]:
        return {
            "n": b["n"],
            "hit_at_k": _safe_mean(b["hits"]) if b["hits"] else None,
            "mrr": _safe_mean(b["mrrs"]) if b["mrrs"] else None,
            "keyword_recall": _safe_mean(b["kws"]),
        }

    agg: dict[str, Any] = {
        "n_questions": len(rows),
        "n_scored_hit_mrr": len(non_refusal),
        "hit_at_k_mean": _safe_mean(hit_vals),
        "mrr_mean": _safe_mean(mrr_vals),
        "keyword_recall_mean": _safe_mean(kw_all),
        "refusal_correct_rate": _safe_mean(refusal_vals),
        "n_refusal": len(refusal),
        "by_category": {k: _bucket_summary(v) for k, v in by_category.items()},
        "by_language": {k: _bucket_summary(v) for k, v in by_lang.items()},
        "latency_ms_mean": _safe_mean([float(r["latency_ms"]) for r in rows]),
    }

    if ragas_enabled:
        agg["faithfulness_mean"] = _safe_mean([r.get("faithfulness") for r in rows])
        agg["answer_relevance_mean"] = _safe_mean([r.get("answer_relevance") for r in rows])

    return agg


# ---------------------------------------------------------------------------
# Stdout summary
# ---------------------------------------------------------------------------


def print_summary(rows: list[dict[str, Any]], agg: dict[str, Any], out_path: Path, collection_count: int) -> None:
    ts_utc = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    ragas_on = "on" if agg.get("faithfulness_mean") is not None or agg.get("answer_relevance_mean") is not None else "off"

    n_nr = agg["n_scored_hit_mrr"]
    n_ref = agg["n_refusal"]
    hit_pct = (agg["hit_at_k_mean"] or 0.0) * 100
    hit_count = sum(1 for r in rows if r["category"] != "refusal" and r["hit_at_k"])
    kw_pct = (agg["keyword_recall_mean"] or 0.0) * 100
    refusal_pct = (agg["refusal_correct_rate"] or 0.0) * 100
    refusal_hits = sum(1 for r in rows if r["category"] == "refusal" and r["refusal_correct"])

    print()
    print(f"=== Day 5 Baseline Eval — {ts_utc} ===")
    print(f"Corpus: {collection_count} chunks  |  Model: {settings.llm_model}  |  RAGAS: {ragas_on}")
    print()
    print("Overall (excluding refusal):")
    print(f"  hit@{settings.top_k}         : {hit_count}/{n_nr} = {hit_pct:.1f}%")
    print(f"  MRR           : {(agg['mrr_mean'] or 0.0):.2f}")
    print(f"  keyword recall: {kw_pct:.1f}%")
    if n_ref:
        print(f"  refusal correct: {refusal_hits}/{n_ref} = {refusal_pct:.0f}%")
    if ragas_on == "on":
        if agg.get("faithfulness_mean") is not None:
            print(f"  faithfulness  : {agg['faithfulness_mean']:.2f}")
        if agg.get("answer_relevance_mean") is not None:
            print(f"  answer relev. : {agg['answer_relevance_mean']:.2f}")
    print()
    print("By category:")
    for cat, b in agg["by_category"].items():
        hit_s = f"{(b['hit_at_k'] * 100):.1f}%" if b["hit_at_k"] is not None else "n/a"
        mrr_s = f"{b['mrr']:.2f}" if b["mrr"] is not None else "n/a"
        kw_s = f"{(b['keyword_recall'] * 100):.0f}%"
        print(f"  {cat:<22} n={b['n']:<3} hit@{settings.top_k}={hit_s:<7} MRR={mrr_s:<5} kw={kw_s}")
    print()
    print("By language:")
    for lang, b in agg["by_language"].items():
        hit_s = f"{(b['hit_at_k'] * 100):.1f}%" if b["hit_at_k"] is not None else "n/a"
        mrr_s = f"{b['mrr']:.2f}" if b["mrr"] is not None else "n/a"
        kw_s = f"{(b['keyword_recall'] * 100):.0f}%"
        print(f"  {lang}  n={b['n']:<3} hit@{settings.top_k}={hit_s:<7} MRR={mrr_s:<5} kw={kw_s}")
    print()

    # Failures (hit@k == False, non-refusal)
    failures = [r for r in rows if r["category"] != "refusal" and r["hit_at_k"] is False]
    if failures:
        print("Failures (hit@k=0):")
        for r in failures:
            preview = ", ".join(r["retrieved_sources"][:3]) or "(none)"
            qprev = r["question"][:60]
            print(f"  {r['id']}  [{r['category']}, {r['language']}]  \"{qprev}\"  -> retrieved {preview}")
        print()

    # Low-keyword-recall warnings (non-refusal, recall < 0.5)
    low_kw = [r for r in rows if r["category"] != "refusal" and r["keyword_recall"] < 0.5]
    if low_kw:
        print("Low keyword recall (<50%, non-refusal):")
        for r in low_kw:
            qprev = r["question"][:60]
            print(f"  {r['id']}  recall={r['keyword_recall']:.2f}  missing={r['keyword_misses']}  q=\"{qprev}\"")
        print()

    print(f"Output: {out_path}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run the deterministic + optional RAGAS eval.")
    parser.add_argument("--ragas", action="store_true", help="Also compute RAGAS faithfulness + answer_relevance (slow).")
    parser.add_argument("--qa-path", type=Path, default=DEFAULT_QA_PATH)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    args = parser.parse_args(argv)

    args.out_dir.mkdir(parents=True, exist_ok=True)

    # Load YAML with explicit UTF-8 (Windows defaults to cp1252 otherwise).
    with open(args.qa_path, encoding="utf-8") as f:
        qa = yaml.safe_load(f)
    if not isinstance(qa, list) or not qa:
        print(f"qa_set.yaml is empty or malformed at {args.qa_path}", file=sys.stderr)
        return 2

    print(f"Loaded {len(qa)} questions from {args.qa_path}")

    # Warm singletons (embedding model, Chroma client, LLM). Per-question
    # chains are resolved by language inside evaluate_question.
    print("Warming chain (embeddings + Chroma + Ollama)...")
    get_chain()
    coll = get_collection()
    collection_count = coll.count()
    print(f"  collection={settings.collection_name} count={collection_count}")

    # Run questions.
    rows: list[dict[str, Any]] = []
    for i, q in enumerate(qa, start=1):
        print(f"  [{i:>2}/{len(qa)}] {q['id']}  ({q['language']}, {q['category']})  {q['question'][:70]}")
        row = evaluate_question(q)
        rows.append(row)
        # Progress signal
        if row["category"] == "refusal":
            tag = f"refusal_ok={row['refusal_correct']}"
        else:
            tag = f"hit={row['hit_at_k']} mrr={row['mrr']:.2f}"
        print(f"        -> {tag} kw={row['keyword_recall']:.2f} ({row['latency_ms']} ms)")

    if args.ragas:
        print("\nRunning RAGAS faithfulness + answer_relevance (slow, on-demand)...")
        run_ragas(rows)

    agg = aggregate(rows, ragas_enabled=args.ragas)

    out = {
        "timestamp_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "config": {
            "top_k": settings.top_k,
            "fetch_k": settings.fetch_k,
            "mmr_lambda": settings.mmr_lambda,
            # Day 6 deferred item: search_type was missing from the dump, so
            # Row 6's similarity-vs-MMR run had to be inferred from filenames.
            "search_type": settings.search_type,
            "language_pref_ratio": settings.language_pref_ratio,
            "chunk_size": settings.chunk_size,
            "chunk_overlap": settings.chunk_overlap,
            "llm_model": settings.llm_model,
            "embedding_model": settings.embedding_model,
            "collection_name": settings.collection_name,
            "collection_count": collection_count,
            "ragas_enabled": bool(args.ragas),
        },
        "aggregates": agg,
        "rows": rows,
    }

    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    out_path = args.out_dir / f"run_{stamp}.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)

    print_summary(rows, agg, out_path, collection_count)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
