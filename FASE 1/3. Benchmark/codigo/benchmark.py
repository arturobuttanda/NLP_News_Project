"""Batch retrieval and integration with the instructor's unmodified evaluator."""
import hashlib
import itertools
import json
from pathlib import Path
import subprocess
import sys

import pandas as pd


def run_benchmark(args, engine_factory=None):
    if engine_factory is None:
        from information_retrieval import InformationRetrievalEngine
        engine_factory = InformationRetrievalEngine

    queries = pd.read_csv(args.benchmark, sep="\t", dtype=str).fillna("")
    if not {"query_id", "difficulty_level", "query"} <= set(queries.columns):
        raise ValueError("Queries require query_id, difficulty_level and query columns")
    if queries.difficulty_level.value_counts().to_dict() != {"1": 5, "2": 5, "3": 5}:
        raise ValueError("Assignment requires five queries per difficulty level (1, 2, 3)")
    if queries.query_id.duplicated().any() or queries[["query_id", "query"]].eq("").any().any():
        raise ValueError("Query IDs must be unique and queries must not be empty")
    engine = engine_factory()
    if args.qrels:
        judgments = pd.read_csv(args.qrels, sep="\t", dtype={"query_id": str, "document_id": str})
        if not {"query_id", "document_id", "relevance"} <= set(judgments.columns):
            raise ValueError("Qrels require query_id, document_id, relevance")
        if set(judgments.query_id) != set(queries.query_id):
            raise ValueError("Queries and qrels must contain exactly the same query IDs")
        if set(judgments.document_id) - set(engine.df_raw.id.astype(str)):
            raise ValueError("Qrels contain document IDs absent from this corpus")
        counts = judgments[judgments.relevance > 0].groupby("query_id").size()
        if len(queries) != 15 or not counts.reindex(queries.query_id, fill_value=0).between(9, 15).all():
            raise ValueError("Benchmark requires 15 queries and 9–15 relevant documents per query")
        if not judgments.relevance.isin([0, 1, 2]).all() or judgments.duplicated(["query_id", "document_id"]).any():
            raise ValueError("Invalid or duplicate relevance judgments")
        if not judgments.groupby("query_id").relevance.apply(lambda x: {1, 2} <= set(x)).all():
            raise ValueError("Each query must include both partially and highly relevant documents")
    if args.top_k != 10 and args.qrels:
        raise ValueError("The instructor evaluator requires top_k=10")
    configs = list(itertools.product(["title", "description"], ["unigram", "bigram", "both"], ["frequency", "binary", "tfidf"])) if args.all_configs else [(args.content, args.feature, args.representation)]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    summary = []
    evaluator = Path(__file__).resolve().parent / "evaluate_ir_metrics.py"
    for content, feature, representation in configs:
        name = f"{content}_{feature}_{representation}"
        target = args.output_dir / name
        target.mkdir(parents=True, exist_ok=True)
        ranking_rows, detail_rows, query_rows = [], [], []
        for item in queries.itertuples(index=False):
            result = engine.search(item.query, content, feature, representation, args.top_k)
            query_rows.append({"query_id": item.query_id, "query": item.query, "normalized_query": result["normalized_query"], "elapsed_seconds": result["elapsed_seconds"], "vocab_size": result["vocab_size"]})
            for hit in result["results"]:
                ranking_rows.append({"query_id": item.query_id, "document_id": hit["id"], "rank": hit["rank"], "score": hit["score"]})
                detail_rows.append({"query_id": item.query_id, "query": item.query, **hit})
        ranking_path = target / "rankings.tsv"
        pd.DataFrame(ranking_rows).to_csv(ranking_path, sep="\t", index=False)
        pd.DataFrame(detail_rows).to_csv(target / "query_results.tsv", sep="\t", index=False)
        pd.DataFrame(query_rows).to_csv(target / "normalized_queries.tsv", sep="\t", index=False)
        if args.qrels:
            completed = subprocess.run([sys.executable, str(evaluator), str(ranking_path), str(args.qrels.resolve()), "--output", str(target / "evaluation_results.tsv")], capture_output=True, text=True)
            (target / "evaluator_output.txt").write_text(completed.stdout + completed.stderr, encoding="utf-8")
            if completed.returncode:
                raise RuntimeError(completed.stdout + completed.stderr)
            metrics = pd.read_csv(target / "evaluation_results.tsv", sep="\t").iloc[-1].to_dict()
            summary.append({"configuration": name, **metrics})
        print(f"[OK] {name}: {len(ranking_rows)} ranked documents")
    if summary:
        pd.DataFrame(summary).to_csv(args.output_dir / "comparison.tsv", sep="\t", index=False)
    inputs = [args.benchmark, engine.raw_data_path, engine.data_path] + ([args.qrels] if args.qrels else [])
    manifest = {"python": sys.version, "configurations": configs, "top_k": args.top_k, "documents": len(engine.df_raw), "queries": len(queries), "sha256": {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in inputs}}
    (args.output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
