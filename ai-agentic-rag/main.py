"""CLI for the agentic RAG demonstration and its evaluations."""
from __future__ import annotations

import argparse
import json
import os
import sys
import uuid
from pathlib import Path

from dotenv import dotenv_values
from openai import APIError, APIConnectionError

from rag import Index, LocalEmbedder, OpenRouter, fact_coverage, load_chunks, retrieval_metrics, run_agent

ROOT = Path(__file__).resolve().parent


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Agentic semantic RAG demo")
    parser.add_argument("--data", type=Path, default=ROOT / "data/knowledge.json")
    parser.add_argument("--eval-data", type=Path, default=ROOT / "data/eval.json")
    parser.add_argument("--env-file", type=Path, default=ROOT / ".env")
    parser.add_argument("--model", help="Override OPENROUTER_MODEL")
    parser.add_argument("--k", type=int, default=3, help="Chunks per search (default: 3)")
    parser.add_argument("--max-searches", type=int, default=3)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "runs")
    sub = parser.add_subparsers(dest="command", required=True)
    ask = sub.add_parser("ask", help="Ask the agent a question")
    ask.add_argument("question")
    sub.add_parser("eval-retrieval", help="Evaluate a single direct retrieval, without LLM calls")
    sub.add_parser("eval-agent", help="Evaluate the agent on the local labeled dataset")
    return parser.parse_args()


def save_report(directory: Path, report: dict) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"{report['type']}-{uuid.uuid4().hex[:12]}.json"
    path.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return path


def run(args: argparse.Namespace) -> int:
    if args.k < 1 or args.max_searches < 1:
        raise ValueError("--k and --max-searches must be positive")
    settings = dotenv_values(args.env_file) if args.env_file.exists() else {}
    key = os.getenv("OPENROUTER_API_KEY") or settings.get("OPENROUTER_API_KEY")
    model = (args.model or os.getenv("OPENROUTER_MODEL") or settings.get("OPENROUTER_MODEL")
             or "nvidia/nemotron-3.5-lightning:free")
    if args.command != "eval-retrieval" and not key:
        raise ValueError("OPENROUTER_API_KEY is required for agent runs")
    print("Loading local embedding model and building index...", flush=True)
    chunks = load_chunks(args.data)
    index = Index(chunks, LocalEmbedder())
    decide = OpenRouter(key, model) if args.command != "eval-retrieval" else None
    if args.command == "ask":
        result = run_agent(args.question, index, decide, max_searches=args.max_searches, k=args.k)
        print(f"\n{result['status'].upper()}: {result['answer']}")
        print("Citations:", ", ".join(result["citations"]) or "(none)")
        print("Searches:", result["search_count"])
        for event in result["trace"]:
            if event["action"] == "search":
                print(f"  query: {event['query']} -> {', '.join(hit['id'] for hit in event['results'])}")
        report = {"type": "ask", "result": result}
    else:
        cases = json.loads(args.eval_data.read_text(encoding="utf-8"))
        if not isinstance(cases, list) or not cases:
            raise ValueError("Evaluation dataset must be a non-empty list")
        known_ids = {chunk.id for chunk in chunks}
        results = []
        for case in cases:
            if not set(case["relevant"]).issubset(known_ids):
                raise ValueError(f"Unknown relevant chunk in case {case['id']}")
            print(f"Evaluating {case['id']}...", flush=True)
            if args.command == "eval-retrieval":
                hits = index.search(case["question"], args.k)
                ranked_ids = [hit.chunk.id for hit in hits]
                evidence_text = " ".join(hit.chunk.text for hit in hits)
                item = {"id": case["id"], "ranked_ids": ranked_ids}
            else:
                agent = run_agent(case["question"], index, decide, max_searches=args.max_searches, k=args.k)
                ranked_ids = agent["retrieved_ids"]
                evidence_text = " ".join(chunk.text for chunk in chunks if chunk.id in ranked_ids)
                item = {"id": case["id"], "agent": agent,
                        "answer_fact_coverage": fact_coverage(agent["answer"], case["facts"])}
            item["retrieval"] = retrieval_metrics(ranked_ids, case["relevant"], args.k)
            item["evidence_fact_coverage"] = fact_coverage(evidence_text, case["facts"])
            results.append(item)
        summary = {
            "cases": len(results),
            "mean_precision_at_k": sum(r["retrieval"]["precision"] for r in results) / len(results),
            "mean_recall_at_k": sum(r["retrieval"]["recall"] for r in results) / len(results),
            "mrr_at_k": sum(r["retrieval"]["reciprocal_rank"] for r in results) / len(results),
            "mean_evidence_fact_coverage": sum(r["evidence_fact_coverage"] for r in results) / len(results),
        }
        if args.command == "eval-agent":
            summary["mean_answer_fact_coverage"] = sum(r["answer_fact_coverage"] for r in results) / len(results)
        report = {"type": args.command, "k": args.k, "summary": summary, "results": results}
        print(json.dumps(summary, indent=2))
    print(f"Report: {save_report(args.output_dir, report)}")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(run(arguments()))
    except (ValueError, KeyError, TypeError, OSError, APIError, APIConnectionError) as error:
        print(f"error: {error}", file=sys.stderr)
        sys.exit(1)
