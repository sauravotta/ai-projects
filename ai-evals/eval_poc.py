#!/usr/bin/env python3
"""Local-first evaluation demo with direct checks and an optional OpenRouter judge."""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import statistics
import sys
import time
import urllib.error
import urllib.request
import uuid
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


LABELS = {"pass", "fail"}
DEFAULT_DATASET = Path(__file__).parent / "data" / "sample.jsonl"
DEFAULT_HUMAN_RATINGS = Path(__file__).parent / "data" / "human_ratings.csv"


def load_dataset(path: Path) -> list[dict[str, Any]]:
    cases = []
    seen_ids = set()
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError as exc:
        raise ValueError(f"Could not read dataset {path}: {exc}") from exc

    for line_number, line in enumerate(lines, start=1):
        if not line.strip():
            continue
        try:
            case = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"{path}:{line_number}: invalid JSON: {exc.msg}") from exc
        if not isinstance(case, dict):
            raise ValueError(f"{path}:{line_number}: each dataset row must be a JSON object")
        required = ("id", "question", "reference_answer", "answer", "expected_label", "business_impact_usd")
        missing = [field for field in required if field not in case]
        if missing:
            raise ValueError(f"{path}:{line_number}: missing fields: {', '.join(missing)}")
        case_id = case["id"]
        if not isinstance(case_id, str) or not case_id.strip():
            raise ValueError(f"{path}:{line_number}: id must be a non-empty string")
        if case_id in seen_ids:
            raise ValueError(f"{path}:{line_number}: duplicate id {case_id!r}")
        if not isinstance(case["expected_label"], str) or case["expected_label"] not in LABELS:
            raise ValueError(f"{path}:{line_number}: expected_label must be 'pass' or 'fail'")
        if not all(isinstance(case[field], str) for field in ("question", "reference_answer", "answer")):
            raise ValueError(f"{path}:{line_number}: question, reference_answer, and answer must be strings")
        impact = case["business_impact_usd"]
        if isinstance(impact, bool) or not isinstance(impact, (int, float)) or not math.isfinite(impact) or impact < 0:
            raise ValueError(f"{path}:{line_number}: business_impact_usd must be a finite non-negative number")
        seen_ids.add(case_id)
        cases.append(case)

    if not cases:
        raise ValueError(f"Dataset {path} contains no cases")
    return cases


def load_human_ratings(path: Path | None) -> dict[str, dict[str, str]]:
    if path is None:
        return {}
    try:
        with path.open(newline="", encoding="utf-8") as file:
            reader = csv.DictReader(file)
            if reader.fieldnames != ["case_id", "reviewer_id", "label"]:
                raise ValueError(f"{path}: CSV header must be case_id,reviewer_id,label")
            ratings: dict[str, dict[str, str]] = defaultdict(dict)
            for line_number, row in enumerate(reader, start=2):
                case_id = (row["case_id"] or "").strip()
                reviewer = (row["reviewer_id"] or "").strip()
                label = (row["label"] or "").strip().lower()
                if not case_id or not reviewer:
                    raise ValueError(f"{path}:{line_number}: case_id and reviewer_id must not be empty")
                if label not in LABELS:
                    raise ValueError(f"{path}:{line_number}: label must be 'pass' or 'fail'")
                if reviewer in ratings[case_id]:
                    raise ValueError(f"{path}:{line_number}: duplicate rating for {case_id!r} by {reviewer!r}")
                ratings[case_id][reviewer] = label
    except OSError as exc:
        raise ValueError(f"Could not read human ratings {path}: {exc}") from exc
    return ratings


def direct_check(answer: str, reference_answer: str) -> str:
    normalize = lambda text: " ".join(text.casefold().split())
    return "pass" if normalize(answer) == normalize(reference_answer) else "fail"


def confusion_metrics(actual: list[str], predicted: list[str]) -> dict[str, Any]:
    if len(actual) != len(predicted) or not actual:
        raise ValueError("actual and predicted labels must have the same non-zero length")
    tp = sum(a == "pass" and p == "pass" for a, p in zip(actual, predicted))
    tn = sum(a == "fail" and p == "fail" for a, p in zip(actual, predicted))
    fp = sum(a == "fail" and p == "pass" for a, p in zip(actual, predicted))
    fn = sum(a == "pass" and p == "fail" for a, p in zip(actual, predicted))
    return {
        "true_positive": tp,
        "false_positive": fp,
        "true_negative": tn,
        "false_negative": fn,
        "precision": tp / (tp + fp) if tp + fp else None,
        "recall": tp / (tp + fn) if tp + fn else None,
        "accuracy": (tp + tn) / len(actual),
    }


def cohens_kappa(first: list[str], second: list[str]) -> float | None:
    if len(first) != len(second) or not first:
        return None
    observed = sum(a == b for a, b in zip(first, second)) / len(first)
    first_counts = Counter(first)
    second_counts = Counter(second)
    expected = sum(first_counts[label] * second_counts[label] for label in LABELS) / (len(first) ** 2)
    if math.isclose(expected, 1.0):
        return None
    return (observed - expected) / (1 - expected)


def agreement_report(
    ratings: dict[str, dict[str, str]],
    case_ids: list[str],
    judge_labels: dict[str, str],
) -> dict[str, Any]:
    reviewers = sorted({reviewer for by_reviewer in ratings.values() for reviewer in by_reviewer})
    human_pairs = []
    for index, first_reviewer in enumerate(reviewers):
        for second_reviewer in reviewers[index + 1 :]:
            pairs = [
                (by_reviewer[first_reviewer], by_reviewer[second_reviewer])
                for case_id in case_ids
                if (by_reviewer := ratings.get(case_id, {})).get(first_reviewer)
                and by_reviewer.get(second_reviewer)
            ]
            if pairs:
                human_pairs.append(
                    {
                        "reviewer_a": first_reviewer,
                        "reviewer_b": second_reviewer,
                        "paired_cases": len(pairs),
                        "cohens_kappa": cohens_kappa(
                            [pair[0] for pair in pairs], [pair[1] for pair in pairs]
                        ),
                    }
                )

    judge_pairs = []
    for reviewer in reviewers:
        pairs = [
            (judge_labels[case_id], ratings[case_id][reviewer])
            for case_id in case_ids
            if case_id in judge_labels and reviewer in ratings.get(case_id, {})
        ]
        if pairs:
            judge_pairs.append(
                {
                    "reviewer": reviewer,
                    "paired_cases": len(pairs),
                    "cohens_kappa": cohens_kappa(
                        [pair[0] for pair in pairs], [pair[1] for pair in pairs]
                    ),
                }
            )
    return {"human_vs_human": human_pairs, "judge_vs_human": judge_pairs}


def openrouter_judge(
    case: dict[str, Any],
    api_key: str,
    model: str,
    temperature: float,
) -> tuple[str, dict[str, Any], float]:
    rubric = (
        "Mark pass only when the answer is factually consistent with the reference and answers the question. "
        "Otherwise mark fail. Ignore style differences. Use the reference as the source of truth."
    )
    prompt = (
        f"Rubric: {rubric}\nQuestion: {case['question']}\n"
        f"Reference answer: {case['reference_answer']}\nCandidate answer: {case['answer']}\n"
        'Return only JSON of the form {"label":"pass" or "fail","reason":"brief explanation"}.'
    )
    request_body = json.dumps(
        {
            "model": model,
            "temperature": temperature,
            "max_tokens": 120,
            "messages": [
                {"role": "system", "content": "You are a careful evaluator. Follow the rubric exactly."},
                {"role": "user", "content": prompt},
            ],
        }
    ).encode("utf-8")
    request = urllib.request.Request(
        "https://openrouter.ai/api/v1/chat/completions",
        data=request_body,
        headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
        method="POST",
    )
    started = time.monotonic()
    try:
        with urllib.request.urlopen(request, timeout=60) as response:
            payload = json.loads(response.read().decode("utf-8"))
    except (urllib.error.URLError, TimeoutError) as exc:
        raise RuntimeError(f"OpenRouter judge request failed: {exc}") from exc
    except (json.JSONDecodeError, UnicodeDecodeError) as exc:
        raise RuntimeError("OpenRouter returned invalid JSON") from exc
    elapsed_ms = (time.monotonic() - started) * 1000
    try:
        content = payload["choices"][0]["message"]["content"]
        decoded = json.loads(content)
        label = decoded["label"]
        reason = decoded.get("reason", "")
        usage = payload.get("usage", {})
    except (KeyError, IndexError, TypeError, json.JSONDecodeError) as exc:
        raise RuntimeError("OpenRouter judge returned an invalid verdict; expected JSON with a label") from exc
    if not isinstance(label, str) or label not in LABELS or not isinstance(reason, str):
        raise RuntimeError("OpenRouter judge verdict must contain a 'pass' or 'fail' label and string reason")
    if not isinstance(usage, dict):
        raise RuntimeError("OpenRouter judge returned invalid token usage metadata")
    return label, {
        "reason": reason,
        "prompt_tokens": usage.get("prompt_tokens"),
        "completion_tokens": usage.get("completion_tokens"),
        "reported_cost_usd": usage.get("cost"),
    }, elapsed_ms


def token_cost(usage: dict[str, Any]) -> float | None:
    reported_cost = usage.get("reported_cost_usd")
    if isinstance(reported_cost, (int, float)) and not isinstance(reported_cost, bool):
        if not math.isfinite(reported_cost) or reported_cost < 0:
            raise ValueError("OpenRouter reported an invalid cost")
        return float(reported_cost)
    input_rate = os.environ.get("OPENROUTER_INPUT_COST_PER_1M")
    output_rate = os.environ.get("OPENROUTER_OUTPUT_COST_PER_1M")
    if input_rate is None or output_rate is None:
        if input_rate is not None or output_rate is not None:
            raise ValueError("Set both OPENROUTER_INPUT_COST_PER_1M and OPENROUTER_OUTPUT_COST_PER_1M")
        return None
    try:
        input_rate_value = float(input_rate)
        output_rate_value = float(output_rate)
        prompt_tokens = int(usage["prompt_tokens"])
        completion_tokens = int(usage["completion_tokens"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("Could not calculate judge cost from token usage and configured rates") from exc
    if (
        not math.isfinite(input_rate_value)
        or not math.isfinite(output_rate_value)
        or input_rate_value < 0
        or output_rate_value < 0
        or prompt_tokens < 0
        or completion_tokens < 0
    ):
        raise ValueError("Judge token counts and configured rates must be finite and non-negative")
    return (prompt_tokens * input_rate_value + completion_tokens * output_rate_value) / 1_000_000


def summarize(
    cases: list[dict[str, Any]],
    results: list[dict[str, Any]],
    ratings: dict[str, dict[str, str]],
    judge_enabled: bool,
) -> dict[str, Any]:
    actual = [case["expected_label"] for case in cases]
    direct_predictions = [result["direct_label"] for result in results]
    direct_mistake_cost = sum(
        float(case["business_impact_usd"])
        for case, prediction in zip(cases, direct_predictions)
        if prediction != case["expected_label"]
    )
    summary: dict[str, Any] = {
        "case_count": len(cases),
        "direct": {
            **confusion_metrics(actual, direct_predictions),
            "business_mistake_cost_usd": direct_mistake_cost,
        },
        "judge": None,
        "human_agreement": agreement_report(ratings, [case["id"] for case in cases], {}),
    }
    if judge_enabled:
        judge_predictions = [result["judge_label"] for result in results]
        costs = [result["judge_cost_usd"] for result in results]
        total_cost = None if any(cost is None for cost in costs) else sum(costs)
        latencies = [result["judge_latency_ms"] for result in results]
        summary["judge"] = {
            **confusion_metrics(actual, judge_predictions),
            "business_mistake_cost_usd": sum(
                float(case["business_impact_usd"])
                for case, prediction in zip(cases, judge_predictions)
                if prediction != case["expected_label"]
            ),
            "llm_cost_usd": total_cost,
            "mean_latency_ms": statistics.mean(latencies) if latencies else None,
            "p95_latency_ms": sorted(latencies)[math.ceil(0.95 * len(latencies)) - 1] if latencies else None,
        }
        summary["human_agreement"] = agreement_report(
            ratings,
            [case["id"] for case in cases],
            {case["id"]: result["judge_label"] for case, result in zip(cases, results)},
        )
    majority_label = Counter(actual).most_common(1)[0][0]
    summary["majority_baseline"] = {
        "label": majority_label,
        "cohens_kappa_vs_reference": cohens_kappa(actual, [majority_label] * len(actual)),
    }
    return summary


def run_evaluation(
    cases: list[dict[str, Any]],
    ratings: dict[str, dict[str, str]],
    repeats: int,
    api_key: str | None,
    model: str,
    temperature: float,
    experiment_id: str,
) -> dict[str, Any]:
    all_runs = []
    traces = []
    for run_number in range(1, repeats + 1):
        run_results = []
        for case in cases:
            direct_label = direct_check(case["answer"], case["reference_answer"])
            result: dict[str, Any] = {
                "case_id": case["id"],
                "direct_label": direct_label,
            }
            if api_key:
                judge_label, usage, latency_ms = openrouter_judge(case, api_key, model, temperature)
                result.update(
                    {
                        "judge_label": judge_label,
                        "judge_reason": usage["reason"],
                        "judge_prompt_tokens": usage["prompt_tokens"],
                        "judge_completion_tokens": usage["completion_tokens"],
                        "judge_cost_usd": token_cost(usage),
                        "judge_latency_ms": latency_ms,
                    }
                )
                traces.append(
                    {
                        "experiment_id": experiment_id,
                        "run": run_number,
                        "case_id": case["id"],
                        "evaluator": "openrouter",
                        "model": model,
                        "input": case["question"],
                        "reference_answer": case["reference_answer"],
                        "candidate_answer": case["answer"],
                        "label": judge_label,
                        "latency_ms": latency_ms,
                        "usage": {
                            "prompt_tokens": usage["prompt_tokens"],
                            "completion_tokens": usage["completion_tokens"],
                            "cost_usd": result["judge_cost_usd"],
                        },
                    }
                )
            else:
                traces.append(
                    {
                        "experiment_id": experiment_id,
                        "run": run_number,
                        "case_id": case["id"],
                        "evaluator": "direct",
                        "input": case["question"],
                        "reference_answer": case["reference_answer"],
                        "candidate_answer": case["answer"],
                        "label": direct_label,
                    }
                )
            run_results.append(result)
        all_runs.append(
            {
                "run": run_number,
                "summary": summarize(cases, run_results, ratings, bool(api_key)),
                "results": run_results,
            }
        )

    direct_accuracies = [run["summary"]["direct"]["accuracy"] for run in all_runs]
    judge_accuracies = [
        run["summary"]["judge"]["accuracy"] for run in all_runs if run["summary"]["judge"] is not None
    ]
    return {
        "experiment_id": experiment_id,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "judge": {"enabled": bool(api_key), "provider": "OpenRouter" if api_key else None, "model": model if api_key else None},
        "repeat_count": repeats,
        "summary": all_runs[0]["summary"],
        "repeat_variability": {
            "direct_accuracy_mean": statistics.mean(direct_accuracies),
            "direct_accuracy_stdev": statistics.pstdev(direct_accuracies),
            "judge_accuracy_mean": statistics.mean(judge_accuracies) if judge_accuracies else None,
            "judge_accuracy_stdev": statistics.pstdev(judge_accuracies) if judge_accuracies else None,
            "note": "Repeated runs of deterministic direct checks have zero variability; repeated judge calls can vary.",
        },
        "runs": all_runs,
        "traces": traces,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run the local AI evaluation POC.")
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET, help="JSONL evaluation dataset")
    parser.add_argument(
        "--human-ratings",
        type=Path,
        default=DEFAULT_HUMAN_RATINGS,
        help="CSV with case_id,reviewer_id,label (use an empty file/header to skip)",
    )
    parser.add_argument("--output-dir", type=Path, default=Path("runs"), help="Directory for result and trace files")
    parser.add_argument("--repeat", type=int, default=1, help="Number of complete evaluation runs (default: 1)")
    parser.add_argument("--model", default=os.environ.get("OPENROUTER_MODEL", "openai/gpt-4o-mini"))
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--offline", action="store_true", help="Do not call OpenRouter, even if a key is set")
    args = parser.parse_args(argv)
    if args.repeat < 1:
        parser.error("--repeat must be at least 1")
    if not 0 <= args.temperature <= 2:
        parser.error("--temperature must be between 0 and 2")

    try:
        cases = load_dataset(args.dataset)
        ratings = load_human_ratings(args.human_ratings)
        unknown_rating_cases = set(ratings) - {case["id"] for case in cases}
        if unknown_rating_cases:
            raise ValueError(
                "Human ratings contain case IDs not present in the dataset: "
                + ", ".join(sorted(unknown_rating_cases))
            )
        api_key = None if args.offline else os.environ.get("OPENROUTER_API_KEY")
        experiment_id = "eval-" + uuid.uuid4().hex[:10]
        report = run_evaluation(
            cases,
            ratings,
            args.repeat,
            api_key,
            args.model,
            args.temperature,
            experiment_id,
        )
        args.output_dir.mkdir(parents=True, exist_ok=True)
        result_path = args.output_dir / f"{experiment_id}.json"
        trace_path = args.output_dir / f"{experiment_id}.traces.jsonl"
        result_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        with trace_path.open("w", encoding="utf-8") as trace_file:
            for trace in report["traces"]:
                trace_file.write(json.dumps(trace) + "\n")
    except (OSError, ValueError, RuntimeError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    print(f"Experiment: {experiment_id}")
    print(f"Cases: {len(cases)} | runs: {args.repeat} | judge: {'OpenRouter' if api_key else 'off'}")
    print(f"Direct accuracy: {report['summary']['direct']['accuracy']:.1%}")
    print(
        "Direct confusion matrix (positive=pass): "
        f"TP={report['summary']['direct']['true_positive']} "
        f"FP={report['summary']['direct']['false_positive']} "
        f"TN={report['summary']['direct']['true_negative']} "
        f"FN={report['summary']['direct']['false_negative']}"
    )
    print(f"Direct business mistake cost: ${report['summary']['direct']['business_mistake_cost_usd']:.2f}")
    for pair in report["summary"]["human_agreement"]["human_vs_human"]:
        kappa = pair["cohens_kappa"]
        rendered = "undefined (no label variation)" if kappa is None else f"{kappa:.3f}"
        print(f"Human Cohen's kappa ({pair['reviewer_a']} vs {pair['reviewer_b']}): {rendered}")
    if report["summary"]["judge"] is not None:
        judge = report["summary"]["judge"]
        cost = "unknown" if judge["llm_cost_usd"] is None else f"${judge['llm_cost_usd']:.6f}"
        print(
            f"Judge accuracy: {judge['accuracy']:.1%} | "
            f"business mistake cost: ${judge['business_mistake_cost_usd']:.2f} | "
            f"LLM cost: {cost} | mean latency: {judge['mean_latency_ms']:.0f} ms"
        )
    print(f"Repeat direct accuracy stdev: {report['repeat_variability']['direct_accuracy_stdev']:.3f}")
    print(f"Results: {result_path}\nLocal traces: {trace_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
