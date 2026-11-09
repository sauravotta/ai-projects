# AI Evals POC

A local-first, small CLI demo of practical LLM evaluation. It compares exact/reference-based checks with an optional LLM judge, accepts human ratings from CSV, and writes experiment results and local trace records. The default run is offline and needs no API key or third-party Python packages.

## What it demonstrates

| Use case | POC implementation |
|---|---|
| Direct measurement | Normalized exact-match check against each reference answer |
| LLM-as-a-judge | Optional OpenRouter chat completion; the judge receives a rubric, question, reference answer, and candidate answer, and returns a bounded JSON verdict |
| Human review | Import one or more reviewers' `pass` / `fail` ratings from CSV |
| Metric portfolio | Accuracy plus a confusion matrix, precision, and recall for both direct checks and the optional judge |
| Cost of mistakes | Sum of per-case business impact when an evaluator disagrees with the expected label |
| LLM cost and latency | Judge token usage and latency; use provider-reported cost or optional configured per-token rates |
| Cohen's kappa | Pairwise human agreement and, when enabled, judge-versus-human agreement; also reports a majority-label baseline |
| Observability | Local JSONL trace records grouped by experiment and run; JSON report contains per-run outcomes |
| Noise / repeatability | `--repeat N` repeats each evaluation and reports accuracy variability across runs |

This is a teaching POC, not a production evaluation platform. Its sample has only eight cases. A credible quality estimate or release gate needs a representative, carefully labeled set of **1,000 or more cases**; repeating a small set does not add independent test coverage.

## Requirements

- Python 3.9 or newer
- No package installation is required
- Optional: an [OpenRouter API key](https://openrouter.ai/keys) to run the LLM judge

## Run locally

From the repository root:

```bash
cd ai-evals
python3 -m venv .venv
source .venv/bin/activate
python eval_poc.py
```

The default command performs direct checks and calculates agreement from the bundled human ratings. It makes no network requests. Example output:

```text
Experiment: eval-...
Cases: 8 | runs: 1 | judge: off
Direct accuracy: 87.5%
Direct confusion matrix (positive=pass): TP=4 FP=0 TN=3 FN=1
Direct business mistake cost: $15.00
Human Cohen's kappa (reviewer-1 vs reviewer-2): ...
Repeat direct accuracy stdev: 0.000
Results: runs/eval-....json
Local traces: runs/eval-....traces.jsonl
```

Run the focused tests with the standard library:

```bash
python -m unittest discover -s tests -v
```

### Optional OpenRouter judge

Set the key in the current shell; do not put credentials in the dataset or commit them:

```bash
export OPENROUTER_API_KEY="your-key"
python eval_poc.py --repeat 3
```

The default judge model is `openai/gpt-4o-mini`. Override it with `OPENROUTER_MODEL` or `--model`:

```bash
export OPENROUTER_MODEL="provider/model-name"
python eval_poc.py --model "provider/model-name" --temperature 0.2
```

Each judge call is a real API request and may incur provider charges. The report uses cost returned by the provider when available. If cost is not included in the response, configure both rates to get an estimate (USD per one million tokens):

```bash
export OPENROUTER_INPUT_COST_PER_1M="0.15"
export OPENROUTER_OUTPUT_COST_PER_1M="0.60"
```

Without a reported cost or both configured rates, the judge cost is shown as unknown rather than being presented as zero. Latency is measured around each judge request. The business-mistake impact values are illustrative USD amounts supplied by the dataset; they are not API charges.

To force offline operation even when a key is set:

```bash
python eval_poc.py --offline
```

## Dataset format

The default dataset is [`data/sample.jsonl`](./data/sample.jsonl), one JSON object per line:

```json
{"id":"case-01","question":"What is 2 + 2?","reference_answer":"4","answer":"4","expected_label":"pass","business_impact_usd":0}
```

Required fields:

- `id`: unique case identifier, also used to connect human ratings to cases
- `question`: input shown to the judge and included in traces
- `reference_answer`: expected answer shown to the judge and used by direct exact matching
- `answer`: candidate output being evaluated (this POC evaluates saved outputs; it does not call a candidate/generation model)
- `expected_label`: ground-truth `pass` or `fail`, used to calculate metrics
- `business_impact_usd`: impact amount charged against an incorrect evaluator decision

Direct evaluation lowercases and collapses whitespace, then requires the candidate answer to match the reference exactly. This is intentionally simple; paraphrases can fail exact match even when semantically correct. The LLM judge is a separate, optional assessment using the rubric and reference answer. Treat either result as an example to inspect, not as ground truth for real deployments.

Use your own dataset with:

```bash
python eval_poc.py --dataset path/to/cases.jsonl
```

## Human review and agreement

The included [`data/human_ratings.csv`](./data/human_ratings.csv) is an example of two reviewers labeling the sample cases. The required CSV header is:

```csv
case_id,reviewer_id,label
```

Add one row per review, with `label` set to `pass` or `fail`. For example:

```csv
case-01,reviewer-alex,pass
case-01,reviewer-sam,fail
```

Provide a different file with `--human-ratings path/to/reviews.csv`. To omit human annotations, use a header-only CSV with that exact header. The report calculates Cohen's kappa for reviewer pairs with shared cases, plus judge-versus-reviewer agreement when the LLM judge is enabled. Kappa is undefined when chance agreement is 100%; the report writes `null` in that case. The majority-label baseline illustrates how a constant majority prediction can have zero kappa when both labels occur.

## Metrics and business impact

`pass` is the positive class:

- **TP**: expected pass, predicted pass
- **FP**: expected fail, predicted pass
- **TN**: expected fail, predicted fail
- **FN**: expected pass, predicted fail
- **Precision** = TP / (TP + FP); **recall** = TP / (TP + FN)
- **Business mistake cost** = sum of `business_impact_usd` for incorrect predictions

The judge section reports its business mistake cost beside LLM cost and mean/p95 request latency. The direct evaluator does not make LLM calls, so it has no LLM usage cost.

## Repeats, results, and traces

Repeat a run to see whether judge decisions vary:

```bash
python eval_poc.py --repeat 5
```

The default direct check is deterministic, so its repeated-run standard deviation is zero. Repeating judge calls can expose model variability, but it does not replace a large representative dataset or independent human labels.

Every invocation writes two files under `runs/` (ignored by Git):

- `<experiment-id>.json`: settings, aggregate summary, per-run metrics, case-level verdicts, and repeat variability
- `<experiment-id>.traces.jsonl`: one local trace record per case per run, including input and answer text

Choose another output directory with `--output-dir path/to/output`. Trace files contain evaluation inputs and candidate answers; store them appropriately and avoid sharing sensitive data.

## Project files

| File | Purpose |
|---|---|
| `eval_poc.py` | CLI, validation, evaluation, metrics, optional OpenRouter judge, result and trace output |
| `data/sample.jsonl` | Small illustrative dataset |
| `data/human_ratings.csv` | Example human annotations |
| `tests/test_eval_poc.py` | Standard-library unit and offline CLI tests |
