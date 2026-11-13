# AI Self-Improvement POC (Java)

A Java demonstration of the self-improvement ideas. It covers the three areas in that section:

1. **Improver:** analyze results and propose a bounded change to a prompt, model, or reasoning effort; a person approves or rejects the proposal.
2. **Held-out evaluation:** after approval, compare the proposed change against a baseline on data not used to select it. Keep the change only when it clears quality, regression, cost, and uncertainty gates; otherwise roll back.
3. **Recursive self-improvement:** evaluate whether the improver should change its own selection recipe, using separate meta-training and meta-held-out data.

The default POC runs locally, deterministically, and offline. An explicit `--llm` option uses [OpenRouter](https://openrouter.ai) as the improver and recursive meta-improver. In either mode, the bundled outcomes and recipe scores are **synthetic fixture data** for demonstrating the workflow; they are not model benchmarks or evidence of real-world improvement. OpenRouter chooses among bounded proposals; it does not generate candidate agent answers or update model weights.

## What happens in a run

```text
training results
    -> improver selects a permitted candidate
    -> human approval
    -> paired held-out evaluation + guardrails
    -> keep the change or roll back
    -> meta-training results select an improver recipe
    -> human approval
    -> paired meta-held-out evaluation
    -> keep or roll back the recipe change
```

### 1. Improver proposal

The improver compares the baseline against three pre-defined, inspectable changes:

| Candidate | Change |
|---|---|
| `prompt_checklist` | Add a short verification checklist to the prompt |
| `model_upgrade` | Select the stronger local simulator model |
| `reasoning_medium` | Raise reasoning effort from low to medium |

Without `--llm`, the local selector chooses the best permitted candidate. With `--llm`, OpenRouter receives the training metrics and eligible proposal list, then recommends one option. The default `accuracy_first` recipe ranks accuracy first; the recursive `business_aware` recipe weighs business mistake impact as well.

These are allowlisted configuration proposals, not generated executable code. A proposal reports its training score and rationale before asking for approval. The chosen candidate is checked against the eligible list in code; an unknown or ineligible LLM choice fails explicitly.

### 2. Held-out promotion gate

The candidate is evaluated against the baseline on the separate `heldout` split only after approval. Paired bootstrap resampling estimates the uncertainty interval for the accuracy change. The gate requires:

- The configured minimum accuracy and per-category accuracy floors
- No category regression beyond the configured allowance
- Improvement at least as large as the configured minimum, with a paired-bootstrap lower bound above zero
- Candidate model on the permitted-model list
- Evaluation spend under the configured budget

If any condition fails, the report marks the proposal `rolled_back`; otherwise, it marks it `kept`. A favorable result on this tiny synthetic sample is just a workflow demonstration, not a statistical guarantee.

### 3. Recursive recipe tuning

The recursive stage compares the current selector (`accuracy_first` by default) with `business_aware`. The local selector or optional OpenRouter uses `meta_train` scores only, asks for separate human approval, and validates the choice on `meta_heldout`. If that gate fails, the original recipe remains active. Keeping a recipe means the next improvement cycle would use it; this one-shot demo does not launch an unbounded recursive loop.

## Run locally

### Requirements

- JDK 21 or newer
- Maven 3.8 or newer
- No API key required for the default offline run; OpenRouter is opt-in

From the repository root:

```bash
cd ai-self-improvement
mvn test
mvn package
```

The shaded JAR includes its runtime dependencies. Start the interactive demo:

```bash
java -jar target/ai-self-improvement-1.0.0.jar
```

The offline interactive run asks separately whether to approve the agent change and the recursive recipe change. Enter `y` or `yes` to approve; pressing Enter rejects. In a non-interactive environment, proposals are safely rejected unless explicitly approved.

For a repeatable, scripted demonstration, explicitly approve both proposed changes:

```bash
java -jar target/ai-self-improvement-1.0.0.jar --approve
```

### Optional OpenRouter improver

The default command never contacts a service. To let OpenRouter recommend the prompt/model/reasoning candidate and recursive recipe:

1. Create an API key at [OpenRouter](https://openrouter.ai/keys).
2. From `ai-self-improvement/`, create a local `.env` file and add the key. `.env` is gitignored:

   ```bash
   cp .env.example .env
   ```

   Edit `.env`:

   ```dotenv
   OPENROUTER_API_KEY=your-key
   OPENROUTER_MODEL=openai/gpt-4o-mini
   ```

   Alternatively, export the variables in your shell:

   ```bash
   export OPENROUTER_API_KEY="your-key"
   export OPENROUTER_MODEL="openai/gpt-4o-mini"
   ```

3. Explicitly enable OpenRouter:

   ```bash
   java -jar target/ai-self-improvement-1.0.0.jar --llm
   ```

   For a non-interactive approved demo, include `--approve`:

   ```bash
   java -jar target/ai-self-improvement-1.0.0.jar --llm --approve
   ```

`--model provider/model-name` overrides `OPENROUTER_MODEL`. If the model is omitted, the default is `openai/gpt-4o-mini`. Configure a model available to your OpenRouter account; requests use the OpenAI-compatible Chat Completions endpoint. This makes up to two real API calls per run (candidate improver and recursive improver), which may incur charges. The report records provider-reported token counts and cost when present.

The requests contain only aggregate **training** metrics, eligible option descriptions, and meta-training recipe scores. Held-out case outcomes and meta-held-out scores are not sent to OpenRouter. API errors, invalid JSON, and out-of-allowlist recommendations stop the run with an error; the program does not silently fall back to a local recommendation after a requested LLM call fails.

If you see HTTP 401, check the key. For 402 or 429 responses, check your OpenRouter credits, model access, or rate limits; choose another model with `OPENROUTER_MODEL` or `--model`.

Example OpenRouter run output (usage values depend on the response):

```text
OpenRouter candidate improver: model=openai/gpt-4o-mini, prompt_tokens=..., completion_tokens=..., cost=...
Improver proposal: prompt -> prompt_checklist
Add a short verification checklist to the prompt.
Training accuracy: 50.0% -> 100.0% (selector score +0.500; held-out data not used)
Improver rationale: It improves training metrics without exceeding the configured budget.
Human approval: approved by --approve for agent change prompt_checklist.
Held-out accuracy: 53.8% -> 100.0%; 90% paired-bootstrap interval [0.231, 0.692]
Promotion gate passed: keep the candidate change.

OpenRouter recursive improver: model=openai/gpt-4o-mini, prompt_tokens=..., completion_tokens=..., cost=...
Recursive proposal: tune the improver recipe accuracy_first -> business_aware
Recursive improver rationale: It scored higher on meta-training data.
...
Recursive gate passed: keep business_aware as the improver recipe.

Run report: runs/improve-....json
```

The output is written to `ai-self-improvement/runs/`, ignored by Git.

Run only the tests:

```bash
mvn test
```

## Inspect and customize

### Evaluation data

[`data/evaluations.jsonl`](./data/evaluations.jsonl) contains simulated per-case results with:

- `split`: `train` or `heldout`
- `category`, `expected`, and `business_impact_usd`
- `outcomes`: a `pass` / `fail` result, simulated inference cost, and latency for each policy variant

Run with another JSONL dataset:

```bash
java -jar target/ai-self-improvement-1.0.0.jar --dataset path/to/evaluations.jsonl
```

The code filters by split before scoring proposals; it never uses held-out results to choose the candidate. Keep a real held-out set private to the improvement process and use it only for promotion decisions.

### Recursive recipe data

[`data/recipe_evaluations.jsonl`](./data/recipe_evaluations.jsonl) contains synthetic `recipe_scores` for `accuracy_first` and `business_aware`, partitioned into `meta_train` and `meta_heldout`. Supply another file with `--recipe-dataset path/to/recipes.jsonl`.

### Guardrails

Edit [`config.json`](./config.json) to set:

- `permitted_models`: allowlist of simulated model IDs
- `minimum_accuracy` and `minimum_category_accuracy`: quality floors
- `maximum_category_regression`: tolerated per-category regression
- `minimum_improvement` and `confidence_level`: promotion thresholds
- `maximum_evaluation_budget_usd`: maximum simulated candidate evaluation cost
- `bootstrap_samples`: number of paired resamples (100 through 1,000,000)
- `initial_improver_recipe`: `accuracy_first` or `business_aware`
- `minimum_recipe_score`: recursive recipe quality floor

Pass a separate configuration with `--config path/to/config.json`. Guardrails are validated; malformed or out-of-range values produce a clear error.

### Options and output

```text
java -jar target/ai-self-improvement-1.0.0.jar --help
```

Help includes `--llm` to enable OpenRouter and `--model NAME` to select the OpenRouter model.

| Option | Purpose |
|---|---|
| `--approve` | Explicitly approve both changes for a scripted demo |
| `--llm` | Use OpenRouter for both improvement recommendations |
| `--model NAME` | Override the OpenRouter model |
| `--dataset PATH` | Change the simulated agent evaluation data |
| `--recipe-dataset PATH` | Change the meta-evaluation data |
| `--config PATH` | Change allowlists, floors, budget, and gates |
| `--seed N` | Set the paired-bootstrap random seed |
| `--output-dir PATH` | Set where the local JSON report is written |

Each report records the proposal, approval outcome, held-out metrics, gate failures, and final keep/rollback decision. It does not edit the config, prompt, or model automatically.

## Limitations

- The simulator replays pre-recorded outcomes; it does not generate candidate agent answers or update model weights. With `--llm`, OpenRouter selects among fixed candidate proposals but does not execute them.
- The example datasets are small synthetic fixtures. A serious system needs representative, versioned data, independent held-out sets, and a sample size adequate for its risks.
- Bootstrap intervals are illustrative; repeated rows or synthetic scores do not create independent evidence.
- The simulated cost and latency fields are not provider quotes.
- Human approval is required for both types of change. `--approve` is an explicit opt-in intended for scripted demos.
- With `--llm`, training metrics and proposal options are sent to OpenRouter. Use only data approved for external processing; without that flag, the run stays offline.
