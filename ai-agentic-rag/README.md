# Agentic RAG demonstration

This Python proof of concept demonstrates the agentic RAG concepts.

1. Split a local knowledge base into chunks, embed them, and retrieve by semantic similarity.
2. Let an LLM write its own search queries and decide whether to search again, answer, or abstain.
3. Evaluate retrieval with precision, recall and mean reciprocal rank (MRR), and check whether expected facts were found.

The example knowledge base contains **fictional** support policies. Embedding and search run locally; the agent uses a free OpenRouter LLM. No database or hosted vector service is required. This is a small educational demo, not a production-ready RAG system.

## Architecture

```text
local JSON documents -> paragraph chunks -> local embedding model -> in-memory cosine index

question -> OpenRouter decision -> search(query) -> local ranked chunks
                                   ^                    |
                                   +---- repeat --------+
                            -> cited answer or abstention
```

The agent receives retrieved chunks and decides on the next action. It can issue zero to three searches by default, with up to three results per search. The configured search budget is enforced in code. The program validates that every answer has citations and that each citation names an actually retrieved chunk; it does **not** prove that the cited text entails the answer. An answer with fabricated or missing citations, a malformed decision, an API failure, or a request beyond the search budget fails explicitly rather than silently pretending to succeed.

The local model is `BAAI/bge-small-en-v1.5` via FastEmbed. The first run downloads model files from Hugging Face and stores them in FastEmbed's cache, outside the project by default. Later runs use the cached model. Only embeddings and search execute locally: the agent sends the question, its conversation, and retrieved chunk text to OpenRouter and its selected provider. Do not use confidential documents without approving that data transfer. The CLI saves questions, answers, queries, source IDs, scores, model names and usage in ignored local JSON reports; treat reports as potentially sensitive.

## Local setup

Requires Python 3.10+ and internet access for dependency/model downloads and OpenRouter. The embedding model and dependencies require several hundred MB of disk space. Commands below are for macOS/Linux; on Windows activate with `.venv\\Scripts\\activate`.

From the repository root:

```bash
cd ai-agentic-rag
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
cp .env.example .env
```

Edit `.env` and add your own OpenRouter key:

```dotenv
OPENROUTER_API_KEY=your-key
OPENROUTER_MODEL=nvidia/nemotron-3.5-lightning:free
```

The `.env` file is gitignored. Never commit or share the key. Environment variables take precedence over `.env`; `--model` overrides the model in both. The `--env-file PATH` option can select an existing dotenv file without copying it. Do not put keys in command-line arguments.

`nvidia/nemotron-3.5-lightning:free` was used for the live tests described below. Free model availability, rate limits, output quality and routing can change; `openrouter/free` is an alternative router, but can select different models and may produce invalid JSON or hit rate limits. The client requests JSON output and permits OpenAI SDK retries on transient API errors, but it does not silently substitute another model if the requested model fails. Verify pricing on OpenRouter if you choose a model other than a free model.

## Run the agent

From `ai-agentic-rag/`, with the virtual environment active and `.env` configured:

```bash
python main.py ask "Who must be paged for a severity 1 outage and how long are backups retained?"
```

The console shows the answer, cited chunk IDs, search queries and retrieved IDs. The full machine-readable trace and LLM token/cost metadata, when provided by OpenRouter, are written under `runs/`. For an unanswerable question:

```bash
python main.py ask "What is the office address of Aster Support?"
```

The agent should search and abstain rather than invent an address, though LLM output is not deterministic. To choose another model or adjust the limits, place options **before** the subcommand:

```bash
python main.py --model openrouter/free --k 3 --max-searches 3 ask "How long are backups retained?"
```

If a model exceeds the configured budget or returns invalid JSON, a nonexistent citation, or no decision text, the program exits with an error. For HTTP 401 check the key; for 429 wait or choose another available free model. A model that uses many reasoning tokens may still run out of output tokens. Rerunning can produce a different result.

## Run evaluations and tests

```bash
python -m unittest discover -s tests -v
python main.py eval-retrieval
python main.py eval-agent
```

`eval-retrieval` does not use OpenRouter or require a key. It embeds each labeled question, searches once, and scores the first `k` chunks. It needs internet only if the local embedding model is not yet cached. `eval-agent` makes real OpenRouter requests for all five labeled questions, then scores the agent's retrieved evidence and answers. It can make up to four LLM calls per question (three searches plus a final decision), potentially more HTTP attempts if the SDK retries. Free-tier calls can be rate-limited.

Metrics are calculated per case and then averaged:

- **Precision@k**: number of relevant chunks in the first `k` distinct retrieved IDs divided by `k`.
- **Recall@k**: number of relevant chunks in those IDs divided by the number of labeled relevant chunks.
- **MRR@k**: reciprocal rank of the first relevant chunk (zero if none appears in the first `k`), averaged over cases.
- **Evidence fact coverage**: fraction of expected facts whose accepted text alternatives occur in the retrieved chunk text. For an agent this uses evidence from **all** its searches, while its ranking metrics use the first `k` distinct results across those searches.
- **Answer fact coverage**: the same simple substring check applied to the generated answer; reported for `eval-agent` only.

These are transparent smoke-test measures, not a semantic judge: substring matching can miss valid paraphrases or count irrelevant/contradictory text. The five examples are hand-labeled and too small to establish production accuracy, generalization, or statistical confidence. Review citations and answers manually and build a larger, representative evaluation set before relying on the system.

### Observed local test

With the included fixtures and `nvidia/nemotron-3.5-lightning:free`, a local five-case OpenRouter run achieved precision@3 `0.40`, recall@3 `1.00`, MRR@3 `0.90`, evidence fact coverage `1.00`, and answer fact coverage `1.00`. A separate cross-document question returned both cited facts; a question about a nonexistent office address led to an abstention after three searches. These are sample results, not guaranteed outcomes. A direct retrieval-only run achieved the same precision/recall/MRR and evidence coverage on this dataset.

## Customize

- [data/knowledge.json](data/knowledge.json): document IDs, titles and text. Blank lines split a document into numbered paragraph chunks such as `incident:1`.
- [data/eval.json](data/eval.json): questions, relevant chunk IDs, and lists of acceptable substrings for each expected fact. Changing paragraph boundaries changes chunk IDs; update labels accordingly.
- [rag.py](rag.py): local embedding/search, the decision prompt, bounded agent loop, and metric functions.
- [main.py](main.py): CLI, local reports, and evaluation runner. Use `--data`, `--eval-data`, and `--output-dir` before the subcommand to choose alternate paths.

The demo embeds the entire corpus in memory on each run. It has no production authentication, persistent vector index, injection-resistant retrieval isolation, automatic citation entailment verification, or robust corpus/version management. Retrieved text is explicitly treated as data in the prompt, but prompt-injection resistance is not guaranteed.
