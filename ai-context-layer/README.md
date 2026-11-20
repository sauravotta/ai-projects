# Context layer and integrations

This Python proof of concept demonstrates ai context layer with a fictional support scenario. It combines business definitions and procedures with scoped data from simulated CRM and ticketing systems, customer-specific memory, access policies, provenance, and an OpenRouter-powered assistant.

## Architecture

```text
simulated identity -> policy gateway -> context layer -> OpenRouter answer
                      |                 |                  (no tools or writes)
                      |                 +-- glossary / support playbook
                      |                 +-- simulated CRM and ticketing records
                      |                 +-- customer-scoped preference memory
                      |
                      +-> pending ticket-change proposal -> human approve/reject
                                                        -> simulated ticket record
```

The context layer supplies what terms mean, how to work, what the agent can see, and where each piece originated. The gateway applies fixed identity/customer/operation rules **before** assembling context or calling the model. The SQLite database represents the systems of record and a separate memory namespace; the sample uses one memory type (customer preference). Neither the context layer nor the LLM is the authoritative source for the ticket's current state.

| PDF concept | Demo implementation |
|---|---|
| Data dictionary and glossary | `priority` and `next_action` definitions with provenance |
| Work procedures | A short support playbook for P1/P2 |
| Access controls and boundaries | Customer-scoped principals and separate read/propose/approve permissions |
| Provenance and lineage | Each context component identifies its source system, owner, and review date |
| Systems of record | Separate simulated CRM and ticket tables; the ticket table is the authoritative source for the update |
| Memory | Separate preference records scoped to one customer; not an authoritative business record |
| Read vs. write | Context is read-only; a proposed next action requires a separate simulated human approval |
| Data quality | Unknown priority/status and missing records fail explicitly before an LLM call |
| Integration gateway | Local policy checks and audit events; no actual MCP server or external company API |

This is intentionally lightweight: prompts stay in the application, not in a general-purpose context service. The glossary, playbook and their review dates are fictional, fixed examples; the date is metadata, not evidence that content is current. No automatic freshness validation or general data-quality profiling is provided.

## Local setup

Requires Python 3.10+ and an OpenRouter API key for LLM calls. The simulated data, access checks, review workflow and tests run locally without a key. From the repository root:

```bash
cd ai-context-layer
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
cp .env.example .env
```

Edit `.env`:

```dotenv
OPENROUTER_API_KEY=your-key
OPENROUTER_MODEL=nvidia/nemotron-3.5-lightning:free
```

The `.env` file and SQLite files are gitignored. Do not commit or share credentials. Environment variables take precedence over `.env`; `--model` overrides the model. You can use an existing local configuration without copying it using `--env-file ../starter-ai-agent/.env` **before** the subcommand. Free-model availability and rate limits change; check OpenRouter pricing before selecting a model. No real customer information should be used in this demo.

## Run locally

All commands below are run from `ai-context-layer/` with the virtual environment activated.

### 1. Initialize and inspect authorized context

```bash
python main.py init
python main.py context --actor agent-acme --customer acme --ticket T-100
```

`init` seeds fictional data only where records do not already exist. It will not reset changed tickets or clear history. The context includes the Acme CRM summary, ticket T-100, the support glossary and playbook, and Acme's preference memory with source/owner/review-date metadata. The simulated internal email is intentionally excluded.

Try a cross-customer request; it must fail and produce a denial in the audit:

```bash
python main.py context --actor agent-acme --customer zen --ticket T-200
```

The sample identities are `agent-acme` (Acme read and proposal), `agent-zen` (Zen read only), and `reviewer` (approval for both). These are **demonstration labels**, not authenticated users. Anyone with local CLI/database access can choose `reviewer` or edit the database. A real integration needs trusted authentication, authorization, a protected approval channel, and separate identities.

### 2. Ask the assistant using OpenRouter

```bash
python main.py ask --actor agent-acme --customer acme --ticket T-100 \
  --question "What is the next support step and which playbook supports it?"
```

The application authorizes access, assembles the minimized context, then sends the question and that context to OpenRouter and its selected provider. It prints the answer, model, provider-reported usage and source metadata. The model cannot invoke the proposal or approval operations. The answer can be wrong or cite incorrectly; source labels are not automatic factual verification.

The local demo was tested with `nvidia/nemotron-3.5-lightning:free`, and the P1 answer identified the on-call incident team. Responses and free-tier availability vary. A missing key, unauthorized context, API error, empty reply, or reply truncated at the token limit fails explicitly. Free-model calls can return 429 or incur provider delays; select another available model with `--model` if necessary.

### 3. Propose and review a simulated update

```bash
python main.py propose --actor agent-acme --customer acme --ticket T-100 \
  --action "Contact the on-call incident team"
python main.py pending --actor reviewer --customer acme
```

The first command prints a proposal number; replace `1` below with the number displayed:

```bash
python main.py decide --actor reviewer --proposal 1 --decision approve \
  --reason "Checked against the support playbook"
python main.py context --actor agent-acme --customer acme --ticket T-100
python main.py audit --actor agent-acme --customer acme
```

Approval updates only the **simulated** ticket's `next_action`. To decline, use `--decision reject`; it does not change the ticket. Duplicate decisions are rejected, and an approval fails if the ticket was changed or closed since the proposal was created. Proposals are manually entered; the model's answer is not automatically executed. No customer is contacted and no external ticket is modified.

## Tests

```bash
python -m unittest discover -s tests -v
```

The suite uses temporary SQLite databases and mocked OpenRouter responses. It checks access boundaries, minimized context, provenance, data quality, review, rejection, stale updates, repeat initialization, CLI behavior and model-output handling; it does not need a live key. For live validation, run the `ask` command above with your own key. The returned answer is nondeterministic and may not always match the expected playbook.

## Data, safety and limitations

- The gateway policies are application code, not an MCP/API/model gateway. External systems are local SQLite tables; there is no real CRM, ticketing API, service credential injection, rate-limiting gateway, Postgres, or MCP integration.
- Only the requested customer's selected fields and preference memory enter model context, but **the question itself is sent to OpenRouter**. Users must not enter secrets or real personal data. OpenRouter and its provider process the supplied content; review their privacy and retention policies before sending anything sensitive.
- The model is instructed to treat records as untrusted data. Prompt instructions are not a security boundary and do not guarantee prompt-injection resistance. Authorization and write controls are enforced in code; the LLM has no write tool.
- The audit log records actor labels, customer IDs, operations, outcomes and reviewer reasons, but not the prompt or model answer. It is local and editable, not tamper-proof; failures such as invalid or stale submissions are not a comprehensive compliance audit.
- SQLite is a local demo store, not distributed infrastructure. External systems require their own authorization, idempotency, concurrency, secret storage, consistency, and reconciliation. The local optimistic version check does not solve cross-system transactions.
- `context.sqlite3` persists between runs. To start over, stop any process using it and remove that specific file and any SQLite sidecars; this deletes local simulated records and audit history.
