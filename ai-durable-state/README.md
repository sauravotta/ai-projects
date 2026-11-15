# Durable state, human review and swarm demo

This local-first Python demo processes a batch of fictional customer-support requests using **durable state**, **human-in-the-loop review**, and a **swarm** of bounded parallel worker copies. A local web page shows progress and lets a person approve or reject risky or uncertain items. Actions are simulated records in SQLite: no real email, refund, deletion, or account change occurs.

 This demo deliberately uses a custom SQLite state machine rather than LangGraph: it demonstrates comparable ideas, **not** the `interrupt`, `Command`, `Send`, or LangGraph checkpoint APIs.

## What happens

1. Create a batch of five sample support requests. Each batch and item has a separate ID and durable history.
2. A dispatcher runs one worker per request, bounded by `--max-concurrency` (default 3, maximum 8). The default worker is deterministic and offline.
3. Code assigns the risk and simulated action from the fixture's category; the worker supplies only a confidence and rationale. High-risk requests **always** pause for review. Low-risk requests pause if confidence is below **0.80**; otherwise they proceed.
4. The web interface shows the request, proposed action, risk, confidence and rationale. A person approves or rejects each pending item and may add a reason. The decision is stored before work resumes.
5. Approved or automatically authorized actions are written to a simulated local system-of-record table. Rejected items do not execute actions. The batch status, item statuses and audit events survive a process restart.

The bundled fixtures include information, account access, refund and deletion requests. Offline, the access request has confidence 0.65 and requires review; refund and deletion are high-risk and always require review. The information requests are low-risk with confidence 0.95 and complete automatically. These are **illustrative policy rules**, not real eligibility or security decisions. An LLM's self-reported confidence is not calibrated; never use this demo as an actual approval system.

## Local setup

Requires Python 3.10+ on macOS or Linux (the batch lock uses `fcntl`). SQLite is included with Python. From the repository root:

```bash
cd ai-durable-state
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python web.py
```

Open **http://127.0.0.1:8000/**, create a sample batch, and open it to inspect each item. Refresh the page to see updates. Review the pending items using **Approve** or **Reject**; rejected items remain recorded and are never applied. A reason is optional. The page shows audit events and each item's state.

The server binds to loopback by default. It is an educational local interface, not a production service: it has no authentication or authorization and must not be exposed to a network or used with sensitive data. It uses a per-server form token to reduce accidental cross-site submissions, but this is not a substitute for production security.

## Demonstrate durability

The SQLite database defaults to `durable_state.sqlite3` in this folder and is gitignored. Each meaningful transition is committed to SQLite with `synchronous=FULL`. Audit events record creation, results, retries, human decisions, recovery and actions.

1. Start the server and create a batch. Observe completed items and items awaiting review.
2. Stop the server with Ctrl+C. Restart it with the same command and database path.
3. Reopen the batch: completed results and pending decisions are still present. Approve or reject a pending item and refresh to observe it resume.
4. For a stronger crash demonstration, terminate the server while work is in flight and restart it. An item interrupted in the processing stage is rerun; already committed item results and human decisions are retained. The test suite also kills a worker process before and after a simulated action transaction commits and verifies recovery.

The demo enforces one active runner per batch using a local OS file lock. SQLite transactions protect the state and action records. A stable `support-action:<item-id>` idempotency key and a uniqueness constraint ensure at most one simulated action record per item; the action and final state are committed together. A local file lock is not a distributed lease and this design is not intended for multiple machines or a shared network filesystem.

**Important:** a real remote action cannot be made atomic with a local SQLite transaction. Use a provider-supported idempotency key, check the actual system of record, reconcile uncertain outcomes, and design compensating actions. Rewinding or replaying a checkpoint does not undo a message, refund, or deletion already performed in the outside world.

## Optional OpenRouter worker

Offline mode does not contact OpenRouter or require a key. To use a free LLM, copy the template and edit it locally:

```bash
cp .env.example .env
# Set OPENROUTER_API_KEY in .env; never commit or share the key.
python web.py --mode openrouter
```

The default model is `nvidia/nemotron-3.5-lightning:free`; use `--model` to select another available model. The worker sends the fictional request text to OpenRouter and its provider and asks for a JSON confidence and rationale. Do not send sensitive requests without approving that data transfer. The code—not the LLM—selects the risk and action. The `.env` file is gitignored; environment variables take precedence for the API key. Model availability, cost, rate limits and response quality can change; verify the model's pricing before use.

Each worker permits up to three requests on connection failures, timeouts, 429/5xx errors, or invalid/empty structured responses, with bounded backoff. Retries are recorded in the item audit trail. A permanent failure is explicitly marked `failed` and **does not** trigger an action or an automatic approval; the demo does not automatically retry failed items across restarts. A failed item can be investigated in the audit trail and a fresh batch started after resolving the underlying issue.

A batch records its worker mode and, for OpenRouter, its model. Restart with the **same** `--mode` and `--model` to resume it. If the server is started with different settings, it reports that the batch cannot be resumed with those settings rather than silently switching workers.

During local validation, a real five-item batch using the free model completed worker processing, with high-risk and low-confidence items paused for review. Free-tier results can vary; one earlier live attempt returned an invalid response, which led to the bounded validation retry described above.

## Tests

```bash
python -m unittest discover -s tests -v
```

Tests use temporary databases and cover state transitions, review gates, duplicate decisions, bounded parallelism, invalid model output, retry limits, and abrupt process termination around the action transaction. They do not require a live API key. The optional OpenRouter path can also be exercised locally with the commands above.

## PDF concepts and implementation boundaries

| PDF concept | Demo implementation |
|---|---|
| Durable execution, checkpoints, per-thread state | SQLite state and event records per batch and per item, committed at each transition |
| Nodes, edges, reducers | Python functions and explicit states; per-item transactions combine parallel outcomes without an in-memory reducer |
| `interrupt()` and `Command` | Persist `waiting_review`, then record a human decision and resume; no LangGraph API is used |
| `Send` and `max_concurrency` | One item per thread-pool task, with a configurable concurrency cap |
| `durability=\"sync\"` | SQLite `synchronous=FULL` commits before later work; not LangGraph durability or a guarantee against all hardware failures |
| Idempotency, external side effects, crash testing | Unique action key and transactional simulated system of record; process-kill tests |
| Human threshold and decision fatigue | Risk or low-confidence gate; pending items remain visible until reviewed |
| Rate limits, retries, resource limits and tail latency | Bounded calls and retries; worker-result timing in audit events; slowest work still determines completion |

This demo does not implement LangGraph checkpoints, Postgres, checkpoint rewind, distributed scheduling, calibrated confidence, or production authentication. Every item uses a fixture category rather than an LLM-generated business action. Logs and SQLite can contain request text and human reasons; keep them local and protect them appropriately.

## Files and cleanup

- `workflow.py`: durable state machine, policy, simulated system of record and bounded workers.
- `web.py`: loopback UI and process startup/recovery.
- `tests/`: focused unit, integration and process-recovery tests.
- `.env.example`: optional OpenRouter configuration; no credential is included.

To start over, stop the server first, then delete the specific `durable_state.sqlite3` database and its associated `.lock` files in this folder. This removes the simulated history; it does not revert any real-world action.
