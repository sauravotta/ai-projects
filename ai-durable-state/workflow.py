"""Durable, bounded local support-request workflow."""
from __future__ import annotations

import fcntl
import json
import logging
import os
import sqlite3
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
from contextlib import contextmanager
from pathlib import Path

FIXTURES = [
    ("How do I change my notification settings?", "information"),
    ("Please remove my account and all my data.", "deletion"),
    ("Refund my annual subscription.", "refund"),
    ("Where can I find the getting-started guide?", "information"),
    ("I cannot sign in; please help me troubleshoot.", "access"),
]
POLICIES = {
    "information": ("low", "provide_guidance"),
    "access": ("low", "provide_guidance"),
    "refund": ("high", "queue_refund_review"),
    "deletion": ("high", "queue_deletion_review"),
}
CONFIDENCE_THRESHOLD = 0.8


class Store:
    def __init__(self, path: str | Path):
        self.path = Path(path).resolve()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.connect() as conn:
            conn.executescript("""
                CREATE TABLE IF NOT EXISTS batches (
                    id TEXT PRIMARY KEY, mode TEXT, model TEXT,
                    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
                );
                CREATE TABLE IF NOT EXISTS items (
                    id TEXT PRIMARY KEY, batch_id TEXT NOT NULL REFERENCES batches(id),
                    text TEXT NOT NULL, category TEXT NOT NULL, status TEXT NOT NULL,
                    risk TEXT, confidence REAL, rationale TEXT, action TEXT, error TEXT,
                    decision TEXT, reason TEXT, attempts INTEGER NOT NULL DEFAULT 0,
                    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                    updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
                );
                CREATE TABLE IF NOT EXISTS events (
                    id INTEGER PRIMARY KEY AUTOINCREMENT, batch_id TEXT NOT NULL,
                    item_id TEXT, event TEXT NOT NULL, details TEXT NOT NULL,
                    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
                );
                CREATE TABLE IF NOT EXISTS simulated_actions (
                    idempotency_key TEXT PRIMARY KEY, item_id TEXT NOT NULL UNIQUE,
                    action TEXT NOT NULL, created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
                );
            """)

    @contextmanager
    def connect(self):
        conn = sqlite3.connect(self.path, timeout=30)
        try:
            conn.row_factory = sqlite3.Row
            conn.execute("PRAGMA foreign_keys=ON")
            conn.execute("PRAGMA synchronous=FULL")
            with conn:
                yield conn
        finally:
            conn.close()

    @staticmethod
    def _event(conn: sqlite3.Connection, batch_id: str, item_id: str | None,
               event: str, details: dict) -> None:
        conn.execute("INSERT INTO events(batch_id,item_id,event,details) VALUES(?,?,?,?)",
                     (batch_id, item_id, event, json.dumps(details)))

    def create_batch(self) -> str:
        batch_id = uuid.uuid4().hex
        with self.connect() as conn:
            conn.execute("INSERT INTO batches(id) VALUES(?)", (batch_id,))
            for text, category in FIXTURES:
                item_id = uuid.uuid4().hex
                conn.execute("INSERT INTO items(id,batch_id,text,category,status) VALUES(?,?,?,?,?)",
                             (item_id, batch_id, text, category, "queued"))
                self._event(conn, batch_id, item_id, "queued", {})
        return batch_id

    def list_batches(self) -> list[dict]:
        with self.connect() as conn:
            ids = [row["id"] for row in conn.execute("SELECT id FROM batches ORDER BY created_at DESC, rowid DESC")]
        return [{"id": batch_id, "status": self.get_batch(batch_id)["status"]} for batch_id in ids]

    def get_batch(self, batch_id: str) -> dict:
        with self.connect() as conn:
            batch = conn.execute("SELECT * FROM batches WHERE id=?", (batch_id,)).fetchone()
            if batch is None:
                raise ValueError("Batch not found")
            items = [dict(row) for row in conn.execute(
                "SELECT * FROM items WHERE batch_id=? ORDER BY rowid", (batch_id,))]
            events = [dict(row) for row in conn.execute(
                "SELECT * FROM events WHERE batch_id=? ORDER BY id", (batch_id,))]
        statuses = {item["status"] for item in items}
        if statuses & {"queued", "processing", "approved", "ready_action"}:
            status = "running"
        elif "waiting_review" in statuses:
            status = "waiting_review"
        elif "failed" in statuses:
            status = "completed_with_errors"
        else:
            status = "completed"
        return {"id": batch_id, "status": status, "items": items, "events": events}

    def review(self, item_id: str, decision: str, reason: str) -> str:
        if decision not in ("approve", "reject") or len(reason) > 500:
            raise ValueError("Invalid decision or reason")
        with self.connect() as conn:
            conn.execute("BEGIN IMMEDIATE")
            row = conn.execute("SELECT * FROM items WHERE id=?", (item_id,)).fetchone()
            if row is None:
                raise ValueError("Item not found")
            if row["status"] != "waiting_review":
                raise ValueError("Item is not awaiting review")
            next_status = "approved" if decision == "approve" else "rejected"
            conn.execute("UPDATE items SET status=?,decision=?,reason=?,updated_at=CURRENT_TIMESTAMP WHERE id=?",
                         (next_status, decision, reason, item_id))
            self._event(conn, row["batch_id"], item_id, "human_decision",
                        {"decision": decision, "reason": reason})
            return row["batch_id"]

    def recover(self, batch_id: str) -> None:
        with self.connect() as conn:
            conn.execute("BEGIN IMMEDIATE")
            rows = conn.execute("SELECT id FROM items WHERE batch_id=? AND status='processing'", (batch_id,)).fetchall()
            for row in rows:
                conn.execute("UPDATE items SET status='queued',updated_at=CURRENT_TIMESTAMP WHERE id=?", (row["id"],))
                self._event(conn, batch_id, row["id"], "recovered", {"from": "processing"})

    def bind_mode(self, batch_id: str, mode: str, model: str) -> None:
        selected_model = model if mode == "openrouter" else None
        with self.connect() as conn:
            conn.execute("BEGIN IMMEDIATE")
            row = conn.execute("SELECT mode,model FROM batches WHERE id=?", (batch_id,)).fetchone()
            if row is None:
                raise ValueError("Batch not found")
            if row["mode"] is None:
                conn.execute("UPDATE batches SET mode=?,model=? WHERE id=?", (mode, selected_model, batch_id))
                self._event(conn, batch_id, None, "configured", {"mode": mode, "model": selected_model})
            elif row["mode"] != mode or row["model"] != selected_model:
                raise ValueError("Batch was created with a different worker mode or model")

    def pending(self, batch_id: str) -> list[dict]:
        with self.connect() as conn:
            return [dict(row) for row in conn.execute(
                "SELECT * FROM items WHERE batch_id=? AND status IN ('queued','approved','ready_action') ORDER BY rowid",
                (batch_id,))]

    def claim(self, item_id: str) -> bool:
        with self.connect() as conn:
            cursor = conn.execute(
                "UPDATE items SET status='processing',attempts=attempts+1,updated_at=CURRENT_TIMESTAMP "
                "WHERE id=? AND status='queued'", (item_id,))
            return cursor.rowcount == 1

    def record_result(self, item: dict, result: dict, duration_seconds: float | None = None) -> None:
        risk, action = POLICIES[item["category"]]
        confidence = result["confidence"]
        status = "waiting_review" if risk == "high" or confidence < CONFIDENCE_THRESHOLD else "ready_action"
        with self.connect() as conn:
            conn.execute("BEGIN IMMEDIATE")
            cursor = conn.execute(
                "UPDATE items SET status=?,risk=?,confidence=?,rationale=?,action=?,error=NULL,"
                "updated_at=CURRENT_TIMESTAMP WHERE id=? AND status='processing'",
                (status, risk, confidence, result["rationale"], action, item["id"]))
            if cursor.rowcount != 1:
                raise RuntimeError("Cannot checkpoint worker result: item is not processing")
            self._event(conn, item["batch_id"], item["id"], "worker_result",
                        {"risk": risk, "confidence": confidence, "status": status,
                         "duration_seconds": duration_seconds})

    def fail(self, item: dict, message: str) -> None:
        with self.connect() as conn:
            conn.execute("BEGIN IMMEDIATE")
            cursor = conn.execute(
                "UPDATE items SET status='failed',error=?,updated_at=CURRENT_TIMESTAMP WHERE id=? AND status='processing'",
                (message[:500], item["id"]))
            if cursor.rowcount != 1:
                raise RuntimeError("Cannot checkpoint failure: item is not processing")
            self._event(conn, item["batch_id"], item["id"], "worker_failed", {"error": message[:500]})

    def retry_event(self, item: dict, attempt: int, error_type: str, delay: int) -> None:
        with self.connect() as conn:
            self._event(conn, item["batch_id"], item["id"], "worker_retry",
                        {"attempt": attempt, "error_type": error_type, "delay_seconds": delay})

    def apply_action(self, item_id: str) -> None:
        with self.connect() as conn:
            conn.execute("BEGIN IMMEDIATE")
            row = conn.execute("SELECT * FROM items WHERE id=?", (item_id,)).fetchone()
            if row is None:
                raise ValueError("Item not found")
            if row["status"] == "completed":
                return
            if row["status"] not in ("approved", "ready_action"):
                raise ValueError("Item is not authorized for action")
            key = "support-action:" + item_id
            conn.execute("INSERT OR IGNORE INTO simulated_actions(idempotency_key,item_id,action) VALUES(?,?,?)",
                         (key, item_id, row["action"]))
            existing = conn.execute("SELECT item_id, action FROM simulated_actions WHERE idempotency_key=?", (key,)).fetchone()
            if existing is None or existing["item_id"] != item_id or existing["action"] != row["action"]:
                raise RuntimeError("Idempotency key conflicts with recorded action")
            conn.execute("UPDATE items SET status='completed',updated_at=CURRENT_TIMESTAMP WHERE id=?", (item_id,))
            self._event(conn, row["batch_id"], item_id, "action_applied", {"idempotency_key": key, "action": row["action"]})


def offline_worker(item: dict) -> dict:
    return {"confidence": 0.95 if item["category"] != "access" else 0.65,
            "rationale": f"Deterministic classification: {item['category']}."}


def validate_result(result: object) -> dict:
    if not isinstance(result, dict):
        raise ValueError("Worker response must be an object")
    confidence, rationale = result.get("confidence"), result.get("rationale")
    if isinstance(confidence, bool) or not isinstance(confidence, (int, float)) or not 0 <= confidence <= 1:
        raise ValueError("Worker confidence must be a number from 0 to 1")
    if not isinstance(rationale, str) or not rationale.strip() or len(rationale) > 1000:
        raise ValueError("Worker rationale must be nonempty and at most 1000 characters")
    return {"confidence": float(confidence), "rationale": rationale.strip()}


def openrouter_worker(item: dict, model: str, on_retry=None) -> dict:
    from dotenv import load_dotenv
    from openai import OpenAI, APIConnectionError, APITimeoutError, RateLimitError, InternalServerError

    load_dotenv(Path(__file__).resolve().parent / ".env")
    key = os.getenv("OPENROUTER_API_KEY")
    if not key:
        raise ValueError("OPENROUTER_API_KEY is required for OpenRouter mode")
    client = OpenAI(base_url="https://openrouter.ai/api/v1", api_key=key, timeout=45, max_retries=0)
    for attempt in range(3):
        try:
            response = client.chat.completions.create(
                model=model, temperature=0, max_tokens=2048, response_format={"type": "json_object"},
                messages=[{"role": "system", "content": (
                    "Assess your confidence in understanding this support request. Return only JSON "
                    'with numeric "confidence" from 0 to 1 and a short "rationale". '
                    "Do not make or execute any business decision.")},
                    {"role": "user", "content": item["text"]}])
            if not response.choices or not response.choices[0].message.content:
                raise ValueError("OpenRouter returned no content")
            try:
                return validate_result(json.loads(response.choices[0].message.content))
            except json.JSONDecodeError as exc:
                raise ValueError("OpenRouter returned invalid JSON") from exc
        except (APIConnectionError, APITimeoutError, RateLimitError, InternalServerError, ValueError) as exc:
            if attempt == 2:
                raise
            delay = 2 ** attempt
            if on_retry:
                on_retry(attempt + 1, type(exc).__name__, delay)
            time.sleep(delay)
    raise RuntimeError("OpenRouter retry budget exhausted")


@contextmanager
def batch_lock(store: Store, batch_id: str):
    lock_path = Path(str(store.path) + "." + batch_id + ".lock")
    with lock_path.open("a") as file:
        fcntl.flock(file, fcntl.LOCK_EX)
        try:
            yield True
        finally:
            fcntl.flock(file, fcntl.LOCK_UN)


class Manager:
    def __init__(self, store: Store, mode: str = "offline", max_concurrency: int = 3,
                 model: str = "nvidia/nemotron-3.5-lightning:free"):
        if mode not in ("offline", "openrouter") or not 1 <= max_concurrency <= 8:
            raise ValueError("Mode must be offline or openrouter; concurrency must be 1..8")
        self.store, self.mode, self.max_concurrency, self.model = store, mode, max_concurrency, model
        self._threads: dict[str, threading.Thread] = {}
        self._reschedule: set[str] = set()
        self._mutex = threading.Lock()

    def start(self, batch_id: str) -> None:
        self.store.bind_mode(batch_id, self.mode, self.model)
        with self._mutex:
            current = self._threads.get(batch_id)
            if current and current.is_alive():
                self._reschedule.add(batch_id)
                return
            thread = threading.Thread(target=self._run_scheduled, args=(batch_id,), daemon=True)
            self._threads[batch_id] = thread
            thread.start()

    def _run_scheduled(self, batch_id: str) -> None:
        try:
            while True:
                self.run(batch_id)
                with self._mutex:
                    if batch_id in self._reschedule:
                        self._reschedule.remove(batch_id)
                    else:
                        self._threads.pop(batch_id, None)
                        return
        except Exception:
            logging.exception("Batch %s stopped unexpectedly; restart the server to recover", batch_id)
            with self._mutex:
                self._threads.pop(batch_id, None)
                self._reschedule.discard(batch_id)

    def run(self, batch_id: str) -> None:
        with batch_lock(self.store, batch_id) as acquired:
            if not acquired:
                return
            self.store.bind_mode(batch_id, self.mode, self.model)
            self.store.recover(batch_id)
            while True:
                items = self.store.pending(batch_id)
                if not items:
                    return
                with ThreadPoolExecutor(max_workers=self.max_concurrency) as executor:
                    futures = [executor.submit(self._run_item, item) for item in items]
                    for future in as_completed(futures):
                        future.result()

    def _run_item(self, item: dict) -> None:
        if item["status"] in ("approved", "ready_action"):
            self.store.apply_action(item["id"])
            return
        if not self.store.claim(item["id"]):
            return
        started = time.monotonic()
        try:
            result = (offline_worker(item) if self.mode == "offline" else
                      openrouter_worker(item, self.model,
                                        lambda attempt, error, delay: self.store.retry_event(item, attempt, error, delay)))
            result = validate_result(result)
        except Exception as exc:
            logging.error("Worker failed for item %s: %s", item["id"], type(exc).__name__)
            safe_messages = {
                "OPENROUTER_API_KEY is required for OpenRouter mode",
                "OpenRouter returned no content",
                "OpenRouter returned invalid JSON",
                "Worker response must be an object",
                "Worker confidence must be a number from 0 to 1",
                "Worker rationale must be nonempty and at most 1000 characters",
            }
            detail = str(exc) if isinstance(exc, ValueError) and str(exc) in safe_messages else type(exc).__name__
            status = getattr(exc, "status_code", None)
            if isinstance(status, int):
                detail += f" (HTTP {status})"
            self.store.fail(item, f"Worker failed: {detail}")
            return
        self.store.record_result(item, result, round(time.monotonic() - started, 3))
