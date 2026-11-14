import os
import subprocess
import sys
import tempfile
import threading
import time
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from workflow import Manager, Store, offline_worker, openrouter_worker, validate_result


class WorkflowTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.store = Store(Path(self.temp.name) / "state.sqlite3")
        self.batch = self.store.create_batch()

    def counts(self):
        with self.store.connect() as conn:
            return conn.execute("SELECT COUNT(*) FROM simulated_actions").fetchone()[0]

    def test_batch_review_and_idempotent_actions(self):
        manager = Manager(self.store)
        manager.run(self.batch)
        batch = self.store.get_batch(self.batch)
        self.assertEqual("waiting_review", batch["status"])
        self.assertEqual(2, self.counts())
        pending = [i for i in batch["items"] if i["status"] == "waiting_review"]
        self.assertEqual(3, len(pending))
        with self.assertRaises(ValueError):
            self.store.apply_action(pending[0]["id"])
        self.store.review(pending[0]["id"], "approve", "Verified")
        self.store.review(pending[1]["id"], "reject", "Not authorized")
        self.store.review(pending[2]["id"], "approve", "")
        with self.assertRaises(ValueError):
            self.store.review(pending[0]["id"], "reject", "")
        manager.run(self.batch)
        self.assertEqual(4, self.counts())
        self.store.apply_action(pending[0]["id"])
        self.assertEqual(4, self.counts())
        batch = self.store.get_batch(self.batch)
        self.assertEqual("completed", batch["status"])
        self.assertEqual("reject", next(i for i in batch["items"] if i["id"] == pending[1]["id"])["decision"])
        self.assertEqual(3, len([e for e in batch["events"] if e["event"] == "human_decision"]))

    def test_no_lost_updates_and_bounded_concurrency(self):
        running = 0
        peak = 0
        guard = threading.Lock()

        def worker(item):
            nonlocal running, peak
            with guard:
                running += 1
                peak = max(peak, running)
            time.sleep(0.06)
            with guard:
                running -= 1
            return offline_worker(item)

        with patch("workflow.offline_worker", worker):
            Manager(self.store, max_concurrency=2).run(self.batch)
        self.assertEqual(2, peak)
        self.assertEqual(5, len([e for e in self.store.get_batch(self.batch)["events"] if e["event"] == "worker_result"]))

    def test_second_manager_respects_batch_lock(self):
        manager1, manager2 = Manager(self.store), Manager(Store(self.store.path))
        thread1 = threading.Thread(target=manager1.run, args=(self.batch,))
        thread2 = threading.Thread(target=manager2.run, args=(self.batch,))
        thread1.start()
        thread2.start()
        thread1.join(timeout=10)
        thread2.join(timeout=10)
        self.assertFalse(thread1.is_alive() or thread2.is_alive())
        batch = self.store.get_batch(self.batch)
        self.assertEqual(5, len([e for e in batch["events"] if e["event"] == "worker_result"]))
        self.assertTrue(all(item["attempts"] == 1 for item in batch["items"]))
        self.assertEqual(2, self.counts())

    def test_start_during_existing_run_is_rescheduled(self):
        manager = Manager(self.store)
        entered, release = threading.Event(), threading.Event()
        calls = []

        def fake_run(batch_id):
            calls.append(batch_id)
            if len(calls) == 1:
                entered.set()
                if not release.wait(5):
                    raise TimeoutError("test timed out waiting for release")

        with patch.object(manager, "run", side_effect=fake_run):
            manager.start(self.batch)
            self.assertTrue(entered.wait(5))
            manager.start(self.batch)
            release.set()
            thread = manager._threads.get(self.batch)
            if thread:
                thread.join(timeout=5)
        self.assertEqual([self.batch, self.batch], calls)

    def test_invalid_worker_results_fail_closed(self):
        with patch("workflow.offline_worker", return_value={"confidence": 2, "rationale": "bad"}):
            Manager(self.store).run(self.batch)
        self.assertEqual(0, self.counts())
        self.assertEqual("completed_with_errors", self.store.get_batch(self.batch)["status"])
        self.assertTrue(all(i["status"] == "failed" for i in self.store.get_batch(self.batch)["items"]))
        for value in (float("nan"), True, -0.1, 1.1):
            with self.assertRaises(ValueError):
                validate_result({"confidence": value, "rationale": "test"})

    def test_batch_configuration_cannot_change_across_restart(self):
        self.store.bind_mode(self.batch, "offline", "ignored")
        with self.assertRaisesRegex(ValueError, "different"):
            self.store.bind_mode(self.batch, "openrouter", "free")

    def test_openrouter_retries_rate_limits_and_validates_result(self):
        import openai

        class FakeRateLimit(Exception):
            pass

        response = MagicMock()
        response.choices[0].message.content = '{"confidence":0.9,"rationale":"clear request"}'
        client = MagicMock()
        client.chat.completions.create.side_effect = [FakeRateLimit(), response]
        events = []
        with patch.object(openai, "OpenAI", return_value=client), \
             patch.object(openai, "RateLimitError", FakeRateLimit), \
             patch("dotenv.load_dotenv"), patch.dict(os.environ, {"OPENROUTER_API_KEY": "test-key"}), \
             patch("workflow.time.sleep") as sleep:
            result = openrouter_worker({"text": "test"}, "free", lambda *args: events.append(args))
        self.assertEqual(0.9, result["confidence"])
        self.assertEqual([(1, "FakeRateLimit", 1)], events)
        sleep.assert_called_once_with(1)
        self.assertEqual(2, client.chat.completions.create.call_count)

    def test_openrouter_retry_limit_fails(self):
        import openai

        class FakeRateLimit(Exception):
            pass

        client = MagicMock()
        client.chat.completions.create.side_effect = FakeRateLimit()
        with patch.object(openai, "OpenAI", return_value=client), \
             patch.object(openai, "RateLimitError", FakeRateLimit), \
             patch("dotenv.load_dotenv"), patch.dict(os.environ, {"OPENROUTER_API_KEY": "test-key"}), \
             patch("workflow.time.sleep"):
            with self.assertRaises(FakeRateLimit):
                openrouter_worker({"text": "test"}, "free")
        self.assertEqual(3, client.chat.completions.create.call_count)

    def test_openrouter_invalid_json_is_retried_without_fallback(self):
        import openai

        invalid, valid = MagicMock(), MagicMock()
        invalid.choices[0].message.content = "not json"
        valid.choices[0].message.content = '{"confidence":0.65,"rationale":"uncertain"}'
        client = MagicMock()
        client.chat.completions.create.side_effect = [invalid, valid]
        events = []
        with patch.object(openai, "OpenAI", return_value=client), \
             patch("dotenv.load_dotenv"), patch.dict(os.environ, {"OPENROUTER_API_KEY": "test-key"}), \
             patch("workflow.time.sleep"):
            result = openrouter_worker({"text": "test"}, "free", lambda *args: events.append(args))
        self.assertEqual(0.65, result["confidence"])
        self.assertEqual([(1, "ValueError", 1)], events)

    def test_recover_after_process_termination(self):
        items = self.store.get_batch(self.batch)["items"]
        self.assertTrue(self.store.claim(items[0]["id"]))
        self.store.record_result(items[0], offline_worker(items[0]))
        child_code = """
import sys, time
from workflow import Store, batch_lock
s = Store(sys.argv[1])
with batch_lock(s, sys.argv[2]):
    s.apply_action(sys.argv[3])
    assert s.claim(sys.argv[4])
    print('ready', flush=True)
    time.sleep(60)
"""
        child = subprocess.Popen(
            [sys.executable, "-c", child_code, str(self.store.path), self.batch, items[0]["id"], items[1]["id"]],
            cwd=str(Path(__file__).resolve().parents[1]), stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        try:
            self.assertEqual("ready", child.stdout.readline().strip())
            child.kill()
            child.wait(timeout=10)
        finally:
            if child.poll() is None:
                child.kill()
                child.wait()
            child.stdout.close()
            child.stderr.close()
        self.assertEqual(1, self.counts())
        self.assertEqual("processing", self.store.get_batch(self.batch)["items"][1]["status"])
        Manager(self.store).run(self.batch)
        batch = self.store.get_batch(self.batch)
        self.assertEqual(2, self.counts())
        self.assertEqual(1, len([e for e in batch["events"] if e["event"] == "recovered"]))
        self.assertEqual(2, batch["items"][1]["attempts"])
        Manager(self.store).run(self.batch)
        self.assertEqual(2, self.counts())

    def test_review_persists_across_store_instances(self):
        Manager(self.store).run(self.batch)
        item = next(i for i in self.store.get_batch(self.batch)["items"] if i["status"] == "waiting_review")
        Store(self.store.path).review(item["id"], "approve", "accepted")
        Manager(Store(self.store.path)).run(self.batch)
        persisted = next(i for i in self.store.get_batch(self.batch)["items"] if i["id"] == item["id"])
        self.assertEqual(("completed", "approve", "accepted"), (persisted["status"], persisted["decision"], persisted["reason"]))

    def test_kill_before_action_commit_rolls_back_and_replay_applies_once(self):
        item = self.store.get_batch(self.batch)["items"][0]
        self.store.claim(item["id"])
        self.store.record_result(item, offline_worker(item))
        child_code = """
import sys, time
from workflow import Store, batch_lock
s = Store(sys.argv[1])
with batch_lock(s, sys.argv[2]):
    with s.connect() as conn:
        conn.execute('BEGIN IMMEDIATE')
        conn.execute('INSERT INTO simulated_actions(idempotency_key,item_id,action) VALUES(?,?,?)',
                     ('support-action:' + sys.argv[3], sys.argv[3], 'provide_guidance'))
        print('inserted', flush=True)
        time.sleep(60)
"""
        child = subprocess.Popen(
            [sys.executable, "-c", child_code, str(self.store.path), self.batch, item["id"]],
            cwd=str(Path(__file__).resolve().parents[1]), stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        try:
            self.assertEqual("inserted", child.stdout.readline().strip())
            child.kill()
            child.wait(timeout=10)
        finally:
            if child.poll() is None:
                child.kill()
                child.wait()
            child.stdout.close()
            child.stderr.close()
        self.assertEqual(0, self.counts())
        Manager(self.store).run(self.batch)
        self.assertEqual(2, self.counts())
        self.store.apply_action(item["id"])
        self.assertEqual(2, self.counts())


if __name__ == "__main__":
    unittest.main()
