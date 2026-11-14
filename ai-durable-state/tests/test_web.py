import http.client
import re
import subprocess
import sys
import tempfile
import threading
import time
import unittest
from http.server import HTTPServer
from pathlib import Path
from urllib.parse import urlencode

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from web import make_handler, _resume_batches
from workflow import Manager, Store


class WebTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.store = Store(Path(self.temp.name) / "web.sqlite3")
        self.manager = Manager(self.store)
        self.server = HTTPServer(("127.0.0.1", 0), make_handler(self.store, self.manager, "test-token"))
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()

    def tearDown(self):
        self.server.shutdown()
        self.server.server_close()
        self.thread.join()
        for thread in list(self.manager._threads.values()):
            thread.join(timeout=10)
        self.temp.cleanup()

    def request(self, method, path, values=None):
        connection = http.client.HTTPConnection("127.0.0.1", self.server.server_port, timeout=5)
        body = urlencode(values) if values is not None else None
        headers = {"Content-Type": "application/x-www-form-urlencoded"} if values is not None else {}
        connection.request(method, path, body=body, headers=headers)
        response = connection.getresponse()
        result = response.status, dict(response.getheaders()), response.read().decode()
        connection.close()
        return result

    def wait_for(self, batch, predicate):
        end = time.monotonic() + 10
        while time.monotonic() < end:
            data = self.store.get_batch(batch)
            if predicate(data):
                return data
            time.sleep(0.02)
        self.fail("Batch did not reach expected state")

    def test_http_create_review_and_duplicate_submission(self):
        status, _, page = self.request("GET", "/")
        self.assertEqual(200, status)
        self.assertIn('value="test-token"', page)
        self.assertEqual(403, self.request("POST", "/batches", {"csrf_token": "wrong"})[0])
        status, headers, _ = self.request("POST", "/batches", {"csrf_token": "test-token"})
        self.assertEqual(303, status)
        batch_id = headers["Location"].split("/")[-1]
        batch = self.wait_for(batch_id, lambda b: b["status"] == "waiting_review")
        item = next(i for i in batch["items"] if i["status"] == "waiting_review")
        status, _, page = self.request("GET", "/batch/" + batch_id)
        self.assertEqual(200, status)
        self.assertIn("Audit events", page)
        self.assertIn(item["text"], page)
        form = {"csrf_token": "test-token", "decision": "approve", "reason": "<script>unsafe</script>"}
        path = f"/items/{item['id']}/review"
        self.assertEqual(303, self.request("POST", path, form)[0])
        self.wait_for(batch_id, lambda b: any(i["id"] == item["id"] and i["status"] == "completed" for i in b["items"]))
        self.assertEqual(409, self.request("POST", path, form)[0])
        _, _, page = self.request("GET", "/batch/" + batch_id)
        self.assertIn("&lt;script&gt;unsafe&lt;/script&gt;", page)
        self.assertNotIn("<script>unsafe</script>", page)

    def test_restart_resumes_approved_item(self):
        batch_id = self.store.create_batch()
        self.manager.run(batch_id)
        batch = self.store.get_batch(batch_id)
        item = next(i for i in batch["items"] if i["status"] == "waiting_review")
        self.store.review(item["id"], "approve", "approved before restart")
        new_manager = Manager(Store(self.store.path))
        warnings = _resume_batches(Store(self.store.path), new_manager)
        self.assertEqual({}, warnings)
        self.wait_for(batch_id, lambda b: any(i["id"] == item["id"] and i["status"] == "completed" for i in b["items"]))
        for thread in list(new_manager._threads.values()):
            thread.join(timeout=10)

    def test_server_process_restart_preserves_pending_review(self):
        project = Path(__file__).resolve().parents[1]
        db = str(Path(self.temp.name) / "restart.sqlite3")

        def start_server():
            process = subprocess.Popen(
                [sys.executable, "-u", "web.py", "--db", db, "--port", "0"],
                cwd=str(project), stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, text=True)
            line = process.stderr.readline()
            match = re.search(r"http://127\.0\.0\.1:(\d+)/", line)
            if not match:
                process.terminate()
                process.wait(timeout=10)
                process.stderr.close()
                self.fail("Server failed to start: " + line)
            return process, int(match.group(1))

        def call(port, method, path, values=None):
            connection = http.client.HTTPConnection("127.0.0.1", port, timeout=5)
            body = urlencode(values) if values is not None else None
            headers = {"Content-Type": "application/x-www-form-urlencoded"} if values is not None else {}
            connection.request(method, path, body=body, headers=headers)
            response = connection.getresponse()
            result = response.status, dict(response.getheaders()), response.read().decode()
            connection.close()
            return result

        def stop(process):
            if process.poll() is None:
                process.terminate()
                process.wait(timeout=10)
            process.stderr.close()

        process, port = start_server()
        try:
            page = call(port, "GET", "/")[2]
            token = re.search(r'name="csrf_token" value="([^"]+)"', page).group(1)
            status, headers, _ = call(port, "POST", "/batches", {"csrf_token": token})
            self.assertEqual(303, status)
            batch_id = headers["Location"].split("/")[-1]
            store = Store(db)
            deadline = time.monotonic() + 10
            while time.monotonic() < deadline and store.get_batch(batch_id)["status"] != "waiting_review":
                time.sleep(0.02)
            batch = store.get_batch(batch_id)
            self.assertEqual("waiting_review", batch["status"])
            item = next(i for i in batch["items"] if i["status"] == "waiting_review")
        finally:
            stop(process)
        process, port = start_server()
        try:
            status, _, page = call(port, "GET", "/batch/" + batch_id)
            self.assertEqual(200, status)
            self.assertIn(item["text"], page)
            token = re.search(r'name="csrf_token" value="([^"]+)"', page).group(1)
            status, _, _ = call(port, "POST", f"/items/{item['id']}/review",
                                {"csrf_token": token, "decision": "approve", "reason": "after restart"})
            self.assertEqual(303, status)
            deadline = time.monotonic() + 10
            while time.monotonic() < deadline:
                current = next(i for i in store.get_batch(batch_id)["items"] if i["id"] == item["id"])
                if current["status"] == "completed":
                    break
                time.sleep(0.02)
            self.assertEqual(("completed", "approve", "after restart"),
                             (current["status"], current["decision"], current["reason"]))
        finally:
            stop(process)


if __name__ == "__main__":
    unittest.main()
