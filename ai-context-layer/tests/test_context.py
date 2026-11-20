import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from context_layer import AccessDenied, DataQualityError, Gateway, Store, ask_openrouter


class ContextTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.db = Path(self.tmp.name) / "test.sqlite3"
        self.store = Store(self.db)
        self.store.seed()
        self.gateway = Gateway(self.store)

    def test_scoped_context_provenance_and_minimization(self):
        result = self.gateway.context("agent-acme", "acme", "T-100")
        encoded = json.dumps(result)
        self.assertIn("Acme Robotics", encoded)
        self.assertNotIn("Zen Bikes", encoded)
        self.assertNotIn("private@zen.invalid", encoded)
        self.assertNotIn("internal@acme.invalid", encoded)
        self.assertEqual("acme", result["memory"]["namespace"])
        self.assertEqual(["Prefers concise status updates."], result["memory"]["preferences"])
        self.assertEqual("Simulated CRM", result["customer"]["provenance"]["source_system"])
        self.assertEqual("Support Operations", result["ticket"]["provenance"]["owner"])
        self.assertTrue(result["ticket"]["provenance"]["review_date"])

    def test_cross_customer_and_mismatched_ticket_are_denied(self):
        with self.assertRaises(AccessDenied):
            self.gateway.context("agent-acme", "zen", "T-200")
        with self.assertRaises(AccessDenied):
            self.gateway.context("agent-acme", "acme", "T-200")
        with self.assertRaises(AccessDenied):
            self.gateway.context("unknown", "acme", "T-100")
        with self.store.connect() as conn:
            self.assertEqual(3, conn.execute("SELECT COUNT(*) FROM audit WHERE outcome='denied'").fetchone()[0])

    def test_bad_data_fails_closed(self):
        with self.store.connect() as conn:
            conn.execute("UPDATE tickets SET priority='BAD' WHERE id='T-100'")
        with self.assertRaises(DataQualityError):
            self.gateway.context("agent-acme", "acme", "T-100")

    def test_proposal_requires_separate_approval_and_single_apply(self):
        proposal = self.gateway.propose("agent-acme", "acme", "T-100", "Page incident team")
        pending = self.gateway.pending_proposals("reviewer", "acme")
        self.assertEqual("Page incident team", pending[0]["action"])
        self.assertEqual(proposal, pending[0]["id"])
        with self.assertRaises(AccessDenied):
            self.gateway.pending_proposals("agent-acme", "acme")
        with self.store.connect() as conn:
            self.assertEqual("", conn.execute("SELECT next_action FROM tickets WHERE id='T-100'").fetchone()[0])
        with self.assertRaises(AccessDenied):
            self.gateway.decide("agent-acme", proposal, "approve", "")
        self.gateway.decide("reviewer", proposal, "approve", "Checked")
        with self.assertRaises(ValueError):
            self.gateway.decide("reviewer", proposal, "approve", "")
        with self.store.connect() as conn:
            ticket = conn.execute("SELECT next_action,version FROM tickets WHERE id='T-100'").fetchone()
            self.assertEqual(("Page incident team", 1), tuple(ticket))
            audit = conn.execute("SELECT outcome FROM audit WHERE operation='ticket.decide'").fetchall()
            self.assertEqual(["approve"], [r[0] for r in audit])

    def test_rejection_and_stale_proposals_do_not_write(self):
        first = self.gateway.propose("agent-acme", "acme", "T-100", "First")
        stale = self.gateway.propose("agent-acme", "acme", "T-100", "Second")
        rejected = self.gateway.propose("agent-acme", "acme", "T-100", "Third")
        self.gateway.decide("reviewer", rejected, "reject", "Not suitable")
        self.gateway.decide("reviewer", first, "approve", "")
        with self.assertRaisesRegex(ValueError, "changed"):
            self.gateway.decide("reviewer", stale, "approve", "")
        with self.store.connect() as conn:
            self.assertEqual("First", conn.execute("SELECT next_action FROM tickets WHERE id='T-100'").fetchone()[0])
            self.assertEqual("pending", conn.execute("SELECT status FROM proposals WHERE id=?", (stale,)).fetchone()[0])

    def test_unauthorized_writes_and_cross_customer_audit(self):
        with self.assertRaises(AccessDenied):
            self.gateway.propose("agent-zen", "zen", "T-200", "change")
        with self.assertRaises(AccessDenied):
            self.gateway.audit_log("agent-zen", "acme")

    def test_model_request_uses_only_provided_context(self):
        context = self.gateway.context("agent-acme", "acme", "T-100")
        with patch("openai.OpenAI") as factory:
            response = MagicMock()
            response.model = "test/free"
            response.choices[0].message.content = "Contact on-call (Support playbook)."
            response.usage.model_dump.return_value = {"prompt_tokens": 25}
            factory.return_value.chat.completions.create.return_value = response
            answer, metadata = ask_openrouter("What next?", context, "test/free", "key")
            sent = factory.return_value.chat.completions.create.call_args.kwargs
            self.assertIn("on-call", answer)
            self.assertEqual(25, metadata["usage"]["prompt_tokens"])
            self.assertIn("untrusted data", sent["messages"][0]["content"])
            self.assertNotIn("zen", sent["messages"][1]["content"].lower())
            self.assertNotIn("internal_email", sent["messages"][1]["content"])

    def test_truncated_model_answer_fails_explicitly(self):
        context = self.gateway.context("agent-acme", "acme", "T-100")
        with patch("openai.OpenAI") as factory:
            response = MagicMock()
            response.choices[0].message.content = "Partial answer"
            response.choices[0].finish_reason = "length"
            factory.return_value.chat.completions.create.return_value = response
            with self.assertRaisesRegex(ValueError, "truncated"):
                ask_openrouter("What next?", context, "test/free", "key")

    def test_cli_end_to_end_and_no_model_call_when_denied(self):
        project = Path(__file__).resolve().parents[1]
        command = [sys.executable, "main.py", "--db", str(self.db)]
        result = subprocess.run(command + ["context", "--actor", "agent-acme", "--customer", "acme",
                                           "--ticket", "T-100"], cwd=project, text=True, capture_output=True)
        self.assertEqual(0, result.returncode, result.stderr)
        self.assertEqual("P1", json.loads(result.stdout)["ticket"]["data"]["priority"])
        env = os.environ.copy()
        env.pop("OPENROUTER_API_KEY", None)
        denied = subprocess.run(command + ["ask", "--actor", "agent-acme", "--customer", "zen", "--ticket", "T-200",
                                            "--question", "What next?"], cwd=project, text=True, capture_output=True,
                                env=env)
        self.assertNotEqual(0, denied.returncode)
        self.assertIn("Not authorized", denied.stderr)

    def test_seed_does_not_overwrite_changes_or_duplicate_memory(self):
        with self.store.connect() as conn:
            conn.execute("UPDATE customers SET plan='Updated' WHERE id='acme'")
        self.store.seed()
        with self.store.connect() as conn:
            self.assertEqual("Updated", conn.execute("SELECT plan FROM customers WHERE id='acme'").fetchone()[0])
            self.assertEqual(1, conn.execute("SELECT COUNT(*) FROM memory WHERE customer_id='acme'").fetchone()[0])


if __name__ == "__main__":
    unittest.main()
