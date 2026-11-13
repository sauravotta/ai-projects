import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

from rag import Chunk, Index, OpenRouter, cosine, decode_decision, fact_coverage, load_chunks, retrieval_metrics, run_agent


class FakeEmbedder:
    def embed(self, texts):
        return [[float("alpha" in text.lower()), float("beta" in text.lower()), 0.1] for text in texts]


class RagTests(unittest.TestCase):
    def setUp(self):
        self.chunks = [Chunk("a:1", "A", "alpha fact"), Chunk("b:1", "B", "beta fact")]
        self.index = Index(self.chunks, FakeEmbedder())

    def test_chunks_and_duplicate_ids(self):
        with tempfile.TemporaryDirectory() as folder:
            file = Path(folder) / "docs.json"
            file.write_text(json.dumps([{"id": "d", "title": "Title", "text": "first\n\nsecond"}]))
            self.assertEqual(["d:1", "d:2"], [chunk.id for chunk in load_chunks(file)])
            file.write_text(json.dumps([{"id": "d", "title": "A", "text": "x"}] * 2))
            with self.assertRaisesRegex(ValueError, "Duplicate"):
                load_chunks(file)

    def test_search_uses_cosine_similarity_and_validates_vectors(self):
        self.assertEqual("b:1", self.index.search("beta", 1)[0].chunk.id)
        with self.assertRaises(ValueError):
            cosine([1], [1, 2])
        with self.assertRaises(ValueError):
            cosine([0], [1])

    def test_agent_can_search_multiple_times_before_answering(self):
        decisions = iter([
            {"action": "search", "query": "alpha"},
            {"action": "search", "query": "beta"},
            {"action": "answer", "answer": "alpha and beta", "citations": ["a:1", "b:1"]},
        ])
        messages_seen = []

        def decide(messages):
            messages_seen.append(json.loads(json.dumps(messages)))
            return next(decisions), {"model": "fake", "usage": {}}

        result = run_agent("both?", self.index, decide, k=1)
        self.assertEqual(2, result["search_count"])
        self.assertEqual(["a:1", "b:1"], result["retrieved_ids"])
        self.assertIn("remaining_searches", messages_seen[1][-1]["content"])

    def test_agent_can_abstain_without_search(self):
        result = run_agent("unknown?", self.index, lambda _: ({"action": "abstain", "answer": "Unknown"}, {}))
        self.assertEqual("abstain", result["status"])
        self.assertEqual(0, result["search_count"])

    def test_unretrieved_and_missing_citations_are_rejected(self):
        for citations in ([], ["unknown:1"]):
            with self.subTest(citations=citations), self.assertRaisesRegex(ValueError, "retrieved"):
                run_agent("q", self.index, lambda _: ({"action": "answer", "answer": "x", "citations": citations}, {}))

    def test_search_budget_is_enforced(self):
        with self.assertRaisesRegex(ValueError, "budget"):
            run_agent("q", self.index, lambda _: ({"action": "search", "query": "alpha"}, {}), max_searches=1)

    def test_bad_actions_and_empty_search_are_rejected(self):
        for decision in ({"action": "other"}, {"action": "search", "query": ""}):
            with self.subTest(decision=decision), self.assertRaises(ValueError):
                run_agent("q", self.index, lambda _: (decision, {}))

    def test_repeated_hits_are_deduplicated_in_evidence(self):
        decisions = iter([
            {"action": "search", "query": "alpha"},
            {"action": "search", "query": "alpha"},
            {"action": "answer", "answer": "alpha", "citations": ["a:1"]},
        ])
        result = run_agent("q", self.index, lambda _: (next(decisions), {}), k=1)
        self.assertEqual(["a:1"], result["retrieved_ids"])

    def test_invalid_budget_and_question_are_rejected(self):
        for question, budget in (("", 1), ("q", 0)):
            with self.subTest(question=question, budget=budget), self.assertRaises(ValueError):
                run_agent(question, self.index, lambda _: ({}, {}), max_searches=budget)

    def test_evaluation_metrics_and_fact_coverage(self):
        metrics = retrieval_metrics(["a", "b", "c"], ["b", "c"], 3)
        self.assertAlmostEqual(2 / 3, metrics["precision"])
        self.assertEqual(1, metrics["recall"])
        self.assertEqual(0.5, metrics["reciprocal_rank"])
        self.assertEqual(0.5, fact_coverage("10 MINUTES", [["10 minutes"], ["manager"]]))

    def test_json_decisions_are_strict_and_support_fences(self):
        self.assertEqual("search", decode_decision('```json\n{"action":"search"}\n```')["action"])
        with self.assertRaisesRegex(ValueError, "invalid JSON"):
            decode_decision("not json")
        with self.assertRaisesRegex(ValueError, "JSON object"):
            decode_decision("[]")

    def test_openrouter_requests_structured_decisions_and_records_usage(self):
        with patch("openai.OpenAI") as factory:
            response = MagicMock()
            response.model = "provider/free-model"
            response.choices[0].message.content = '{"action":"search","query":"alpha"}'
            response.usage.model_dump.return_value = {"prompt_tokens": 10}
            factory.return_value.chat.completions.create.return_value = response
            client = OpenRouter("test-key", "test-model")
            decision, metadata = client([{"role": "user", "content": "question"}])
            self.assertEqual("search", decision["action"])
            self.assertEqual("provider/free-model", metadata["model"])
            self.assertEqual(10, metadata["usage"]["prompt_tokens"])
            request = factory.return_value.chat.completions.create.call_args.kwargs
            self.assertEqual({"type": "json_object"}, request["response_format"])

    def test_openrouter_reports_empty_response_with_finish_reason(self):
        with patch("openai.OpenAI") as factory:
            response = MagicMock()
            response.model = "provider/free-model"
            response.choices[0].message.content = ""
            response.choices[0].finish_reason = "length"
            factory.return_value.chat.completions.create.return_value = response
            with self.assertRaisesRegex(ValueError, "finish: length"):
                OpenRouter("test-key", "test-model")([])


if __name__ == "__main__":
    unittest.main()
