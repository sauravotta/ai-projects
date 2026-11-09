import csv
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import eval_poc


class MetricsTests(unittest.TestCase):
    def test_confusion_matrix_uses_pass_as_positive(self):
        metrics = eval_poc.confusion_metrics(
            ["pass", "pass", "fail", "fail"],
            ["pass", "fail", "pass", "fail"],
        )
        self.assertEqual(
            (metrics["true_positive"], metrics["false_positive"], metrics["true_negative"], metrics["false_negative"]),
            (1, 1, 1, 1),
        )
        self.assertEqual(metrics["precision"], 0.5)
        self.assertEqual(metrics["recall"], 0.5)

    def test_kappa_corrects_for_chance_and_handles_undefined_case(self):
        actual = ["pass", "pass", "pass", "fail"]
        always_pass = ["pass"] * 4
        self.assertEqual(eval_poc.cohens_kappa(actual, always_pass), 0.0)
        self.assertIsNone(eval_poc.cohens_kappa(["pass", "pass"], ["pass", "pass"]))

    def test_token_cost_uses_configured_per_million_rates(self):
        usage = {"prompt_tokens": 1_000_000, "completion_tokens": 500_000}
        with patch.dict(
            "os.environ",
            {
                "OPENROUTER_INPUT_COST_PER_1M": "0.15",
                "OPENROUTER_OUTPUT_COST_PER_1M": "0.60",
            },
        ):
            self.assertAlmostEqual(eval_poc.token_cost(usage), 0.45)

    def test_direct_check_normalizes_case_and_whitespace(self):
        self.assertEqual(eval_poc.direct_check("  Paris\n", "paris"), "pass")
        self.assertEqual(eval_poc.direct_check("Lyon", "Paris"), "fail")


class InputTests(unittest.TestCase):
    def test_dataset_rejects_duplicate_ids(self):
        row = {
            "id": "same",
            "question": "q",
            "reference_answer": "a",
            "answer": "a",
            "expected_label": "pass",
            "business_impact_usd": 0,
        }
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "dataset.jsonl"
            path.write_text(json.dumps(row) + "\n" + json.dumps(row) + "\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "duplicate id"):
                eval_poc.load_dataset(path)

    def test_human_csv_requires_valid_labels(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "ratings.csv"
            with path.open("w", newline="", encoding="utf-8") as file:
                writer = csv.writer(file)
                writer.writerow(["case_id", "reviewer_id", "label"])
                writer.writerow(["case-01", "reviewer", "maybe"])
            with self.assertRaisesRegex(ValueError, "label must be"):
                eval_poc.load_human_ratings(path)


class CliTests(unittest.TestCase):
    def test_offline_cli_writes_report_and_traces(self):
        with tempfile.TemporaryDirectory() as directory:
            output_dir = Path(directory) / "runs"
            exit_code = eval_poc.main(
                [
                    "--offline",
                    "--repeat",
                    "2",
                    "--output-dir",
                    str(output_dir),
                ]
            )
            self.assertEqual(exit_code, 0)
            reports = list(output_dir.glob("eval-*.json"))
            traces = list(output_dir.glob("eval-*.traces.jsonl"))
            self.assertEqual(len(reports), 1)
            self.assertEqual(len(traces), 1)
            report = json.loads(reports[0].read_text(encoding="utf-8"))
            self.assertEqual(report["repeat_count"], 2)
            self.assertIsNone(report["summary"]["judge"])
            self.assertEqual(report["summary"]["case_count"], 8)
            self.assertEqual(report["summary"]["human_agreement"]["human_vs_human"][0]["paired_cases"], 8)
            self.assertEqual(report["repeat_variability"]["direct_accuracy_stdev"], 0)
            self.assertEqual(len(traces[0].read_text(encoding="utf-8").splitlines()), 16)

    def test_judge_metrics_include_usage_and_human_agreement(self):
        cases = eval_poc.load_dataset(eval_poc.DEFAULT_DATASET)
        ratings = eval_poc.load_human_ratings(eval_poc.DEFAULT_HUMAN_RATINGS)

        def fake_judge(case, api_key, model, temperature):
            return case["expected_label"], {
                "reason": "Matches the reference.",
                "prompt_tokens": 10,
                "completion_tokens": 2,
                "reported_cost_usd": 0.001,
            }, 10.0

        with patch.object(eval_poc, "openrouter_judge", side_effect=fake_judge):
            report = eval_poc.run_evaluation(
                cases,
                ratings,
                repeats=1,
                api_key="test-key",
                model="test-model",
                temperature=0,
                experiment_id="test-experiment",
            )
        self.assertEqual(report["summary"]["judge"]["accuracy"], 1)
        self.assertAlmostEqual(report["summary"]["judge"]["llm_cost_usd"], 0.008)
        self.assertEqual(len(report["summary"]["human_agreement"]["judge_vs_human"]), 2)


if __name__ == "__main__":
    unittest.main()
