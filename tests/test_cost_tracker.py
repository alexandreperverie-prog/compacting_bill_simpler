from __future__ import annotations

from pathlib import Path
import sys
import tempfile
import unittest

SRC_DIR = Path(__file__).resolve().parents[1] / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from compacting_bill_simpler.regulatory.cost_tracker import UsageTracker


class CostTrackerTests(unittest.TestCase):
    def test_usage_tracker_computes_gpt51_costs(self) -> None:
        tracker = UsageTracker()
        tracker.record_chat(
            "gpt-5.1",
            input_tokens=1_000_000,
            cached_input_tokens=100_000,
            output_tokens=100_000,
            reasoning_tokens=20_000,
        )
        report = tracker.to_report()

        self.assertAlmostEqual(report["total_cost_usd"], 2.1375, places=6)
        self.assertEqual(report["models"]["gpt-5.1"]["tokens"]["cached_input"], 100_000)
        self.assertEqual(report["models"]["gpt-5.1"]["tokens"]["billable_input"], 900_000)

    def test_write_report_writes_json_file(self) -> None:
        tracker = UsageTracker()
        tracker.record_embedding("text-embedding-3-large", total_tokens=1000)

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "cost_report.json"
            tracker.write_report(path)
            self.assertTrue(path.exists())
            self.assertIn("total_cost_usd", path.read_text(encoding="utf-8"))


if __name__ == "__main__":
    unittest.main()
