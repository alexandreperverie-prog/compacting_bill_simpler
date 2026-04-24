from __future__ import annotations

import gzip
import json
import tempfile
from pathlib import Path
import sys
import unittest

SRC_DIR = Path(__file__).resolve().parents[1] / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from compacting_bill_simpler.regulatory.config import PipelineConfig
from compacting_bill_simpler.regulatory.orchestrator import allocate_trace_dir, trace_bill


class TraceTests(unittest.TestCase):
    def test_allocate_trace_dir_increments_versions(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            (root / "trace_v1").mkdir()
            (root / "trace_v7").mkdir()

            config = PipelineConfig(output_dir=root)
            trace_dir = allocate_trace_dir(config, trace_root=root)

            self.assertEqual(trace_dir.name, "trace_v8")

    def test_trace_bill_writes_reduced_stage_artifacts(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            csv_path = tmp_path / "bills.csv.gz"
            with gzip.open(csv_path, "wt", encoding="utf-8") as handle:
                handle.write("code,title,full_text\n")
                handle.write(
                    '0021,"Test Bill","AN ACT concerning finance. Be it enacted. SECTION 1. A company shall file a report."\n'
                )

            config = PipelineConfig(
                input_csv=csv_path,
                output_dir=tmp_path / "dataset" / "processed",
                limit=1,
            )
            trace_dir = tmp_path / "dataset" / "processed" / "trace_v1"

            result_dir = trace_bill(config, "0021", trace_dir)

            expected_files = {
                "00_manifest.json",
                "01_raw_text.txt",
                "01_bill.json",
                "02_cleaned_text.txt",
                "03_sentences.jsonl",
                "04_chunks.jsonl",
                "05_document_profile.json",
                "06_legal_blocks.json",
                "07_scope_extraction.json",
                "08_effects_extraction.json",
                "09_facts.json",
                "10_quality.json",
                "11_summary.txt",
                "12_preview.json",
                "13_pipeline_timing.json",
            }
            self.assertEqual(expected_files, {path.name for path in result_dir.iterdir()})

            manifest = json.loads((result_dir / "00_manifest.json").read_text(encoding="utf-8"))
            self.assertEqual(manifest["bill_id"], "0021")
            self.assertEqual(manifest["cleaned_text_artifact"], "02_cleaned_text.txt")
            self.assertEqual(manifest["legal_blocks_artifact"], "06_legal_blocks.json")
            self.assertEqual(manifest["facts_artifact"], "09_facts.json")
            self.assertEqual(manifest["timing_artifact"], "13_pipeline_timing.json")
