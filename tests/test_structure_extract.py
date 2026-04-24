from __future__ import annotations

import unittest
from pathlib import Path
import sys

SRC_DIR = Path(__file__).resolve().parents[1] / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from compacting_bill_simpler.regulatory.config import PipelineConfig
from compacting_bill_simpler.regulatory.models import BillRecord, ChunkRecord, SentenceRecord
from compacting_bill_simpler.regulatory.stages import build_legal_blocks, build_summary, classify_legal_blocks, consolidate_facts


class StructureExtractTests(unittest.TestCase):
    def test_build_legal_blocks_groups_scope_and_enforcement(self) -> None:
        bill = BillRecord("B1", "Test", None, "unused")
        config = PipelineConfig()
        sentences = [
            SentenceRecord("B1", 0, 0, "47-18-3202 Scope. This part applies to persons that conduct business in this state.", 12, 0, 84, 0),
            SentenceRecord("B1", 1, 0, "It applies only when revenue exceeds twenty-five million dollars.", 10, 85, 150, 1),
            SentenceRecord("B1", 2, 1, "47-18-3212 Enforcement. A court may impose a civil penalty of up to $7,500 for each violation.", 18, 151, 260, 2),
        ]
        chunks = [
            ChunkRecord("B1", 0, "47-18-3202 Scope. This part applies to persons that conduct business in this state. It applies only when revenue exceeds twenty-five million dollars.", [0, 1], 0, 150, sentences[:2]),
            ChunkRecord("B1", 1, "47-18-3212 Enforcement. A court may impose a civil penalty of up to $7,500 for each violation.", [2], 151, 260, [sentences[2]]),
        ]

        structural_blocks = build_legal_blocks(bill, sentences, chunks)
        blocks = classify_legal_blocks(bill, structural_blocks, config=config, llm_client=None)

        self.assertEqual([block.role for block in blocks], ["scope", "enforcement"])
        self.assertEqual(blocks[0].sentence_ids, [0, 1])
        self.assertEqual(blocks[1].sentence_ids, [2])
        self.assertEqual(structural_blocks[0].source_mode, "deterministic")

    def test_consolidate_facts_deduplicates_thresholds(self) -> None:
        bill = BillRecord("B1", "Test", None, "unused")
        config = PipelineConfig()
        del config
        blocks = []
        scope_trace = {
            "response": {
                "applicability": {
                    "applies_to_client": "uncertain",
                    "scope_type": "conditional",
                    "covered_entities": ["controller"],
                    "thresholds": [
                        {"type": "revenue", "operator": ">", "value": "25000000", "unit": "USD", "metric": "revenue"},
                        {"type": "revenue", "operator": ">", "value": "25000000", "unit": "USD", "metric": "revenue"},
                    ],
                    "scope_summary": "Revenue threshold applies.",
                }
            }
        }
        effects_trace = {"response": {"obligations": [], "prohibitions": [], "powers": [], "sanctions": []}}

        facts = consolidate_facts(bill, blocks, scope_trace, effects_trace)

        self.assertEqual(len(facts["applicability"]["thresholds"]), 1)
        self.assertEqual(facts["applicability"]["covered_entities"], ["controller"])

    def test_build_summary_includes_quantitative_thresholds(self) -> None:
        facts = {
            "title": "Test Bill",
            "applicability": {
                "scope_summary": "A threshold-based law.",
                "thresholds": [
                    {
                        "type": "revenue",
                        "operator": ">",
                        "value": "25000000",
                        "unit": "USD",
                        "metric": "revenue",
                        "text": "Exceed twenty-five million dollars ($25,000,000) in revenue.",
                    }
                ],
                "condition_logic_summary": "Applies only if revenue exceeds $25M.",
                "excluded_entities": [],
            },
            "obligations": [],
            "prohibitions": [],
            "sanctions": [],
            "definitions": [],
        }

        summary = build_summary(facts, {"judge_verdict": "approved", "judge_score": 0.9, "judge_issues": []})

        self.assertIn("THRESHOLDS - Exceed twenty-five million dollars ($25,000,000) in revenue.", summary)
        self.assertIn("LOGIC - Applies only if revenue exceeds $25M.", summary)


if __name__ == "__main__":
    unittest.main()
