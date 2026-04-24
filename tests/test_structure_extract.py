from __future__ import annotations

import unittest
from pathlib import Path
import sys
import json

SRC_DIR = Path(__file__).resolve().parents[1] / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from compacting_bill_simpler.regulatory.config import PipelineConfig
from compacting_bill_simpler.regulatory.models import BillRecord, ChunkRecord, LegalBlock, SentenceRecord
from compacting_bill_simpler.regulatory.stages import (
    build_legal_blocks,
    build_summary,
    canonicalize_facts,
    classify_legal_blocks,
    consolidate_facts,
    refine_legal_blocks,
    validate_and_repair_facts,
)


class _FakeResponse:
    def __init__(self, payload: dict) -> None:
        self.choices = [type("Choice", (), {"message": type("Msg", (), {"content": json.dumps(payload)})()})()]


class _FakeCompletions:
    def __init__(self, payloads: list[dict]) -> None:
        self._payloads = payloads

    def create(self, **kwargs):  # noqa: ANN003
        del kwargs
        return _FakeResponse(self._payloads.pop(0))


class _FakeChat:
    def __init__(self, payloads: list[dict]) -> None:
        self.completions = _FakeCompletions(payloads)


class _FakeClient:
    def __init__(self, payloads: list[dict]) -> None:
        self.chat = _FakeChat(payloads)


class StructureExtractTests(unittest.TestCase):
    def test_build_legal_blocks_groups_scope_and_enforcement(self) -> None:
        cleaned = (
            "47-18-3202 Scope. This part applies to persons that conduct business in this state. "
            "It applies only when revenue exceeds twenty-five million dollars. "
            "47-18-3212 Enforcement. A court may impose a civil penalty of up to $7,500 for each violation."
        )
        bill = BillRecord("B1", "Test", None, cleaned, cleaned_text=cleaned)
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
        self.assertIn("47-18-3202 Scope.", structural_blocks[0].text)
        self.assertNotIn("47-18-3212 Enforcement.", structural_blocks[0].text)

    def test_build_legal_blocks_splits_mid_chunk_on_new_heading(self) -> None:
        cleaned = (
            "47-18-3202 Scope. This part applies to persons doing business in this state. "
            "47-18-3203 Consumer rights. A consumer may access personal information."
        )
        bill = BillRecord("B1", "Test", None, cleaned, cleaned_text=cleaned)
        sentences = [
            SentenceRecord("B1", 0, 0, cleaned, 30, 0, len(cleaned), 0),
        ]
        chunks = [
            ChunkRecord("B1", 0, cleaned, [0], 0, len(cleaned), sentences),
        ]

        blocks = build_legal_blocks(bill, sentences, chunks)

        self.assertEqual(len(blocks), 2)
        self.assertIn("47-18-3202 Scope.", blocks[0].text)
        self.assertNotIn("47-18-3203 Consumer rights.", blocks[0].text)
        self.assertIn("47-18-3203 Consumer rights.", blocks[1].text)

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

    def test_canonicalize_facts_preserves_conjunctive_branch_logic(self) -> None:
        bill = BillRecord("B1", "Test", None, "unused")
        facts = {
            "title": "Test Bill",
            "applicability": {
                "thresholds": [
                    {"type": "revenue", "operator": ">", "value": "25000000", "unit": "USD", "metric": "revenue", "text": "Exceed $25,000,000 in revenue"},
                    {"type": "consumer_count", "operator": ">=", "value": "25000", "unit": "consumers", "metric": "consumers_processed", "text": "At least 25,000 consumers"},
                    {"type": "other", "operator": ">", "value": "50", "unit": "percent", "metric": "gross_revenue_from_sale_of_personal_information", "text": "More than 50% of gross revenue from sale of personal information"},
                    {"type": "consumer_count", "operator": ">=", "value": "175000", "unit": "consumers", "metric": "consumers_processed_per_calendar_year", "text": "At least 175,000 consumers per calendar year"},
                ]
            },
            "powers": [],
            "sanctions": [],
            "prohibitions": [],
        }

        canonical = canonicalize_facts(bill, facts)
        thresholds = canonical["applicability"]["thresholds"]
        percent_row = next(row for row in thresholds if row.get("logic_tag") == "branch_1_revenue_share")
        branch_1_consumer = next(row for row in thresholds if row.get("logic_tag") == "branch_1_consumer_count")

        self.assertFalse(percent_row["standalone"])
        self.assertEqual(percent_row["branch_id"], "branch_1")
        self.assertEqual(branch_1_consumer["branch_id"], "branch_1")
        self.assertIn("revenue > 25000000 AND", canonical["applicability"]["condition_logic"])

    def test_build_summary_uses_enforcement_powers_when_sanctions_empty(self) -> None:
        facts = {
            "title": "Test Bill",
            "applicability": {
                "scope_summary": "Scope.",
                "thresholds": [],
                "condition_logic_summary": "Logic.",
                "excluded_entities": [],
            },
            "obligations": [],
            "prohibitions": [
                {"text": "A violation of this part shall not serve as the basis for a private right of action."}
            ],
            "powers": [
                {"text": "The attorney general and reporter has exclusive authority to enforce this part."},
                {"text": "The attorney general and reporter may issue a civil investigative demand."},
            ],
            "sanctions": [],
            "definitions": [],
        }

        summary = build_summary(facts, {"judge_verdict": "approved", "judge_score": 0.9, "judge_issues": []})

        self.assertIn("exclusive authority to enforce", summary)
        self.assertIn("private right of action", summary)

    def test_refine_legal_blocks_splits_only_targeted_parent(self) -> None:
        config = PipelineConfig(mode="live", dry_run=False)
        bill = BillRecord("B1", "Test", None, "unused")
        sentences = [
            SentenceRecord("B1", 0, 0, "A consumer may request access.", 6, 0, 30, 0),
            SentenceRecord("B1", 1, 0, "A consumer may request deletion.", 6, 31, 65, 1),
            SentenceRecord("B1", 2, 1, "A controller shall respond within 45 days.", 8, 66, 110, 2),
            SentenceRecord("B1", 3, 1, "A controller shall provide appeal instructions.", 8, 111, 165, 3),
        ]
        chunks = [
            ChunkRecord("B1", 0, "A consumer may request access. A consumer may request deletion.", [0, 1], 0, 65, sentences[:2]),
            ChunkRecord("B1", 1, "A controller shall respond within 45 days. A controller shall provide appeal instructions.", [2, 3], 66, 165, sentences[2:]),
        ]
        parent = LegalBlock(
            bill_id="B1",
            block_id="BLK-001",
            role="consumer_rights",
            heading="47-18-3203 Consumer rights",
            text=" ".join(sentence.text for sentence in sentences),
            level=1,
            chunk_ids=[0, 1],
            sentence_ids=[0, 1, 2, 3],
            char_start=0,
            char_end=165,
            source_mode="llm",
            confidence=0.9,
            secondary_roles=["controller_duties"],
            is_mixed=True,
            notes=["llm_role"],
        )
        client = _FakeClient(
            [
                {
                    "subblocks": [
                        {
                            "start_sentence_id": 0,
                            "end_sentence_id": 1,
                            "role": "consumer_rights",
                            "secondary_roles": [],
                            "is_mixed": False,
                            "heading_hint": "consumer rights core",
                            "confidence": 0.92,
                            "notes": ["rights cluster"],
                        },
                        {
                            "start_sentence_id": 2,
                            "end_sentence_id": 3,
                            "role": "controller_duties",
                            "secondary_roles": [],
                            "is_mixed": False,
                            "heading_hint": "controller response",
                            "confidence": 0.91,
                            "notes": ["duty cluster"],
                        },
                    ]
                }
            ]
        )

        refined = refine_legal_blocks(bill, [parent], sentences, chunks, config=config, llm_client=client)

        self.assertEqual(len(refined), 3)
        self.assertTrue(any(block.parent_block_id == "BLK-001" for block in refined))
        self.assertEqual([block.role for block in refined[1:]], ["consumer_rights", "controller_duties"])
        self.assertEqual(refined[1].sentence_ids, [0, 1])
        self.assertEqual(refined[2].sentence_ids, [2, 3])

    def test_validate_and_repair_facts_dry_run_keeps_validated_trace(self) -> None:
        config = PipelineConfig(mode="dry-run", dry_run=True)
        bill = BillRecord("B1", "Test", None, "unused", cleaned_text="This part does not apply to consumer reporting agencies under the Fair Credit Reporting Act.")
        sentences = [SentenceRecord("B1", 0, 0, bill.cleaned_text or "", 12, 0, 90, 0)]
        facts_raw = {
            "bill_id": "B1",
            "title": "Test",
            "applicability": {
                "covered_entities": ["controller"],
                "excluded_entities": [],
                "thresholds": [],
                "scope_summary": "Test.",
            },
            "obligations": [],
            "prohibitions": [],
            "powers": [],
            "sanctions": [],
            "rights": [],
            "eligibility_conditions": [],
            "definitions": [],
            "trace": {"scope_mode": "full_context", "effects_mode": "full_context", "no_legal_blocks": True},
        }

        facts_raw, facts_canonical, verification, repairs, facts_validated = validate_and_repair_facts(
            bill,
            sentences,
            facts_raw,
            config=config,
            llm_client=None,
        )

        self.assertIn("validation_applied", facts_validated["trace"])
        self.assertIn("applicability", verification)
        self.assertFalse(repairs["applicability"]["apply_repair"])
        self.assertEqual(facts_raw["bill_id"], facts_canonical["bill_id"])


if __name__ == "__main__":
    unittest.main()
