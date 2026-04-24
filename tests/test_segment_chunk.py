from __future__ import annotations

import unittest
from pathlib import Path
import sys

SRC_DIR = Path(__file__).resolve().parents[1] / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from compacting_bill_simpler.regulatory.models import BillRecord, SentenceRecord
from compacting_bill_simpler.regulatory.stages.chunk import chunk_bill
from compacting_bill_simpler.regulatory.stages.segment import fix_long_sentences
from compacting_bill_simpler.text_processing import count_cl100k_tokens, segment_into_sentences


class SegmentChunkTests(unittest.TestCase):
    def test_segment_into_sentences_returns_exact_offsets(self) -> None:
        text = (
            "Section 1. A company must file a report by January 1, 2026. "
            "This Act takes effect January 2, 2027."
        )

        rows = segment_into_sentences(text)

        self.assertGreaterEqual(len(rows), 2)
        for row in rows:
            start = row["char_start"]
            end = row["char_end"]
            self.assertEqual(text[start:end].strip(), row["text"])

    def test_fix_long_sentences_preserves_legal_markers(self) -> None:
        sent_text = "The company shall maintain records; provided that it updates them annually."
        sentence = SentenceRecord(
            bill_id="B1",
            sentence_id=0,
            chunk_hint_id=0,
            text=sent_text,
            n_tokens=count_cl100k_tokens(sent_text),
            char_start=10,
            char_end=86,
            source_sentence_id=0,
        )

        parts = fix_long_sentences([sentence], threshold=4, dry_run=True)

        self.assertEqual(len(parts), 2)
        self.assertEqual(parts[0].text, "The company shall maintain records;")
        self.assertTrue(parts[1].text.startswith("provided that"))
        self.assertEqual(parts[0].char_start, 10)
        self.assertEqual(parts[0].char_end, 45)
        self.assertEqual(parts[1].char_start, 46)
        self.assertEqual(parts[0].zone_label, "operative")

    def test_chunk_bill_keeps_overlap_sentence_ids(self) -> None:
        bill = BillRecord(bill_id="B1", title="Bill", jurisdiction=None, raw_text="unused")
        sentences = [
            SentenceRecord("B1", 0, 0, "One two", 2, 0, 7, 0),
            SentenceRecord("B1", 1, 0, "Three four", 2, 8, 18, 1),
            SentenceRecord("B1", 2, 0, "Five six", 2, 19, 27, 2),
            SentenceRecord("B1", 3, 0, "Seven eight", 2, 28, 39, 3),
        ]

        chunks = chunk_bill(bill, sentences, min_tokens=3, max_tokens=5, overlap=2)

        self.assertEqual(len(chunks), 3)
        self.assertEqual(chunks[0].source_sentence_ids, [0, 1])
        self.assertEqual(chunks[1].source_sentence_ids, [1, 2])
        self.assertEqual(chunks[2].source_sentence_ids, [2, 3])
        self.assertIn(chunks[0].zone_label, {"operative", "unknown"})
        self.assertGreaterEqual(chunks[0].operative_ratio, 0.0)


if __name__ == "__main__":
    unittest.main()
