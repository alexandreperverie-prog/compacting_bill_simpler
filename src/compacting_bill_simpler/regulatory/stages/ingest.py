from __future__ import annotations

import hashlib

import pandas as pd

from ...input_files import resolve_maybe_compressed_csv
from ..config import PipelineConfig
from ..models import BillRecord


def ingest_bills(config: PipelineConfig) -> list[BillRecord]:
    df = pd.read_csv(resolve_maybe_compressed_csv(config.input_csv))
    if config.limit is not None:
        df = df.head(config.limit)

    required = {config.text_column, config.id_column}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    # Detect whether id_column contains unique values; if not, fall back to row index
    id_col_values = df[config.id_column].astype(str) if config.id_column in df.columns else None
    use_index_as_id = id_col_values is None or id_col_values.duplicated().any()

    records: list[BillRecord] = []
    for row_idx, row in df.iterrows():
        raw_text = row.get(config.text_column)
        if not isinstance(raw_text, str) or not raw_text.strip():
            continue

        if use_index_as_id:
            bill_id = f"bill_{row_idx:04d}"
        else:
            bill_id = str(row.get(config.id_column))
        title = str(row.get(config.title_column, ""))
        jurisdiction = row.get("initiative_issuer") if "initiative_issuer" in df.columns else None
        if jurisdiction is not None and not isinstance(jurisdiction, str):
            jurisdiction = str(jurisdiction)

        records.append(
            BillRecord(
                bill_id=bill_id,
                title=title,
                jurisdiction=jurisdiction,
                raw_text=raw_text,
                text_hash=hashlib.sha256(raw_text.encode("utf-8")).hexdigest(),
            )
        )

    return records
