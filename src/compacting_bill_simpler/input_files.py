from __future__ import annotations

from pathlib import Path


def resolve_maybe_compressed_csv(path: Path) -> Path:
    """Resolve a CSV input path, accepting either .csv or .csv.gz on disk.

    The configured path remains authoritative when it exists. If it does not
    exist, the function falls back to the compressed or uncompressed sibling
    when the paired form is available.
    """
    if path.exists():
        return path

    path_str = str(path)
    candidates: list[Path] = []
    if path_str.endswith(".csv.gz"):
        candidates.append(Path(path_str[:-3]))
    elif path_str.endswith(".csv"):
        candidates.append(Path(f"{path_str}.gz"))

    for candidate in candidates:
        if candidate.exists():
            return candidate

    return path
