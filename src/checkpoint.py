"""Checkpoint serialization for DocumentExtractionResult.

Saves per-file extraction results to disk so that crashed batch runs can
resume from where they left off. Uses pickle for full object-graph fidelity
and atomic writes to prevent corruption on crash.
"""

from __future__ import annotations

import logging
import os
import pickle
from pathlib import Path

from src.extraction import DocumentExtractionResult

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def save_checkpoint(result: DocumentExtractionResult, path: Path) -> None:
    """Serialize *result* to *path* atomically.

    Writes to a temporary file first, then atomically renames it to *path*.
    This prevents partial-write corruption if the process crashes mid-write.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    tmp_path = path.with_suffix(path.suffix + ".tmp")
    try:
        with open(tmp_path, "wb") as fh:
            pickle.dump(result, fh, protocol=pickle.HIGHEST_PROTOCOL)
        os.replace(tmp_path, path)  # atomic on POSIX
    except Exception:
        # Clean up temp file if write failed
        if tmp_path.exists():
            tmp_path.unlink(missing_ok=True)
        raise


def load_checkpoint(path: Path) -> DocumentExtractionResult | None:
    """Load a serialized DocumentExtractionResult from *path*.

    Returns ``None`` if the file doesn't exist or is corrupt.
    """
    path = Path(path)
    if not path.exists():
        return None
    try:
        with open(path, "rb") as fh:
            result = pickle.load(fh)
        if not isinstance(result, DocumentExtractionResult):
            logger.warning("Checkpoint %s is not a DocumentExtractionResult", path)
            return None
        return result
    except Exception:
        logger.warning("Failed to load checkpoint %s", path, exc_info=True)
        return None


def find_unprocessed_files(
    input_files: list[Path],
    checkpoint_dir: Path,
) -> list[Path]:
    """Return *input_files* that do NOT have a corresponding checkpoint.

    A checkpoint for ``"article1.xml"`` is expected at
    ``<checkpoint_dir>/article1.checkpoint``.
    """
    checkpoint_dir = Path(checkpoint_dir)
    unprocessed = []
    for f in input_files:
        ckpt_path = checkpoint_dir / f"{f.stem}.checkpoint"
        if not ckpt_path.exists():
            unprocessed.append(f)
    return unprocessed
