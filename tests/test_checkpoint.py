"""Test checkpoint serialization and resume."""

import pickle
from pathlib import Path

import pytest

from src.data import Extraction
from src.extraction import DocumentExtractionResult


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def sample_result():
    return DocumentExtractionResult(
        document_id="PMC12345",
        metadata={"doi": "10.1/test", "pmid": "12345"},
        extractions=[
            Extraction(
                extraction_class="Alternative",
                extraction_text="thymol",
                attributes={"standard_name": "thymol"},
            ),
        ],
        warnings=["test warning"],
    )


# ---------------------------------------------------------------------------
# Serialization tests
# ---------------------------------------------------------------------------


class TestCheckpointSerialization:
    def test_roundtrip_document_extraction_result(self, tmp_path, sample_result):
        """Serialize then deserialize a DocumentExtractionResult."""
        from src.checkpoint import save_checkpoint, load_checkpoint

        ckpt_path = tmp_path / "PMC12345.checkpoint"
        save_checkpoint(sample_result, ckpt_path)
        loaded = load_checkpoint(ckpt_path)

        assert loaded is not None
        assert loaded.document_id == "PMC12345"
        assert loaded.metadata == {"doi": "10.1/test", "pmid": "12345"}
        assert len(loaded.extractions) == 1
        assert loaded.extractions[0].extraction_text == "thymol"
        assert loaded.extractions[0].extraction_class == "Alternative"
        assert loaded.warnings == ["test warning"]

    def test_load_missing_checkpoint_returns_none(self, tmp_path):
        """load_checkpoint on a nonexistent path returns None."""
        from src.checkpoint import load_checkpoint

        assert load_checkpoint(tmp_path / "nonexistent.checkpoint") is None

    def test_load_corrupt_checkpoint_returns_none(self, tmp_path):
        """load_checkpoint on a corrupt file returns None rather than crashing."""
        from src.checkpoint import load_checkpoint

        ckpt_path = tmp_path / "corrupt.checkpoint"
        ckpt_path.write_text("not valid json or pickle")
        assert load_checkpoint(ckpt_path) is None

    def test_save_creates_parent_directory(self, tmp_path):
        """save_checkpoint creates the parent directory if needed."""
        from src.checkpoint import save_checkpoint
        from src.extraction import DocumentExtractionResult

        ckpt_path = tmp_path / "subdir" / "test.checkpoint"
        save_checkpoint(
            DocumentExtractionResult(document_id="test"), ckpt_path
        )
        assert ckpt_path.exists()


# ---------------------------------------------------------------------------
# Resume tests
# ---------------------------------------------------------------------------


class TestCheckpointResume:
    def test_find_unprocessed_files(self, tmp_path):
        """Files without checkpoints are returned as unprocessed."""
        from src.checkpoint import find_unprocessed_files, save_checkpoint
        from src.extraction import DocumentExtractionResult

        xml1 = tmp_path / "article1.xml"
        xml1.write_text("<article/>")
        xml2 = tmp_path / "article2.xml"
        xml2.write_text("<article/>")

        checkpoint_dir = tmp_path / "checkpoints"
        checkpoint_dir.mkdir()

        save_checkpoint(
            DocumentExtractionResult(document_id="article1"),
            checkpoint_dir / "article1.checkpoint",
        )

        unprocessed = find_unprocessed_files(
            [xml1, xml2], checkpoint_dir
        )
        assert len(unprocessed) == 1
        assert unprocessed[0].stem == "article2"

    def test_all_processed_returns_empty(self, tmp_path):
        """When all files have checkpoints, returns empty list."""
        from src.checkpoint import find_unprocessed_files, save_checkpoint
        from src.extraction import DocumentExtractionResult

        xml1 = tmp_path / "article1.xml"
        xml1.write_text("<article/>")

        checkpoint_dir = tmp_path / "checkpoints"
        checkpoint_dir.mkdir()

        save_checkpoint(
            DocumentExtractionResult(document_id="article1"),
            checkpoint_dir / "article1.checkpoint",
        )

        unprocessed = find_unprocessed_files([xml1], checkpoint_dir)
        assert len(unprocessed) == 0
