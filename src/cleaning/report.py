"""Cleaning report generation.

Produces a ``CleaningReport`` data object and/or a JSON report file summarizing
all changes made by the cleaning pipeline.  This provides auditability and
transparency for downstream data consumers.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from src.data import Extraction

logger = logging.getLogger(__name__)


@dataclass
class CleaningReport:
    """Tracks all changes across the cleaning pipeline phases."""

    summary: dict[str, Any] = field(default_factory=dict)
    phase_changes: dict[str, int] = field(default_factory=dict)
    phase_warnings: dict[str, list[str]] = field(default_factory=dict)
    dropped_entities: list[dict[str, str]] = field(default_factory=list)
    merged_entities: list[dict[str, Any]] = field(default_factory=list)
    stat_changes: list[dict[str, str]] = field(default_factory=list)

    @property
    def total_changes(self) -> int:
        return sum(self.phase_changes.values())

    @property
    def total_warnings(self) -> int:
        return sum(len(v) for v in self.phase_warnings.values())

    def add_phase(
        self,
        phase: str,
        changes: int = 0,
        warnings: list[str] | None = None,
    ) -> None:
        """Record changes from a pipeline phase."""
        if changes:
            self.phase_changes[phase] = changes
        if warnings:
            self.phase_warnings.setdefault(phase, []).extend(warnings)

    def add_dropped(
        self,
        entity_type: str,
        entity_name: str,
        reason: str,
    ) -> None:
        """Record a dropped entity."""
        self.dropped_entities.append({
            "entity_type": entity_type,
            "entity_name": entity_name,
            "reason": reason,
        })

    def add_merged(
        self,
        entity_type: str,
        canonical_name: str,
        merged_from: list[str],
    ) -> None:
        """Record a merged entity group."""
        self.merged_entities.append({
            "entity_type": entity_type,
            "canonical_name": canonical_name,
            "merged_from": merged_from,
        })

    def add_stat_change(
        self,
        field: str,
        old_value: str,
        new_value: str,
    ) -> None:
        """Record a specific value repair."""
        self.stat_changes.append({
            "field": field,
            "old": old_value,
            "new": new_value,
        })

    def log_summary(self) -> None:
        """Log a human-readable summary at INFO level."""
        lines = [
            "=" * 60,
            "CLEANING REPORT",
            "=" * 60,
            f"Entities: {self.summary.get('entities_before', '?')} → "
            f"{self.summary.get('entities_after', '?')}",
        ]
        for phase, count in self.phase_changes.items():
            lines.append(f"  {phase}: {count} changes")
        for phase, warns in self.phase_warnings.items():
            if warns:
                lines.append(f"  {phase} warnings: {len(warns)}")
        if self.dropped_entities:
            lines.append(f"  Dropped: {len(self.dropped_entities)} entities")
        if self.merged_entities:
            lines.append(f"  Merged: {len(self.merged_entities)} groups")
        logger.info("\n".join(lines))

    def to_dict(self) -> dict[str, Any]:
        """Export report as a JSON-serializable dict."""
        return {
            "summary": self.summary,
            "phase_changes": self.phase_changes,
            "total_warnings": self.total_warnings,
            "dropped_count": len(self.dropped_entities),
            "merged_count": len(self.merged_entities),
            "stat_repair_count": len(self.stat_changes),
            "dropped_entities": self.dropped_entities[:200],      # truncate for readability
            "merged_entities": self.merged_entities[:200],
            "stat_changes": self.stat_changes[:500],
            # Warning details grouped by phase
            "warnings_by_phase": {
                phase: warns[:100]   # truncate for readability
                for phase, warns in self.phase_warnings.items()
            },
        }

    def save(self, path: str | Path) -> Path:
        """Save the report as a JSON file.  Returns the path."""
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        data = self.to_dict()
        with open(p, "w", encoding="utf-8") as fh:
            json.dump(data, fh, indent=2, ensure_ascii=False, default=str)
        logger.info("Cleaning report saved to %s", p)
        return p


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def generate_cleaning_report(
    all_results: list[Any],
    output_dir: str | Path = "output/review",
) -> None:
    """Generate a summary cleaning report for a batch of results.

    Parameters
    ----------
    all_results:
        List of result objects with ``extractions`` and ``document_id`` attributes.
    output_dir:
        Directory where ``cleaning_report.json`` will be written.
    """
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    report = CleaningReport()

    # Aggregate across all results
    total_extractions = 0
    total_documents = len(all_results)
    for result in all_results:
        exts = getattr(result, "extractions", []) or []
        total_extractions += len(exts)

    report.summary = {
        "total_documents": total_documents,
        "total_entities": total_extractions,
        "timestamp": _now_iso(),
    }

    report_path = out / "cleaning_report.json"
    report.save(report_path)


def _now_iso() -> str:
    """Return current time in ISO format."""
    from datetime import datetime, timezone
    return datetime.now(timezone.utc).isoformat()
