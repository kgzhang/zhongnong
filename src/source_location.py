"""SourceLocationResolver — derive source_location from document structure."""
import re
from src.data import Extraction

_TABLE_PATTERN = re.compile(r"(Table|Tab\.)\s*\d+", re.IGNORECASE)
_FIGURE_PATTERN = re.compile(r"(Figure|Fig\.)\s*\d+", re.IGNORECASE)


class SourceLocationResolver:
    """Derive a human-readable source location for an extraction.

    Combines the section identifier with nearby table/figure references
    found in the surrounding text.
    """

    def resolve_location(
        self,
        extraction: Extraction,
        section_id: str,
        section_text: str,
        char_offset: int,
    ) -> str:
        """Return a source location string for *extraction*.

        Always includes *section_id*.  When the extraction has a character
        interval, the surrounding 200 characters of *section_text* are
        scanned for table/figure mentions which are appended.
        """
        parts = [section_id]
        ci = extraction.char_interval
        if ci is not None and ci.start_pos is not None:
            nearby_start = max(0, ci.start_pos - 200)
            nearby_end = min(len(section_text), ci.end_pos + 200)
            nearby_text = section_text[nearby_start:nearby_end]
            details = []
            tm = _TABLE_PATTERN.search(nearby_text)
            fm = _FIGURE_PATTERN.search(nearby_text)
            if tm:
                details.append(tm.group(0))
            if fm:
                details.append(fm.group(0))
            if details:
                parts.append(", ".join(details))
        return ", ".join(parts)

    def resolve_batch(
        self,
        extractions,
        section_id: str,
        section_text: str,
        char_offset: int,
    ) -> None:
        """Set ``source_location`` on every extraction in *extractions*."""
        for ext in extractions:
            ext.attributes = dict(ext.attributes or {})
            ext.attributes["source_location"] = self.resolve_location(
                ext, section_id, section_text, char_offset
            )
