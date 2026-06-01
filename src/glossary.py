"""Glossary index for ALTERNATIVE.tsv with exact, synonym, and fuzzy lookup."""

import csv
import re
from pathlib import Path


_CLASS_MAP = {
    "Plant_Extract": "Plant_Extract",
    "Trace_Element": "Trace_Element",
    "Organic_Acid": "Organic_Acid",
    "Probiotic": "Probiotic",
    "Polysaccharides and Oligosaccharides": "Polysaccharides_and_Oligosaccharides",
    "Enzyme": "Enzyme",
    "Bioactive Peptides": "Bioactive_Peptides",
    "Other": "Other",
    "Composite_Product": "Composite_Product",
}


def _normalize_class(raw: str) -> str:
    """Normalize Alternative_Class name to standard form."""
    return _CLASS_MAP.get(raw, raw)


def _is_element_grouped(text: str) -> bool:
    """Detect if the Alternative column uses element-grouped format.

    Element-grouped lines have patterns like 'Iron (Fe): ...' or 'Zinc (Zn): ...'
    i.e. a name followed by parenthesised content and a colon before the items.
    """
    return bool(re.search(r"[(（]\w+[)）]\s*[：:]", text))


def _split_commas_smart(text: str) -> list[str]:
    """Split on commas, but only those NOT inside parentheses.

    Tracks nesting depth so parenthetical lists like
    ``phytase (e.g., 3-phytase, 6-phytase)`` stay as one piece.
    """
    parts = []
    depth = 0
    current: list[str] = []
    for ch in text:
        if ch in "(（":
            depth += 1
            current.append(ch)
        elif ch in ")）":
            depth -= 1
            current.append(ch)
        elif ch == "," and depth == 0:
            parts.append("".join(current).strip())
            current = []
        else:
            current.append(ch)
    if current:
        parts.append("".join(current).strip())
    return parts


def _find_trailing_paren_group(text: str) -> tuple[str | None, str]:
    """Find the last balanced parentheses group at the end of *text*.

    Returns ``(content_inside, text_before)`` when a trailing ``(...)``
    group preceded by whitespace is found, or ``(None, text)`` otherwise.

    Handles nested parens::
        _find_trailing_paren_group("zinc acetate (Zn(CH₃COO)₂)")
        # => ("Zn(CH₃COO)₂", "zinc acetate")
    """
    end = len(text)
    while end > 0 and text[end - 1].isspace():
        end -= 1

    if end == 0 or text[end - 1] != ")":
        return None, text

    depth = 0
    for i in range(end - 1, -1, -1):
        if text[i] == ")":
            depth += 1
        elif text[i] == "(":
            depth -= 1
            if depth == 0:
                # Require whitespace before the opening paren (same as \s+\()
                if i > 0 and text[i - 1].isspace():
                    return text[i + 1 : end], text[:i].rstrip()
                break
    return None, text


class GlossaryIndex:
    """Index of alternative substances from ALTERNATIVE.tsv."""

    def __init__(self):
        self._exact: dict[str, dict] = {}
        self._synonyms: dict[str, str] = {}
        self._fuzzy_entries: list[dict] = []
        self._loaded = False

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def load(self, tsv_path: str | Path) -> None:
        """Parse the ALTERNATIVE.tsv file and build the index."""
        # Reset index dicts so load() is idempotent
        self._exact.clear()
        self._synonyms.clear()
        self._fuzzy_entries.clear()
        self._loaded = False

        path = Path(tsv_path)
        if not path.is_absolute():
            path = Path(__file__).resolve().parent.parent / tsv_path

        with open(path, encoding="utf-8") as f:
            reader = csv.reader(f, delimiter="\t")
            try:
                header = next(reader)  # skip header row
            except StopIteration:
                return

            alt_class_idx = 0
            subclass_idx = 1
            alt_col_idx = 2

            for row in reader:
                if not row or len(row) <= alt_col_idx or not row[alt_col_idx].strip():
                    continue
                raw_class = row[alt_class_idx].strip()
                subclass = row[subclass_idx].strip() if len(row) > subclass_idx else ""
                alt_text = row[alt_col_idx].strip()

                norm_class = _normalize_class(raw_class)
                grouped = _is_element_grouped(alt_text)
                self._parse_alternatives(alt_text, norm_class, subclass, grouped)

        self._loaded = True

    def lookup(self, name: str) -> dict:
        """Look up a substance name in the index.

        Returns a dict with keys: standard_name, class, subclass, match_source.
        Always returns a dict — when not found class is 'Other'.
        """
        if not name.strip():
            return {
                "standard_name": name,
                "class": "Other",
                "subclass": "",
                "match_source": "Other_未匹配",
            }

        match = self._exact.get(name)
        if match is not None:
            return dict(match, match_source="Exact_匹配")

        canonical = self._synonyms.get(name)
        if canonical is not None:
            match = self._exact.get(canonical)
            if match is not None:
                return dict(match, match_source="Synonym_同义词")

        best = None
        best_dist = 10  # > max allowed < 3

        for entry in self._fuzzy_entries:
            dist = self.levenshtein(name, entry["canonical"])
            if dist < best_dist:
                best_dist = dist
                best = entry

        if best_dist < 3:
            return dict(best["entry"], match_source="Fuzzy_模糊匹配")

        return {
            "standard_name": name,
            "class": "Other",
            "subclass": "",
            "match_source": "Other_未匹配",
        }

    # ------------------------------------------------------------------
    # Levenshtein distance
    # ------------------------------------------------------------------

    @staticmethod
    def levenshtein(a: str, b: str) -> int:
        """Compute the Levenshtein edit distance between two strings."""
        if len(a) < len(b):
            a, b = b, a

        prev = list(range(len(b) + 1))
        for i, ca in enumerate(a):
            curr = [i + 1]
            for j, cb in enumerate(b):
                cost = 0 if ca == cb else 1
                curr.append(min(curr[-1] + 1, prev[j + 1] + 1, prev[j] + cost))
            prev = curr
        return prev[-1]

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _parse_alternatives(
        self,
        text: str,
        norm_class: str,
        subclass: str,
        grouped: bool,
    ) -> None:
        """Parse the Alternative column text and register all items."""

        segments = re.split(r"[；;]", text)

        for seg in segments:
            seg = seg.strip()
            if not seg:
                continue

            if grouped:
                # Element-grouped format: "Header (sym): item1, item2"
                colon_char = "：" if "：" in seg else ":"
                colon_pos = seg.find(colon_char)
                if colon_pos != -1:
                    items_part = seg[colon_pos + 1 :].strip()
                else:
                    items_part = seg
            else:
                items_part = seg

            raw_items = _split_commas_smart(items_part)
            for raw in raw_items:
                if not raw:
                    continue
                self._register_item(raw, norm_class, subclass, grouped)

    def _register_item(
        self,
        raw: str,
        norm_class: str,
        subclass: str,
        grouped: bool,
    ) -> None:
        """Parse a single item and add it to the index."""

        raw = raw.strip()

        # --- (syn. X) pattern ---
        syn_match = re.search(r"\(syn\.\s*(.+?)\)", raw)
        if syn_match:
            synonym = syn_match.group(1).strip()
            standard_name = raw[: syn_match.start()].strip().rstrip(",").strip()
            self._add_entry(standard_name, standard_name, norm_class, subclass)
            self._synonyms[synonym] = standard_name
            return

        # --- trailing (ABBR) pattern (handles nested parens) ---
        abbr, canonical_name = _find_trailing_paren_group(raw)
        if abbr is not None:

            if grouped:
                # In element-grouped format, keep abbreviation in standard_name
                standard_name = raw
            else:
                # In simple format, strip abbreviation from standard_name
                standard_name = canonical_name

            self._add_entry(canonical_name, standard_name, norm_class, subclass)
            return

        # --- no parentheses ---
        self._add_entry(raw, raw, norm_class, subclass)

    def _add_entry(
        self,
        key: str,
        standard_name: str,
        norm_class: str,
        subclass: str,
    ) -> None:
        """Add an entry to the exact index and fuzzy entries list."""

        if not key:
            return

        entry = {
            "standard_name": standard_name,
            "class": norm_class,
            "subclass": subclass,
            "match_source": "Exact_匹配",
        }

        # Exact match index
        self._exact[key] = entry

        # Fuzzy matching uses the canonical (stripped) name as comparison basis
        self._fuzzy_entries.append({"canonical": key, "entry": entry})
