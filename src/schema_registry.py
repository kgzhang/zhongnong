"""Schema Registry — loads YAML config files and provides a typed API."""

from __future__ import annotations

import csv
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from src.data import Extraction

# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass
class AttributeDef:
    name: str
    type: str  # string|integer|float|boolean|text|enum|json_array
    required: bool
    source: str = "llm"  # llm|align|structure|post
    enum_values: list[str] | None = None


@dataclass
class ReferenceDef:
    name: str
    target_entity: str
    target_field: str
    edge_type: str


@dataclass
class InlineRelationDef:
    name: str
    target: list[str]
    via_field: str
    multiple: bool = False


@dataclass
class VocabularyBinding:
    source_path: str
    key_field: str
    class_field: str | None = None
    subclass_field: str | None = None
    match_fields: list[str] = field(default_factory=list)
    match_mode: str = "exact"
    inject_in_prompt: bool = True
    value_delimiter: str | None = None  # For TSV cells with comma-separated values


@dataclass
class EntityDef:
    name: str
    display_name: str
    description: str
    primary_text: str
    extraction_guidance: str = ""
    examples: list[dict] = field(default_factory=list)
    notes: str = ""
    attributes: list[AttributeDef] = field(default_factory=list)
    references: list[ReferenceDef] = field(default_factory=list)
    inline_relations: list[InlineRelationDef] = field(default_factory=list)
    vocabulary: VocabularyBinding | None = None
    dedup_mode: str = "exact"  # "exact", "fuzzy", or "article" (article-scoped)


@dataclass
class RelationDef:
    name: str
    description: str
    source: str | list[str]
    target: str | list[str]
    cardinality: str = "many_to_one"
    co_extracted: bool = False


@dataclass
class GateDef:
    entity: str
    condition: str
    on_fail: str


@dataclass
class ExtractionPhase:
    name: str
    description: str
    extracts: list[str]
    sections: list[str] = field(default_factory=list)
    context_from: list[str] = field(default_factory=list)
    gate: GateDef | None = None


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _levenshtein_distance(a: str, b: str) -> int:
    """Compute Levenshtein distance between two strings."""
    if len(a) < len(b):
        return _levenshtein_distance(b, a)
    if len(b) == 0:
        return len(a)
    prev_row = list(range(len(b) + 1))
    for i, ca in enumerate(a):
        curr_row = [i + 1]
        for j, cb in enumerate(b):
            insertions = prev_row[j + 1] + 1
            deletions = curr_row[j] + 1
            substitutions = prev_row[j] + (0 if ca == cb else 1)
            curr_row.append(min(insertions, deletions, substitutions))
        prev_row = curr_row
    return prev_row[-1]


def _normalize(text: str) -> str:
    """Lowercase, strip whitespace/punctuation, collapse internal whitespace."""
    t = text.strip().lower()
    t = re.sub(r"[^\w\s]", "", t)
    t = re.sub(r"\s+", " ", t)
    return t


# ---------------------------------------------------------------------------
# Vocabulary
# ---------------------------------------------------------------------------


class Vocabulary:
    """TSV-based candidate word list with exact and fuzzy matching."""

    def __init__(self, tsv_path: Path, binding: VocabularyBinding):
        self.entries: list[dict] = []
        self.by_name: dict[str, dict] = {}
        self.binding = binding
        self._load(tsv_path)

    def _load(self, path: Path) -> None:
        if not path.exists():
            return
        delimiter = getattr(self.binding, "value_delimiter", None)
        with open(path, encoding="utf-8") as fh:
            reader = csv.DictReader(fh, delimiter="\t")
            for row in reader:
                self.entries.append(dict(row))
                key_field = self.binding.key_field
                if key_field in row:
                    raw = row[key_field].strip()
                    if delimiter and delimiter in raw:
                        for part in raw.split(delimiter):
                            name = self._clean_name(part)
                            if name:
                                self.by_name[name.lower()] = dict(row)
                    else:
                        name = self._clean_name(raw)
                        if name:
                            self.by_name[name.lower()] = dict(row)

    @staticmethod
    def _clean_name(raw: str) -> str:
        """Extract base name, stripping parentheticals like '(THY)' or '(FeSO₄)'."""
        name = raw.strip()
        name = re.sub(r'\s*\([^)]*\)\s*$', '', name).strip()
        return name

    def lookup(self, name: str) -> dict | None:
        """Case-insensitive exact match."""
        key = name.strip().lower()
        return self.by_name.get(key)

    def fuzzy_match(self, name: str) -> dict | None:
        """Normalize -> substring -> Levenshtein <= 3."""
        norm = _normalize(name)
        if not norm:
            return None

        # First try substring match
        for key, entry in self.by_name.items():
            key_norm = _normalize(key)
            if norm in key_norm or key_norm in norm:
                return entry

        # Then Levenshtein
        best = None
        best_dist = 1000
        for key, entry in self.by_name.items():
            key_norm = _normalize(key)
            dist = _levenshtein_distance(norm, key_norm)
            if dist < best_dist and dist <= 3:
                best_dist = dist
                best = entry
        return best

    def format_for_prompt(self, summary_only: bool = True) -> str:
        """Render vocabulary for prompt injection.

        When ``summary_only=True`` (default), produces a compact category
        summary instead of the full list — suitable for large vocabularies.
        """
        if not self.entries:
            return ""

        if summary_only:
            return self._format_summary()

        lines = ["## Candidate Vocabulary"]
        for entry in self.entries:
            name = entry.get(self.binding.key_field, "")
            cls = entry.get(self.binding.class_field or "", "")
            subclass = entry.get(self.binding.subclass_field or "", "")
            parts = [f"  {name}"]
            if cls:
                parts.append(f"  [{cls}]")
            if subclass:
                parts.append(f"  ({subclass})")
            lines.append("".join(parts))
        return "\n".join(lines)

    def _format_summary(self) -> str:
        """Build a compact category overview for prompt injection.

        Lists classes and subclasses with example names, avoiding a dump
        of every individual entry.
        """
        classes: dict[str, dict[str, list[str]]] = {}
        for entry in self.entries:
            cls = entry.get(self.binding.class_field or "", "Uncategorised")
            sub = entry.get(self.binding.subclass_field or "", "general")
            name = entry.get(self.binding.key_field, "")
            # Extract first 2-3 example names from comma-separated lists
            if self.binding.value_delimiter and self.binding.value_delimiter in name:
                names = [n.strip().split("(")[0].strip()
                         for n in name.split(self.binding.value_delimiter)[:3]]
            else:
                names = [name.strip()]
            classes.setdefault(cls, {}).setdefault(sub, []).extend(names)

        lines = ["## Candidate Vocabulary (categories only — full list used for post-matching)"]
        for cls_name, subclasses in sorted(classes.items()):
            lines.append(f"\n### {cls_name}")
            for sub_name, examples in sorted(subclasses.items()):
                unique = list(dict.fromkeys(examples))[:3]  # deduplicate, max 3
                lines.append(f"  [{sub_name}] e.g. {', '.join(unique)}")
        return "\n".join(lines)

# ---------------------------------------------------------------------------
# Schema Registry
# ---------------------------------------------------------------------------


class SchemaRegistry:
    """Loads YAML schema configs and provides a typed query API."""

    def __init__(self, config_dir: str | Path = "schemas"):
        self.config_dir = Path(config_dir)
        self._entities: dict[str, EntityDef] = {}
        self._relations: dict[str, RelationDef] = {}
        self._phases: list[ExtractionPhase] = []
        self._vocabularies: dict[str, Vocabulary] = {}
        self._load()

    # -- Loading ----------------------------------------------------------

    def _load(self) -> None:
        self._load_entities()
        self._load_relations()
        self._load_phases()
        self._load_vocabularies()

    def _load_entities(self) -> None:
        path = self.config_dir / "entities.yaml"
        if not path.exists():
            return
        with open(path, encoding="utf-8") as fh:
            data = yaml.safe_load(fh)
        for name, raw in (data.get("entities") or {}).items():
            ed = self._parse_entity(name, raw)
            self._entities[name] = ed

    def _parse_entity(self, name: str, raw: dict) -> EntityDef:
        attrs = []
        for a in raw.get("attributes") or []:
            attrs.append(AttributeDef(
                name=a["name"],
                type=a["type"],
                required=a.get("required", False),
                source=a.get("source", "llm"),
                enum_values=a.get("values"),
            ))
        refs = []
        for r in raw.get("references") or []:
            refs.append(ReferenceDef(
                name=r["name"],
                target_entity=r["target_entity"],
                target_field=r["target_field"],
                edge_type=r["edge_type"],
            ))
        inlines = []
        for ir in raw.get("inline_relations") or []:
            inlines.append(InlineRelationDef(
                name=ir["name"],
                target=ir["target"] if isinstance(ir["target"], list) else [ir["target"]],
                via_field=ir["via_field"],
                multiple=ir.get("multiple", False),
            ))
        vocab = None
        if "vocabulary" in raw:
            v = raw["vocabulary"]
            vocab = VocabularyBinding(
                source_path=v["source"],
                key_field=v["key_field"],
                class_field=v.get("class_field"),
                subclass_field=v.get("subclass_field"),
                match_fields=v.get("match_fields", []),
                match_mode=v.get("match_mode", "exact"),
                inject_in_prompt=v.get("inject_in_prompt", True),
                value_delimiter=v.get("value_delimiter"),
            )
        # Read dedup_mode from top-level YAML field (preferred) or notes (legacy)
        raw_notes = raw.get("notes", "")
        dedup_mode = raw.get("dedup_mode", "exact")
        if dedup_mode == "exact" and "dedup_mode:" in raw_notes:
            # Legacy fallback: parse from notes for backward compatibility
            import re as _re2
            dm_match = _re2.search(r"dedup_mode\s*:\s*(\S+)", raw_notes)
            if dm_match:
                dedup_mode = dm_match.group(1).strip()
        return EntityDef(
            name=name,
            display_name=raw.get("display_name", name),
            description=raw.get("description", ""),
            primary_text=raw.get("primary_text", ""),
            extraction_guidance=raw.get("extraction_guidance", ""),
            examples=raw.get("examples", []),
            notes=raw_notes,
            attributes=attrs,
            references=refs,
            inline_relations=inlines,
            vocabulary=vocab,
            dedup_mode=dedup_mode,
        )

    def _load_relations(self) -> None:
        path = self.config_dir / "relations.yaml"
        if not path.exists():
            return
        with open(path, encoding="utf-8") as fh:
            data = yaml.safe_load(fh)
        for name, raw in (data.get("relations") or {}).items():
            source = raw["source"]
            target = raw["target"]
            rd = RelationDef(
                name=name,
                description=raw.get("description", ""),
                source=source if isinstance(source, list) else [source],
                target=target if isinstance(target, list) else [target],
                cardinality=raw.get("cardinality", "many_to_one"),
                co_extracted=raw.get("co_extracted", False),
            )
            self._relations[name] = rd

    def _load_phases(self) -> None:
        path = self.config_dir / "extraction_phases.yaml"
        if not path.exists():
            return
        with open(path, encoding="utf-8") as fh:
            data = yaml.safe_load(fh)
        for name, raw in (data.get("phases") or {}).items():
            gate = None
            if "gate" in raw:
                g = raw["gate"]
                gate = GateDef(
                    entity=g["entity"],
                    condition=g.get("condition", ""),
                    on_fail=g.get("on_fail", "skip_article"),
                )
            ep = ExtractionPhase(
                name=name,
                description=raw.get("description", ""),
                extracts=raw.get("extracts", []),
                sections=raw.get("sections", []),
                context_from=raw.get("context_from", []),
                gate=gate,
            )
            self._phases.append(ep)

    def _load_vocabularies(self) -> None:
        """Load TSV vocabulary files for entities that have vocabulary bindings."""
        for name, ed in self._entities.items():
            if ed.vocabulary:
                tsv_path = self.config_dir.parent / ed.vocabulary.source_path
                self._vocabularies[name] = Vocabulary(tsv_path, ed.vocabulary)

    # -- Public API -------------------------------------------------------

    def entity_def(self, name: str) -> EntityDef:
        return self._entities[name]

    def relation_def(self, name: str) -> RelationDef:
        return self._relations[name]

    def phase_defs(self) -> list[ExtractionPhase]:
        return list(self._phases)

    def build_section_routing(self) -> list[dict]:
        """Return ordered section-routing configuration from phase definitions.

        Each entry is a dict with:
          - ``section``: canonical section key (e.g. "body", "methods", "results")
          - ``entity_names``: list of entity types to extract in this phase
          - ``context_from``: list of prior phase names whose results are injected
            as reference context

        The results can be used by adapters to route entity types to specific
        sections without hardcoding entity lists.
        """
        routing = []
        for phase in self._phases:
            # Map phase section names to canonical keys
            section = "body"
            if phase.sections:
                section = phase.sections[0]
            routing.append({
                "phase": phase.name,
                "section": section,
                "entity_names": list(phase.extracts),
                "context_from": list(phase.context_from),
            })
        return routing

    def get_global_types(self) -> frozenset[str]:
        """Return entity type names whose ``dedup_mode`` is NOT ``"article"``.

        These entity types are shared across articles (global scope) —
        deduplicated by name rather than by name+PMID.  Entity types with
        ``dedup_mode: article`` are article-specific and will NOT appear in
        the returned set.
        """
        return frozenset(
            name for name, ed in self._entities.items()
            if ed.dedup_mode != "article"
        )

    def all_entity_names(self) -> list[str]:
        return list(self._entities.keys())

    def all_relation_names(self) -> list[str]:
        return list(self._relations.keys())

    def vocabulary(self, entity_name: str) -> Vocabulary | None:
        return self._vocabularies.get(entity_name)

    def llm_output_fields(self, entity_name: str) -> list[AttributeDef]:
        """Return only source=llm attributes (exclude align/structure/post)."""
        ed = self.entity_def(entity_name)
        return [a for a in ed.attributes if a.source == "llm"]

    def generate_json_schema(self, entity_names: list[str], strict: bool = True) -> dict:
        """Generate OpenAI response_format json_schema (LLM fields only).

        Produces:
          {"type": "object", "properties": {"extractions": {"type": "array",
           "items": {"anyOf": [variants]}}}}
        Each variant is:
          {"type": "object", "properties": {field: type}, "required": [...],
           "additionalProperties": false}
        """
        variants = []
        for ename in entity_names:
            ed = self.entity_def(ename)
            llm_attrs = self.llm_output_fields(ename)
            props: dict[str, Any] = {
                "entity_type": {"type": "string", "const": ename},
            }
            required: list[str] = []
            for a in llm_attrs:
                json_type = self._attr_json_type(a)
                props[a.name] = json_type
                if a.required:
                    required.append(a.name)
            variant: dict[str, Any] = {
                "type": "object",
                "properties": props,
                "additionalProperties": False,
            }
            if required:
                variant["required"] = required
            variants.append(variant)

        schema: dict[str, Any] = {
            "type": "object",
            "properties": {
                "extractions": {
                    "type": "array",
                    "items": {"anyOf": variants},
                }
            },
            "required": ["extractions"],
            "additionalProperties": False,
        }
        return schema

    def _attr_json_type(self, attr: AttributeDef) -> dict[str, Any]:
        """Map AttributeDef type to JSON Schema type."""
        type_map = {
            "string": "string",
            "integer": "integer",
            "float": "number",
            "boolean": "boolean",
            "text": "string",
            "enum": "string",
            "json_array": "array",
        }
        jt = type_map.get(attr.type, "string")
        result: dict[str, Any] = {"type": jt}
        if attr.type == "enum" and attr.enum_values:
            result["enum"] = attr.enum_values
        if attr.type == "json_array":
            result["items"] = {"type": "string"}
        return result

    def build_extraction_prompt(self, entity_names: list[str]) -> str:
        """Build extraction prompt from entity metadata.

        Uses XML-style structural markers (``<task>``, ``<steps>``,
        ``<entities>``, ``<format>``, ``<prohibitions>``) to help small
        models parse task boundaries correctly.
        """
        parts: list[str] = []

        # Build explicit field list for each entity
        field_list_parts = []
        for ename in entity_names:
            ed = self.entity_def(ename)
            llm_fields = self.llm_output_fields(ename)
            field_names = [f.name for f in llm_fields]
            field_list_parts.append(
                f'  {ename}: primary_key="{ed.primary_text}", '
                f'attributes=[{", ".join(field_names)}]'
            )
        field_list = "\n".join(field_list_parts)

        # --- Build dynamic generics ---
        # Reference rules: which entity types reference which others
        ref_rules: list[str] = []
        exclusivity_rules: list[str] = []
        for ename in entity_names:
            try:
                ed = self.entity_def(ename)
            except (KeyError, AttributeError):
                continue
            # Build reference rules from entity definition
            if ed.references:
                ref_names = ", ".join(
                    f"{r.name} -> {r.target_entity}" for r in ed.references
                )
                ref_rules.append(f"  {ename} entities reference: {ref_names}")
            # Build exclusivity rules from entity notes
            if ed.notes:
                import re as _re
                for line in ed.notes.splitlines():
                    m = _re.match(
                        r"^\s*mutually_exclusive_with\s*:\s*(.+)$",
                        line.strip(), _re.IGNORECASE,
                    )
                    if m:
                        ex_types = [t.strip() for t in m.group(1).split(",") if t.strip()]
                        ex_types.sort()
                        ex_text = " and ".join(ex_types)
                        exclusivity_rules.append(
                            f"- {ename} is mutually exclusive with: {ex_text}. "
                            f"A given item is EITHER {ex_text.split(' and ')[0]} OR "
                            f"{ex_text.split(' and ')[-1]}, NEVER both."
                        )

        # ==================================================================
        # STRUCTURED PROMPT with XML-style markers for small-model clarity
        # ==================================================================

        # --- <task> — what the model must do ---
        parts.append(
            "<task>\n"
            f"Extract structured entities from the scientific paper text below.\n"
            f"Entity types to extract: {', '.join(entity_names)}.\n"
            f"Output valid JSON with the EXACT entity type names and attribute "
            f"field names specified below. ALL keys MUST be in English.\n"
            "</task>"
        )

        # --- <steps> — numbered step-by-step instructions ---
        parts.append(
            "<steps>\n"
            "Step 1: Read the source text carefully from beginning to end.\n"
            "Step 2: Identify every item that matches one of the entity types "
            "defined below. Look at ALL sections of the text.\n"
            "Step 3: For each identified item, determine the correct entity "
            "type and fill in ALL required attributes.\n"
            "Step 4: The primary text value (extraction_text) MUST be a "
            "VERBATIM substring from the source text — at least one complete "
            "word, number, or phrase, long enough to uniquely locate it in "
            "the original document. Do NOT abbreviate, rephrase, normalize, "
            "or translate it.\n"
            "Step 5: Populate all reference fields with values that match "
            "existing entities of the target type.\n"
            "Step 6: Output the result as a JSON array under the key "
            '"extractions".\n'
            "</steps>"
        )

        # --- <format> — output format specification ---
        format_parts = [
            "<format>",
            "Output a JSON object with a single key \"extractions\" containing "
            "an array of entity objects.",
            "",
            "Each entity object has TWO keys:",
            f"  - The entity type name (one of: {', '.join(entity_names)}) as "
            "the key, with the primary text value as its value.",
            "  - \"<EntityType>_attributes\" as the key, with a dict of "
            "attribute field_name: value pairs.",
            "",
            "CORRECT example:",
            '  {"EntityType": "primary_value", "EntityType_attributes": {"field_name": "value"}}',
            "",
            "WRONG — do NOT use \"entity_type\" or \"entity_value\" as JSON keys:",
            '  {"entity_type": "EntityType", "entity_value": "value"}  // WRONG',
            "",
            "Required fields per entity type:",
            field_list,
            "</format>",
        ]
        parts.append("\n".join(format_parts))

        # --- <prohibitions> — explicit negative rules ---
        proh_parts = [
            "<prohibitions>",
            "DO NOT do any of the following:",
            "- Output \"none\", \"unspecified\", \"unknown\", \"Not reported\", "
            "\"N/A\", or \"na\" as a value. Use empty string \"\" instead.",
            "- Invent or guess names that do not appear verbatim in the text.",
            "- Concatenate multiple substance names into a single "
            "extraction_text (e.g., \"thymol_carvacrol\" is WRONG).",
            "- Translate field names to Chinese or any other language — ALL "
            "JSON keys MUST be in English.",
            "- Add parenthetical annotations to extraction_text. Put "
            "supplementary names in their dedicated attribute fields.",
            "- Create an entity without finding its exact mention in the text.",
            "- Use the keys \"entity_type\" or \"entity_value\" in the JSON "
            "output — use the actual entity type name as the key.",
        ]
        if exclusivity_rules:
            proh_parts.append("")
            proh_parts.append("EXCLUSIVITY RULES:")
            proh_parts.extend(exclusivity_rules)
        proh_parts.append("</prohibitions>")
        parts.append("\n".join(proh_parts))

        # --- <rules> — global extraction rules ---
        rules_parts = [
            "<rules>",
            "1. Every extraction_text MUST appear VERBATIM in the source text.",
            "2. Use the canonical full name as the primary value. Put "
            "abbreviations in the abbreviation field.",
            "3. Be case-consistent — use the same canonical name for the same "
            "entity throughout the document.",
            "4. Each entity type has a distinct purpose. Do NOT duplicate the "
            "same real-world item across multiple entity types.",
            "5. Use EXACT values and units as written in the source text. "
            "Do NOT perform unit conversions.",
            "6. When an entity has reference fields, populate them with "
            "values that EXACTLY match the target entity's primary text.",
        ]
        if ref_rules:
            rules_parts.append("")
            rules_parts.append("Reference relationships:")
            rules_parts.extend(ref_rules)
        rules_parts.append("</rules>")
        parts.append("\n".join(rules_parts))

        # --- <entities> — per-entity definitions ---
        entity_parts = ["<entities>"]
        for ename in entity_names:
            ed = self.entity_def(ename)
            ep = [f"## {ed.display_name} ({ename})"]
            if ed.description:
                ep.append(f"\n{ed.description}")
            if ed.extraction_guidance:
                ep.append(f"\n{ed.extraction_guidance}")
            if ed.notes:
                ep.append(f"\nNote: {ed.notes}")

            # Inject enum field constraints
            llm_fields = self.llm_output_fields(ename)
            enum_fields = [a for a in llm_fields if a.type == "enum" and a.enum_values]
            if enum_fields:
                enum_lines = ["\nAllowed values for enum fields:"]
                for a in enum_fields:
                    vals = ", ".join(a.enum_values)
                    enum_lines.append(f"  - {a.name}: [{vals}]")
                ep.append("\n".join(enum_lines))

            # Inject vocabulary summary (categories + examples)
            vocab = self.vocabulary(ename)
            if vocab and vocab.entries and (ed.vocabulary and ed.vocabulary.inject_in_prompt):
                ep.append(f"\n{vocab.format_for_prompt(summary_only=True)}")

            entity_parts.append("\n\n".join(ep))
        entity_parts.append("</entities>")
        parts.append("\n".join(entity_parts))

        # --- Source text marker ---
        parts.append(
            "<source_text>\n"
            "The text to extract from follows below.\n"
            "</source_text>"
        )

        return "\n\n".join(parts).strip()

    def post_process(self, extraction: Extraction) -> Extraction:
        """Apply post-processing pipeline for source=post fields.

        Reads the entity definition to find ``source: post`` attributes,
        then applies the configured pipeline (vocabulary matching,
        unit normalisation, etc.).  Fully generic — no entity-specific
        logic lives here.
        """
        attrs = dict(extraction.attributes) if extraction.attributes else {}

        try:
            ed = self.entity_def(extraction.extraction_class)
        except KeyError:
            extraction.attributes = attrs
            return extraction

        post_fields = [a for a in ed.attributes if a.source == "post"]
        if not post_fields:
            extraction.attributes = attrs
            return extraction

        # If entity has a vocabulary binding, run vocabulary matching
        vocab = self.vocabulary(extraction.extraction_class)
        if vocab and vocab.entries:
            primary_field = ed.primary_text
            lookup_value = attrs.get(primary_field, extraction.extraction_text)
            self._apply_vocabulary(attrs, vocab, lookup_value, ed)
            # Second-pass best-effort match for unmatched entries
            self._second_pass_match(attrs, vocab, lookup_value, ed)

        extraction.attributes = attrs
        return extraction

    def _apply_vocabulary(
        self,
        attrs: dict[str, Any],
        vocab: Vocabulary,
        lookup_value: str,
        ed: EntityDef,
    ) -> None:
        """Enhanced vocabulary matching with cascading match strategies.

        Uses the vocabulary binding's field mappings for the matched row,
        avoiding hardcoded entity-specific column names.
        """
        binding = ed.vocabulary
        if not binding:
            return

        # Determine class and subclass attribute names
        class_attr = "class_field"
        subclass_attr = None
        for a in ed.attributes:
            if a.source == "post" and a.type == "enum" and a.name != "match_source":
                class_attr = a.name
            elif a.source == "post" and a.type == "string" and a.name != "match_source":
                subclass_attr = a.name

        match = self._cascading_match(lookup_value, vocab, binding)

        if match:
            row_class = match.get(binding.class_field or "", "")
            row_subclass = match.get(binding.subclass_field or "", "")
            attrs[class_attr] = row_class or "Other"
            if row_subclass and subclass_attr:
                attrs[subclass_attr] = row_subclass

            # match_source: distinguish exact vs fuzzy
            is_exact = vocab.lookup(lookup_value) is not None
            if not is_exact:
                cleaned = Vocabulary._clean_name(lookup_value)
                is_exact = vocab.lookup(cleaned) is not None
            attrs.setdefault("match_source", "exact" if is_exact else "fuzzy")
        else:
            attrs.setdefault(class_attr, "Other")
            attrs.setdefault("match_source", "unmatched")

    def _cascading_match(
        self,
        lookup_value: str,
        vocab: Vocabulary,
        binding: VocabularyBinding,
    ) -> dict | None:
        """Cascading match strategies from exact to fuzzy.

        1. Exact match (case-insensitive)
        2. Cleaned name (strips parentheticals)
        3. Substring match
        4. Normalized match (strip punctuation, lowercase)
        5. Levenshtein distance <= 3
        6. Comma-separated split and individual match
        """
        name = lookup_value.strip()
        if not name:
            return None

        # 1. Exact match (case-insensitive)
        match = vocab.lookup(name)
        if match:
            return match

        # 2. Cleaned name (strip parentheticals)
        cleaned = Vocabulary._clean_name(name)
        if cleaned.lower() != name.lower():
            match = vocab.lookup(cleaned)
            if match:
                return match

        # 3. Substring match (vocab entry contains extraction name or vice versa)
        norm = _normalize(name)
        if norm:
            for key, entry in vocab.by_name.items():
                key_norm = _normalize(key)
                if norm in key_norm or key_norm in norm:
                    return entry

        # 4. Normalized match (strip all punctuation and spaces, lowercase, compare)
        agg_norm = re.sub(r'[^a-z0-9]', '', name.lower())
        if agg_norm:
            for key, entry in vocab.by_name.items():
                key_agg = re.sub(r'[^a-z0-9]', '', key.lower())
                if agg_norm == key_agg:
                    return entry

        # 5. Levenshtein distance <= 3
        best = None
        best_dist = 1000
        for key, entry in vocab.by_name.items():
            key_norm = _normalize(key)
            dist = _levenshtein_distance(norm, key_norm)
            if dist < best_dist and dist <= 3:
                best_dist = dist
                best = entry
        if best:
            return best

        # 6. Comma-separated lookup: try each part individually
        if ',' in name:
            for part in name.split(','):
                part = part.strip()
                if part:
                    match = self._cascading_match(part, vocab, binding)
                    if match:
                        return match

        return None

    def _second_pass_match(
        self,
        attrs: dict[str, Any],
        vocab: Vocabulary,
        lookup_value: str,
        ed: EntityDef,
    ) -> None:
        """Second-pass best-effort match.

        Uses aggressive normalization (remove spaces and punctuation)
        to find partial matches in the vocabulary.
        """
        binding = ed.vocabulary
        if not binding:
            return

        # Determine class attribute name
        class_attr = None
        for a in ed.attributes:
            if a.source == "post" and a.type == "enum" and a.name != "match_source":
                class_attr = a.name
                break

        if not class_attr:
            return

        # Only run if currently unmatched
        current = attrs.get(class_attr, "")
        if current != "Other":
            return

        name = lookup_value.strip()
        if not name:
            return

        # Aggressive normalization: remove spaces and punctuation
        agg_norm = re.sub(r'[^a-z0-9]', '', name.lower())
        if not agg_norm:
            return

        for key, entry in vocab.by_name.items():
            key_agg = re.sub(r'[^a-z0-9]', '', key.lower())
            if not key_agg:
                continue
            # Check if either contains the other (partial match)
            if agg_norm in key_agg or key_agg in agg_norm:
                row_class = entry.get(binding.class_field or "", "")
                if row_class and row_class != "Other":
                    attrs[class_attr] = row_class
                    attrs["match_source"] = "fuzzy"
                    # Also set subclass if available
                    for a in ed.attributes:
                        if a.source == "post" and a.type == "string" and a.name != "match_source" and a.name != class_attr:
                            row_subclass = entry.get(binding.subclass_field or "", "")
                            if row_subclass:
                                attrs[a.name] = row_subclass
                            break
                    return

    # ------------------------------------------------------------------
    # Gate evaluation
    # ------------------------------------------------------------------

    @staticmethod
    def _parse_gate_condition(condition: str) -> tuple[str, str, list[str]]:
        """Parse a gate condition string into (field_name, operator, excluded_values).

        Supports forms like:
          - "classification not in ['Other', '', 'unmatched']"
          - "classification in ['ValidClass']"
          - "not_empty"  (any non-empty value passes)

        Returns (field, 'not_in'|'in'|'not_empty', [values]).
        """
        field = ""
        operator = ""
        values: list[str] = []
        stripped = condition.strip()

        if not stripped:
            return ("", "any", [])

        if stripped == "not_empty":
            return ("", "not_empty", [])

        # Parse "field not in ['val1', 'val2']" or "field in ['val1']"
        import ast
        m = re.match(r"(\w+)\s+(not\s+in|in)\s+(\[.+\])", stripped)
        if m:
            field = m.group(1)
            operator = m.group(2).replace(" ", "_")  # "not in" → "not_in"
            try:
                values = ast.literal_eval(m.group(3))
            except (ValueError, SyntaxError):
                values = []
        else:
            # Try simpler form: "field not empty" or "field exists"
            m2 = re.match(r"(\w+)\s+(not\s+empty|exists|is\s+set)", stripped)
            if m2:
                field = m2.group(1)
                operator = "not_empty"

        return (field, operator, values)

    def evaluate_gate(
        self,
        extractions: list[Extraction],
        gate_entity: str,
        gate_condition: str,
    ) -> bool:
        """Evaluate a gate condition against extractions.

        Parses *gate_condition* to determine the pass criteria, then checks
        whether any extraction of *gate_entity* satisfies it.

        Returns True if the gate passes (article should continue).

        Condition syntax:
          - ``"<field> not in [<values>]"`` — pass if any entity has the field
            with a value NOT in the exclusion list
          - ``"<field> in [<values>]"`` — pass if any entity has the field
            with a value IN the inclusion list
          - ``"not_empty"`` — pass if there is any extraction of gate_entity at
            all
          - empty string → always pass

        When the field name cannot be resolved to a known attribute, the method
        falls back to checking *all* ``source=post`` enum attributes — the same
        logic used when no condition is provided.
        """
        if not gate_entity or not gate_condition:
            return True  # no gate → pass

        field, operator, values = self._parse_gate_condition(gate_condition)

        if operator == "any":
            return True  # empty/unparseable condition → pass

        # Resolve field to known attribute(s)
        check_attrs: list = []
        if field:
            try:
                ed = self.entity_def(gate_entity)
                for a in ed.attributes:
                    if a.name == field:
                        check_attrs.append(a)
                        break
            except KeyError:
                pass

        # Fallback: if field not found in entity attributes, check all post/enum fields
        if not check_attrs:
            try:
                ed = self.entity_def(gate_entity)
                for a in ed.attributes:
                    if a.source == "post" and a.type == "enum":
                        check_attrs.append(a)
            except KeyError:
                pass

        # Build the set of values to check against
        exclude_set_lower: set[str] = set()
        include_set_lower: set[str] = set()
        if operator == "not_in":
            exclude_set_lower = {v.lower().strip() for v in values if isinstance(v, str)}
        elif operator == "in":
            include_set_lower = {v.lower().strip() for v in values if isinstance(v, str)}

        for ext in extractions:
            if ext.extraction_class != gate_entity:
                continue
            attrs = ext.attributes or {}

            if operator == "not_empty":
                # Any extraction of this entity type → pass
                return True

            for a in check_attrs:
                val = attrs.get(a.name, "")
                if isinstance(val, str):
                    val_lower = val.lower().strip()
                else:
                    val_lower = ""

                if operator == "not_in":
                    if val_lower and val_lower not in exclude_set_lower:
                        return True
                elif operator == "in":
                    if val_lower in include_set_lower:
                        return True

        return False

    def validate_extraction(self, extraction: Extraction) -> list[str]:
        """Validate required fields. Returns list of error messages."""
        errors: list[str] = []
        try:
            ed = self.entity_def(extraction.extraction_class)
        except KeyError:
            errors.append(f"Unknown extraction class: {extraction.extraction_class}")
            return errors

        attrs = extraction.attributes or {}
        for a in ed.attributes:
            if a.required and a.source == "llm":
                val = attrs.get(a.name)
                if val is None or (isinstance(val, str) and val.strip() == ""):
                    errors.append(
                        f"{extraction.extraction_class}: missing required field '{a.name}'"
                    )
        return errors
