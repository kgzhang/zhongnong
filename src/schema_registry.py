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

    def format_for_prompt(self) -> str:
        """Render vocabulary as prompt text."""
        if not self.entries:
            return ""
        lines = ["## Alternative分类（词表）"]
        for entry in self.entries:
            name = entry.get(self.binding.key_field, "")
            cls = entry.get(self.binding.class_field or "", "")
            subclass = entry.get(self.binding.subclass_field or "", "")
            parts = [name]
            if cls:
                parts.append(f"  [{cls}]")
            if subclass:
                parts.append(f"  ({subclass})")
            lines.append("".join(parts))
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
            )
        return EntityDef(
            name=name,
            display_name=raw.get("display_name", name),
            description=raw.get("description", ""),
            primary_text=raw.get("primary_text", ""),
            extraction_guidance=raw.get("extraction_guidance", ""),
            examples=raw.get("examples", []),
            notes=raw.get("notes", ""),
            attributes=attrs,
            references=refs,
            inline_relations=inlines,
            vocabulary=vocab,
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
        """Build extraction prompt from entity metadata."""
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

        # Format instruction — MUST use English field names for JSON keys
        parts.append(
            f'CRITICAL: Output valid JSON with these EXACT English entity types and attribute names.\n'
            f'Entity types to extract: {", ".join(entity_names)}\n'
            f'Required fields per entity:\n{field_list}\n'
            f'JSON format: {{"extractions": [{{"EntityType": "primary_value", "EntityType_attributes": {{"field_name": value, ...}}}}, ...]}}\n'
            f'ALL attribute keys MUST be in English as specified above. DO NOT translate field names to Chinese.'
        )

        for ename in entity_names:
            ed = self.entity_def(ename)
            parts.append(f"## {ed.display_name} ({ename})")
            if ed.description:
                parts.append(f"\n{ed.description}")
            if ed.extraction_guidance:
                parts.append(f"\n{ed.extraction_guidance}")
            if ed.notes:
                parts.append(f"\n注意: {ed.notes}")
            # Inject vocabulary if available
            vocab = self.vocabulary(ename)
            if vocab and vocab.entries and (ed.vocabulary and ed.vocabulary.inject_in_prompt):
                parts.append(f"\n{vocab.format_for_prompt()}")
        return "\n\n".join(parts).strip()

    def post_process(self, extraction: Extraction) -> Extraction:
        """Apply vocabulary matching to populate source=post fields.

        Currently handles: Alternative entities (alternative_class, subclass, match_source).
        """
        attrs = dict(extraction.attributes) if extraction.attributes else {}

        if extraction.extraction_class == "Alternative":
            self._post_process_alternative(attrs)

        extraction.attributes = attrs
        return extraction

    def _post_process_alternative(self, attrs: dict[str, Any]) -> None:
        """Populate alternative_class, subclass, match_source from vocabulary."""
        standard_name = attrs.get("standard_name", "")
        vocab = self.vocabulary("Alternative")
        if not vocab or not vocab.entries:
            # No vocabulary loaded — set defaults
            attrs.setdefault("alternative_class", "Other")
            attrs.setdefault("match_source", "Other_未匹配")
            return

        # Try exact match first
        match = vocab.lookup(standard_name)
        if match:
            attrs["alternative_class"] = match.get("Alternative_Class", "Other")
            attrs["subclass"] = match.get("Subclass", "")
            attrs["match_source"] = "词表精确匹配"
            return

        # Try fuzzy match
        match = vocab.fuzzy_match(standard_name)
        if match:
            attrs["alternative_class"] = match.get("Alternative_Class", "Other")
            attrs["subclass"] = match.get("Subclass", "")
            attrs["match_source"] = "词表模糊匹配"
            return

        # No match
        attrs["alternative_class"] = "Other"
        attrs["match_source"] = "Other_未匹配"

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
