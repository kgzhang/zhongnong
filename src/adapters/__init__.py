"""Format-specific adapters — NOT part of the generic framework.

Each adapter bridges a specific input format (PMC XML, PDF, etc.) to the
generic extraction pipeline:

1. Parse the format → extract plain text + metadata
2. Build a Document object for the extraction pipeline
3. Optionally provide a pre_extractor for metadata-derived entities

Adapters live here to keep the ``src/`` framework free of format-specific logic.
"""
