"""LLM response cache — avoids redundant API calls.

Cache key = SHA256(model_id + system_prompt + user_prompt)[:16].
Stored as JSON files in ``data/cache/``.
"""
from __future__ import annotations

import hashlib
import json
import logging
from pathlib import Path

logger = logging.getLogger(__name__)


class LLMCache:
    """Simple file-based cache for LLM responses."""

    def __init__(self, cache_dir: str | Path = "data/cache"):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def _cache_path(self, cache_key: str) -> Path:
        return self.cache_dir / f"{cache_key}.json"

    def get(self, model_id: str, system_prompt: str, user_prompt: str) -> str | None:
        """Return cached response or None."""
        key = self._make_key(model_id, system_prompt, user_prompt)
        path = self._cache_path(key)
        if path.exists():
            try:
                data = json.loads(path.read_text())
                logger.debug("Cache hit: %s", key)
                return data["response"]
            except (json.JSONDecodeError, KeyError):
                path.unlink(missing_ok=True)
        return None

    def set(self, model_id: str, system_prompt: str, user_prompt: str,
            response: str) -> None:
        """Store response in cache."""
        key = self._make_key(model_id, system_prompt, user_prompt)
        path = self._cache_path(key)
        path.write_text(json.dumps({
            "key": key,
            "model_id": model_id,
            "response": response,
        }, ensure_ascii=False))

    @staticmethod
    def _make_key(model_id: str, system_prompt: str, user_prompt: str) -> str:
        raw = f"{model_id}|{system_prompt}|{user_prompt}"
        return hashlib.sha256(raw.encode()).hexdigest()[:16]

    def clear(self) -> None:
        """Remove all cached entries."""
        for path in self.cache_dir.glob("*.json"):
            path.unlink()
        logger.info("Cache cleared (%s)", self.cache_dir)
