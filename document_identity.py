"""Canonical document identity plus a content hash for integrity checks."""

from __future__ import annotations

import hashlib
from pathlib import Path


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def resolve_document_identity(path: Path, requested_id: str | None) -> tuple[str, str]:
    content_hash = file_sha256(path)
    canonical_id = (requested_id or "").strip()
    if len(canonical_id) > 200 or any(ord(character) < 32 for character in canonical_id):
        raise ValueError("document ID is invalid")
    return canonical_id or content_hash, content_hash
