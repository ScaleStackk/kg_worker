import hashlib
import tempfile
import unittest
from pathlib import Path

from document_identity import resolve_document_identity


class DocumentIdentityTests(unittest.TestCase):
    def test_api_id_is_canonical_and_hash_is_retained(self) -> None:
        content = b"synthetic judgment"
        with tempfile.TemporaryDirectory() as directory:
            pdf = Path(directory) / "judgment.pdf"
            pdf.write_bytes(content)
            document_id, content_hash = resolve_document_identity(pdf, "api-uuid-1")

        self.assertEqual(document_id, "api-uuid-1")
        self.assertEqual(content_hash, hashlib.sha256(content).hexdigest())

    def test_hash_remains_fallback_for_manual_runs(self) -> None:
        content = b"manual judgment"
        with tempfile.TemporaryDirectory() as directory:
            pdf = Path(directory) / "judgment.pdf"
            pdf.write_bytes(content)
            document_id, content_hash = resolve_document_identity(pdf, None)

        self.assertEqual(document_id, content_hash)
