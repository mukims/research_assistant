"""embeddinggemma-2 through Ollama: what is sent, and the guard against a dropped image."""

import os
import tempfile
import types
import unittest
from unittest.mock import patch

import research_assistant.shared.llm as llm
from research_assistant.shared import db


class FakeResponse:
    def __init__(self, payload):
        self._payload = payload

    def raise_for_status(self):
        pass

    def json(self):
        return self._payload


class GemmaEmbeddingTests(unittest.TestCase):
    def _embedder(self, payload):
        sent = []

        def fake_post(url, json=None, timeout=None):
            sent.append((url, json))
            return FakeResponse(payload)

        patcher = patch("requests.post", fake_post)
        patcher.start()
        self.addCleanup(patcher.stop)
        return llm._GemmaEmbeddings("embeddinggemma-2"), sent

    def test_documents_take_the_document_prefix(self):
        emb, sent = self._embedder({"embeddings": [[0.1], [0.2]]})
        self.assertEqual(emb.embed_documents(["a", "b"]), [[0.1], [0.2]])
        self.assertEqual(sent[0][1]["input"], ["title: none | text: a", "title: none | text: b"])
        self.assertTrue(sent[0][0].endswith("/api/embed"))

    def test_query_takes_the_search_prefix(self):
        emb, sent = self._embedder({"embeddings": [[0.5]]})
        self.assertEqual(emb.embed_query("spin qubits"), [0.5])
        self.assertEqual(sent[0][1]["input"], ["task: search result | query: spin qubits"])

    def test_no_documents_means_no_request(self):
        emb, sent = self._embedder({"embeddings": []})
        self.assertEqual(emb.embed_documents([]), [])
        self.assertEqual(sent, [])

    def test_image_is_sent_in_the_dict_form_with_no_prefix(self):
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as fh:
            fh.write(b"\x89PNGfake")
        self.addCleanup(os.unlink, fh.name)
        emb, sent = self._embedder({"embeddings": [[0.3, 0.4]], "prompt_eval_count": 261})
        self.assertEqual(emb.embed_image(fh.name), [0.3, 0.4])
        (item,) = sent[0][1]["input"]
        self.assertEqual(set(item), {"text", "image"})
        self.assertEqual(item["text"], "")
        self.assertNotIn("images", sent[0][1])

    def test_an_image_ollama_dropped_is_an_error_not_a_vector(self):
        """Ollama answers 200 with a text-only vector when the image is ignored."""
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as fh:
            fh.write(b"\x89PNGfake")
        self.addCleanup(os.unlink, fh.name)
        emb, _ = self._embedder({"embeddings": [[0.3]], "prompt_eval_count": 3})
        with self.assertRaisesRegex(RuntimeError, "did not reach the model"):
            emb.embed_image(fh.name)

    def test_ollama_host_gets_a_scheme(self):
        with patch.dict(os.environ, {"OLLAMA_HOST": "10.0.0.5:11434"}):
            self.assertEqual(llm._ollama_host(), "http://10.0.0.5:11434")
        with patch.dict(os.environ, {"OLLAMA_HOST": "http://box:1/"}):
            self.assertEqual(llm._ollama_host(), "http://box:1")


class EmbedderIdentityTests(unittest.TestCase):
    def _collection(self, metadata):
        return types.SimpleNamespace(metadata=metadata, name="c")

    def test_matching_embedder_passes(self):
        with patch.object(db, "EMBED_MODEL", "embeddinggemma-2"):
            db.check_embedder(self._collection({"embedder": "embeddinggemma-2"}))

    def test_other_embedder_is_refused_with_both_names(self):
        with patch.object(db, "EMBED_MODEL", "nomic-embed-text"):
            with self.assertRaisesRegex(RuntimeError, "embeddinggemma-2.*nomic-embed-text"):
                db.check_embedder(self._collection({"embedder": "embeddinggemma-2"}))

    def test_an_index_that_predates_the_record_is_not_checked(self):
        db.check_embedder(self._collection({"hnsw:space": "cosine"}))
        db.check_embedder(self._collection(None))


if __name__ == "__main__":
    unittest.main()
