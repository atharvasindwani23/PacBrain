import copy
import io
import json
import os
import tempfile
import unittest
from contextlib import redirect_stdout
from unittest.mock import patch

from memory.__main__ import main
from memory.core import MemoryStore, extract_procedure
from memory.gbrain import GBrainError
from memory.memorable import MemorableError
from memory.service import gbrain_client, ingest, recall_procedure
from tests.test_memory import trace


class MemorableStub:
    def extract_trace(self, wire):
        self.wire = wire
        return {"request_id": "test-only", "draft": {"title": "Opening", "steps": [
            {"seq": 1, "action": "pacman.step", "command": "pacman.step --action E", "repeat_count": 2}]}}


class BrainStub:
    def __init__(self):
        self.rows = []

    def import_directory(self, directory):
        self.directory = directory
        self.rows = [{"slug": path.stem} for path in directory.glob("*.md")]
        return {"imported": len(self.rows)}

    def search(self, query, limit=20):
        return self.rows

    def get_page(self, slug):
        return {"content": (self.directory / (slug + ".md")).read_text()}


class ServiceTests(unittest.TestCase):
    def test_duplicate_versions_keep_first_ranked_validated_receipt(self):
        original = extract_procedure(trace())
        preferred = copy.deepcopy(original)
        preferred["providers"] = {"memorable": {"request_id": "first-ranked"}}
        later = copy.deepcopy(original)
        later["providers"] = {"memorable": {"request_id": "lower-ranked"}}
        invalid = copy.deepcopy(original)
        invalid["steps"] = []

        class VersionedBrain:
            def __init__(self, versions):
                self.pages = {"pacman-memory/%s-v-%s" % (original["id"], index): version
                              for index, version in enumerate(versions)}

            def search(self, query, limit=20):
                return [{"slug": slug} for slug in self.pages]

            def get_page(self, slug):
                return {"content": "```json\n" + json.dumps(self.pages[slug]) + "\n```"}

        for versions in ([preferred, later], [invalid, preferred, later]):
            with self.subTest(invalid_first=len(versions) == 3), tempfile.TemporaryDirectory() as folder:
                store = MemoryStore(folder)
                result = recall_procedure("testCorridor", store=store, gbrain=VersionedBrain(versions))
                self.assertEqual(result["procedure"]["providers"]["memorable"]["request_id"], "first-ranked")
                self.assertEqual(store.procedures(), [preferred])

    def test_hosted_configuration_selects_http(self):
        values = {"GBRAIN_MCP_URL": "https://example.com/mcp", "GBRAIN_ACCESS_TOKEN": "test-token"}
        with patch.dict(os.environ, values, clear=True), patch("memory.service.GBrainHTTPClient") as hosted, patch("memory.service.GBrainClient") as local:
            self.assertIs(gbrain_client(), hosted.return_value)
            hosted.assert_called_once_with(values["GBRAIN_MCP_URL"], values["GBRAIN_ACCESS_TOKEN"])
            local.assert_not_called()

    def test_explicit_local_overrides_hosted_configuration(self):
        values = {"GBRAIN_TRANSPORT": "local", "GBRAIN_MCP_URL": "https://example.com/mcp",
                  "GBRAIN_ACCESS_TOKEN": "test-token"}
        with patch.dict(os.environ, values, clear=True), patch("memory.service.GBrainHTTPClient") as hosted, patch("memory.service.GBrainClient") as local:
            self.assertIs(gbrain_client(), local.return_value)
            hosted.assert_not_called()

    def test_no_hosted_values_preserves_local_default(self):
        with patch.dict(os.environ, {}, clear=True), patch("memory.service.GBrainClient") as local:
            self.assertIs(gbrain_client(), local.return_value)

    def test_partial_hosted_configuration_never_falls_back(self):
        for values in ({"GBRAIN_MCP_URL": "https://example.com/mcp"},
                       {"GBRAIN_ACCESS_TOKEN": "test-token"}, {"GBRAIN_TRANSPORT": "http"}):
            with self.subTest(values=list(values)), patch.dict(os.environ, values, clear=True), patch("memory.service.GBrainClient") as local:
                with self.assertRaisesRegex(GBrainError, "requires both"):
                    gbrain_client()
                local.assert_not_called()

    def test_unknown_transport_is_rejected(self):
        with patch.dict(os.environ, {"GBRAIN_TRANSPORT": "typo"}, clear=True):
            with self.assertRaisesRegex(GBrainError, "must be auto, local, or http"):
                gbrain_client()

    def test_doctor_handles_hosted_without_disclosing_credentials(self):
        values = {"GBRAIN_MCP_URL": "https://example.com/mcp", "GBRAIN_ACCESS_TOKEN": "test-token"}
        with tempfile.TemporaryDirectory() as folder, patch.dict(os.environ, values, clear=True), redirect_stdout(io.StringIO()) as output:
            code = main(["--env-file", folder + "/absent.env", "--memory-dir", folder, "doctor"])
        result = json.loads(output.getvalue())
        self.assertEqual(code, 0)
        self.assertEqual(result["gbrain_transport"], "http_mcp")
        self.assertTrue(result["gbrain_configured"])
        self.assertNotIn("gbrain_home", result)
        self.assertNotIn("test-token", output.getvalue())
        self.assertNotIn("example.com", output.getvalue())

    def test_fresh_harness_recovers_memory_from_actual_brain_page(self):
        with tempfile.TemporaryDirectory() as source, tempfile.TemporaryDirectory() as target:
            brain = BrainStub()
            result = ingest(trace(), MemoryStore(source), extractor="local", gbrain=brain)
            recovered = recall_procedure("testCorridor", store=MemoryStore(target), gbrain=brain)
            self.assertEqual(recovered["procedure"]["id"], result["procedure_id"])
            self.assertEqual(len(MemoryStore(target).procedures()), 1)
    def test_real_draft_provenance_and_gbrain_result_gated_recall(self):
        with tempfile.TemporaryDirectory() as folder:
            store = MemoryStore(folder)
            memorable, brain = MemorableStub(), BrainStub()
            result = ingest(trace(), store, memorable=memorable, gbrain=brain)
            self.assertTrue(result["gbrain_indexed"])
            self.assertEqual(len(memorable.wire["tool_calls"]), 2)
            recalled = recall_procedure("testCorridor", store=store, gbrain=brain)
            self.assertEqual(recalled["procedure"]["extraction"], "memorable")
            self.assertEqual(recalled["procedure"]["providers"]["memorable"]["request_id"], "test-only")
            brain.rows = []
            self.assertIsNone(recall_procedure("testCorridor", store=store, gbrain=brain)["procedure"])

    def test_missing_key_does_not_fall_back(self):
        class RefusingClient:
            def extract_trace(self, wire):
                raise MemorableError("No credential")
        with tempfile.TemporaryDirectory() as folder:
            store = MemoryStore(folder)
            with self.assertRaises(MemorableError):
                ingest(trace(), store, memorable=RefusingClient())
            self.assertEqual(store.procedures(), [])

    def test_changed_memorable_sequence_is_rejected(self):
        class WrongDraft:
            def extract_trace(self, wire):
                return {"draft": {"title": "Wrong", "steps": [
                    {"seq": 1, "action": "step", "command": "pacman.step --action N"}]}}
        with tempfile.TemporaryDirectory() as folder:
            with self.assertRaises(MemorableError):
                ingest(trace(), MemoryStore(folder), memorable=WrongDraft())
