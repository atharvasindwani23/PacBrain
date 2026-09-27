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
from memory.service import gbrain_client, ingest, recall_procedure, validate_recalled_procedure
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
    def test_equal_reward_ties_preserve_provider_rank_but_better_reward_wins(self):
        first = trace()
        second = dict(trace(), episode_id="another-episode")
        tied = sorted([extract_procedure(first), extract_procedure(second)],
                      key=lambda p: p["id"], reverse=True)
        better_trace = copy.deepcopy(first)
        better_trace["episode_id"] = "better-episode"
        for frame in better_trace["frames"]:
            frame["score"] *= 2
        better = extract_procedure(better_trace)

        class RankedBrain:
            def __init__(self, procedures):
                self.pages = {p["id"]: p for p in procedures}
            def search(self, query, limit=20):
                return [{"slug": pid} for pid in self.pages]
            def get_page(self, slug):
                return {"content": "```json\n" + json.dumps(self.pages[slug]) + "\n```"}

        for pages, expected in ((tied, tied[0]), (tied + [better], better)):
            with self.subTest(better_reward=len(pages) == 3), tempfile.TemporaryDirectory() as folder:
                result = recall_procedure("testCorridor", store=MemoryStore(folder), gbrain=RankedBrain(pages))
                self.assertEqual(result["procedure"]["id"], expected["id"])

    def test_duplicate_versions_keep_first_ranked_validated_receipt(self):
        original = extract_procedure(trace())
        preferred = copy.deepcopy(original)
        preferred["extraction"] = "memorable"
        preferred["providers"] = {"memorable": {"request_id": "first-ranked", "draft": {
            "title": "Opening", "steps": [{"seq": 1, "action": "pacman.step",
                                           "command": "pacman.step --action E", "repeat_count": 2}]}}}
        later = copy.deepcopy(original)
        later["extraction"] = "memorable"
        later["providers"] = copy.deepcopy(preferred["providers"])
        later["providers"]["memorable"]["request_id"] = "lower-ranked"
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

        for case, versions in (("ranked_receipts", [preferred, later]),
                               ("invalid_first", [invalid, preferred, later]),
                               ("local_first", [original, preferred, later])):
            with self.subTest(case=case), tempfile.TemporaryDirectory() as folder:
                store = MemoryStore(folder)
                result = recall_procedure("testCorridor", store=store, gbrain=VersionedBrain(versions))
                self.assertEqual(result["procedure"]["providers"]["memorable"]["request_id"], "first-ranked")
                self.assertEqual(store.procedures(), [preferred])

    def test_recalled_memorable_receipt_must_match_actions(self):
        procedure = extract_procedure(trace())
        procedure["extraction"] = "memorable"
        procedure["providers"]["memorable"] = {"request_id": "wrong-draft", "draft": {
            "title": "Different opening", "steps": [{"seq": 1, "action": "pacman.step",
                "command": "pacman.step --action W", "repeat_count": 2}]}}
        with self.assertRaisesRegex(ValueError, "does not match"):
            validate_recalled_procedure(procedure)
        class CorruptedBrain:
            def search(self, query, limit=20):
                return [{"slug": "pacman-memory/" + procedure["id"] + "-v-test"}]
            def get_page(self, slug):
                return {"content": "```json\n" + json.dumps(procedure) + "\n```"}
        with tempfile.TemporaryDirectory() as folder:
            store = MemoryStore(folder)
            hit = recall_procedure("testCorridor", store=store, gbrain=CorruptedBrain())
            self.assertIsNone(hit["procedure"])
            self.assertEqual(store.procedures(), [])
            # Cached files must not bypass the same check in explicit local mode.
            store.save(procedure)
            self.assertIsNone(recall_procedure("testCorridor", store=store, provider="local")["procedure"])

    def test_ingest_rejects_missing_receipt_id_before_persistence(self):
        class MissingReceiptId(MemorableStub):
            def extract_trace(self, wire):
                return dict(super().extract_trace(wire), request_id=None)
        with tempfile.TemporaryDirectory() as folder:
            store = MemoryStore(folder)
            with self.assertRaisesRegex(ValueError, "missing its request_id"):
                ingest(trace(), store, memorable=MissingReceiptId())
            self.assertEqual(store.procedures(), [])

    def test_stored_but_not_searchable_memory_is_reported_separately(self):
        class DelayedIndexBrain(BrainStub):
            def search(self, query, limit=20):
                return []
        with tempfile.TemporaryDirectory() as folder:
            result = ingest(trace(), MemoryStore(folder), extractor="local", gbrain=DelayedIndexBrain())
            self.assertTrue(result["gbrain_stored"])
            self.assertFalse(result["gbrain_search_verified"])
            self.assertFalse(result["gbrain_indexed"])

    def test_recalled_provenance_distinguishes_missing_and_malformed_receipts(self):
        procedure = extract_procedure(trace())
        self.assertIs(validate_recalled_procedure(procedure), procedure)
        procedure["extraction"] = "unknown"
        with self.assertRaisesRegex(ValueError, "supported extraction method"):
            validate_recalled_procedure(procedure)
        procedure["extraction"] = "memorable"
        with self.assertRaisesRegex(ValueError, "missing its provider receipt"):
            validate_recalled_procedure(procedure)
        procedure["providers"]["memorable"] = None
        with self.assertRaisesRegex(ValueError, "malformed or refused"):
            validate_recalled_procedure(procedure)
        receipt = MemorableStub().extract_trace({})
        procedure["providers"]["memorable"] = dict(receipt, refused="allowance_exhausted")
        with self.assertRaisesRegex(ValueError, "malformed or refused"):
            validate_recalled_procedure(procedure)
        procedure["providers"]["memorable"] = dict(receipt, request_id=None)
        with self.assertRaisesRegex(ValueError, "missing its request_id"):
            validate_recalled_procedure(procedure)
        procedure["providers"]["memorable"] = receipt
        receipt["draft"]["session_id"] = "another-episode"
        with self.assertRaisesRegex(ValueError, "another source episode"):
            validate_recalled_procedure(procedure)

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
            self.assertTrue(result["gbrain_stored"])
            self.assertTrue(result["gbrain_search_verified"])
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
