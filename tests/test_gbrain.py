import json
import os
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import patch

from memory.gbrain import GBrainClient, GBrainError


class GBrainClientTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.home = Path(self.tmp.name).resolve()
        (self.home / ".gbrain").mkdir()
        (self.home / ".gbrain/config.json").write_text("{}")
        self.client = GBrainClient(self.home, timeout=3)
        self.which = patch("memory.gbrain.shutil.which", return_value="/tools/gbrain")
        self.which.start()

    def tearDown(self):
        self.which.stop()
        self.tmp.cleanup()

    @staticmethod
    def response(payload, code=0, stderr=""):
        return subprocess.CompletedProcess([], code, json.dumps(payload), stderr)

    def test_search_preserves_real_rows_and_isolates_environment(self):
        rows = [{"slug": "procedures/escape", "chunk_text": "Escape north", "score": 0.25}]
        with patch.dict(os.environ, {"DATABASE_URL": "secret-db", "GBRAIN_SOURCE": "personal"}), patch("memory.gbrain.subprocess.run", return_value=self.response(rows)) as run:
            self.assertEqual(self.client.search("escape corridor"), rows)
        args, kwargs = run.call_args
        self.assertEqual(args[0], ["gbrain", "search", "escape corridor", "--limit", "5", "--json"])
        self.assertEqual(kwargs["env"]["GBRAIN_HOME"], str(self.home))
        self.assertNotIn("DATABASE_URL", kwargs["env"])
        self.assertNotIn("GBRAIN_SOURCE", kwargs["env"])
        self.assertEqual(kwargs["cwd"], str(self.home))

    def test_import_includes_ignored_memory_and_refreshes_changed_pages(self):
        with patch("memory.gbrain.subprocess.run", return_value=self.response({"imported": 2, "errors": 0})) as run:
            self.assertEqual(self.client.import_directory(self.home)["imported"], 2)
        command = run.call_args[0][0]
        self.assertIn("--no-embed", command)
        self.assertIn("--include-gitignored", command)
        self.assertIn("--fresh", command)

    def test_get_page_returns_canonical_markdown_for_another_runner(self):
        page = {"slug": "procedures/escape", "content": "# Escape\n```json\n{}\n```"}
        with patch("memory.gbrain.subprocess.run", return_value=self.response(page)) as run:
            self.assertEqual(self.client.get_page("procedures/escape"), page)
        self.assertEqual(run.call_args[0][0], ["gbrain", "get", "procedures/escape", "--json", "--include-content"])

    def test_cli_failure_does_not_expose_output(self):
        with patch("memory.gbrain.subprocess.run", return_value=self.response({}, 1, "API_KEY=super-secret live_serve")):
            with self.assertRaises(GBrainError) as error:
                self.client.search("escape")
        self.assertNotIn("super-secret", str(error.exception))
        self.assertIn("lock", str(error.exception))

    def test_timeout_is_actionable(self):
        with patch("memory.gbrain.subprocess.run", side_effect=subprocess.TimeoutExpired("gbrain", 3)):
            with self.assertRaisesRegex(GBrainError, "timed out"):
                self.client.search("escape")

    def test_invalid_json_and_unexpected_shapes_are_errors(self):
        for output in ("progress then {}", "{}", "[1]"):
            with self.subTest(output=output), patch("memory.gbrain.subprocess.run", return_value=subprocess.CompletedProcess([], 0, output, "")):
                with self.assertRaises(GBrainError):
                    self.client.search("escape")

    def test_partial_import_is_not_reported_as_success(self):
        with patch("memory.gbrain.subprocess.run", return_value=self.response({"imported": 1, "errors": 1})):
            with self.assertRaises(GBrainError):
                self.client.import_directory(self.home)

    def test_missing_initialization_and_executable_are_reported(self):
        (self.home / ".gbrain/config.json").unlink()
        self.assertFalse(self.client.available())
        with self.assertRaisesRegex(GBrainError, "not initialized"):
            self.client.search("escape")
        with patch("memory.gbrain.shutil.which", return_value=None):
            with self.assertRaisesRegex(GBrainError, "not installed"):
                self.client.search("escape")

    def test_input_validation(self):
        self.assertEqual(self.client.search(" "), [])
        for value in (0, -1, 101, True, "5"):
            with self.subTest(value=value), self.assertRaises(ValueError):
                self.client.search("escape", value)


if __name__ == "__main__":
    unittest.main()
