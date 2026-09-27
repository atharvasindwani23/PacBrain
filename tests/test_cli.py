import io
import json
import os
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest.mock import patch

from memory.__main__ import main
from tests.test_memory import trace


class LocalCommandTests(unittest.TestCase):
    def test_local_extract_and_recall_ignore_incomplete_hosted_configuration(self):
        with tempfile.TemporaryDirectory() as folder, patch.dict(
                os.environ, {"GBRAIN_MCP_URL": "https://example.com/mcp"}, clear=True):
            input_file = Path(folder) / "trace.json"
            input_file.write_text(json.dumps(trace()))
            base = ["--env-file", folder + "/absent.env", "--memory-dir", folder + "/memory"]
            with redirect_stdout(io.StringIO()) as output:
                code = main(base + ["extract", str(input_file), "--extractor", "local"])
            self.assertEqual(code, 0)
            extracted = json.loads(output.getvalue())
            self.assertFalse(extracted["gbrain_indexed"])
            with redirect_stdout(io.StringIO()) as output:
                code = main(base + ["recall", "--layout", "testCorridor", "--provider", "local"])
            self.assertEqual(code, 0)
            recalled = json.loads(output.getvalue())
            self.assertEqual(recalled["provider"], "local")
            self.assertEqual(recalled["procedure"]["id"], extracted["procedure_id"])


if __name__ == "__main__":
    unittest.main()
