import hashlib
import io
import json
import tempfile
import unittest
import urllib.error
from pathlib import Path
from unittest.mock import patch

from memory.gbrain import GBrainError
from memory.gbrain_http import GBrainHTTPClient, _ToolFailure


class Response(io.BytesIO):
    def __init__(self, body, content_type="application/json", session=None):
        super().__init__(body)
        self.headers = {"Content-Type": content_type}
        if session:
            self.headers["Mcp-Session-Id"] = session


class HTTPMemoryTests(unittest.TestCase):
    def client(self):
        return GBrainHTTPClient("https://brain.example/mcp", "test-token")

    def test_rejects_invalid_credentials_without_transmitting(self):
        for token in ("", "masked••••", "bad\nheader", "has space"):
            with self.subTest(token_length=len(token)), self.assertRaises(GBrainError):
                GBrainHTTPClient("https://brain.example/mcp", token)
        with self.assertRaises(GBrainError):
            GBrainHTTPClient("http://brain.example/mcp", "test-token")
        with self.assertRaises(GBrainError):
            GBrainHTTPClient("https://user:password@brain.example/mcp", "test-token")

    def test_initialization_propagates_session_and_protocol(self):
        c = self.client()
        responses = [
            Response(b'{"jsonrpc":"2.0","id":1,"result":{"protocolVersion":"2025-03-26"}}', session="session-123"),
            Response(b""),
            Response(b'{"jsonrpc":"2.0","id":3,"result":{"tools":[{"name":"search","inputSchema":{}}]}}'),
        ]
        with patch.object(c._opener, "open", side_effect=responses) as opened:
            self.assertIn("search", c.discover_tools())
        request = opened.call_args_list[2].args[0]
        self.assertEqual(request.get_header("Mcp-session-id"), "session-123")
        self.assertEqual(request.get_header("Mcp-protocol-version"), "2025-03-26")
        self.assertEqual(request.get_header("Accept"), "application/json, text/event-stream")

    def test_sse_skips_notifications_until_matching_response(self):
        raw = (b'event: message\ndata: {"jsonrpc":"2.0","method":"notifications/progress"}\n\n'
               b'event: message\ndata: {"jsonrpc":"2.0","id":8,"result":{"ok":true}}\n\n')
        result = GBrainHTTPClient._decode_response(Response(raw, "text/event-stream; charset=utf-8"), 8)
        self.assertEqual(result["result"], {"ok": True})

    def test_http_error_omits_remote_body_and_credentials(self):
        c = self.client()
        error = urllib.error.HTTPError(c.url, 401, "test-token should not leak", {}, None)
        with patch.object(c._opener, "open", side_effect=error), self.assertRaises(GBrainError) as raised:
            c._rpc("initialize", {})
        self.assertNotIn("test-token", str(raised.exception))
        self.assertIn("401", str(raised.exception))

    def test_hosted_proxy_missing_page_error_is_understood(self):
        c = self.client()
        result = {"isError": True, "structuredContent": {
            "what": "memory_refused", "why": json.dumps({"error": "page_not_found", "message": "private details"}),
            "remedy": "Read the message"}}
        with patch.object(c, "discover_tools", return_value={"get_page": {}}), patch.object(c, "_rpc", return_value=result):
            with self.assertRaises(_ToolFailure) as raised:
                c.get_page("pacman-memory/missing")
        self.assertEqual(raised.exception.code, "page_not_found")
        self.assertNotIn("private details", str(raised.exception))

    def test_reads_and_search_results_are_project_scoped(self):
        c = self.client()
        with self.assertRaises(GBrainError):
            c.get_page("people/private-note")
        with patch.object(c, "_call_tool", return_value=[{"slug": "people/private-note"}, {"slug": "pacman-memory/abc"}]):
            self.assertEqual(c.search("corridor"), [{"slug": "pacman-memory/abc"}])

    def test_search_overfetches_before_namespace_filter_and_result_limit(self):
        c = self.client()
        rows = [{"slug": "people/private-note"}, {"slug": "pacman-memory/first"},
                {"slug": "pacman-memory/second"}]
        with patch.object(c, "_call_tool", return_value=rows) as call:
            self.assertEqual(c.search("corridor", limit=1), [{"slug": "pacman-memory/first"}])
        call.assert_called_once_with("search", {"query": "pacman-memory corridor", "limit": 100})

    def test_tool_absence_is_not_reported_as_empty_memory(self):
        c = self.client()
        with patch.object(c, "discover_tools", return_value={}):
            with self.assertRaisesRegex(GBrainError, "does not expose"):
                c.search("corridor")

    def test_import_is_immutable_and_verified_after_write(self):
        c = self.client()
        c._tools = {"get_page": {}, "put_page": {"properties": {"slug": {}, "content": {}}}}
        stored, writes = {}, []

        def call(name, params):
            slug = params["slug"]
            if name == "get_page":
                if slug not in stored:
                    raise _ToolFailure("page_not_found")
                return stored[slug]
            writes.append(dict(params))
            digest = params["content"].split('pacman_payload_sha256: "')[1].split('"')[0]
            stored[slug] = {"content": params["content"], "frontmatter": {
                "pacman_memory": True, "pacman_payload_sha256": digest}}
            return {"status": "created"}

        with tempfile.TemporaryDirectory() as folder, patch.object(c, "discover_tools", return_value=c._tools), patch.object(c, "_call_tool", side_effect=call):
            page = Path(folder, "aabbccddeeff0011.md")
            page.write_text("# Pac-Man\nAn observed opening.\n")
            self.assertEqual(c.import_directory(folder)["imported"], 1)
            self.assertEqual(c.import_directory(folder)["skipped"], 1)
            page.write_text("# Pac-Man\nA new observed opening.\n")
            self.assertEqual(c.import_directory(folder)["imported"], 1)
        self.assertEqual(len(writes), 2)
        self.assertNotEqual(writes[0]["slug"], writes[1]["slug"])
        self.assertTrue(all(w["slug"].startswith("pacman-memory/") for w in writes))
        self.assertTrue(all(set(w) == {"slug", "content"} for w in writes))

    def test_import_refuses_unowned_page_collision(self):
        c = self.client()
        c._tools = {"get_page": {}, "put_page": {}}
        with tempfile.TemporaryDirectory() as folder:
            Path(folder, "aabbccddeeff0011.md").write_text("# Pac-Man\n")
            with patch.object(c, "discover_tools", return_value=c._tools), patch.object(c, "get_page", return_value={"content": "Unrelated", "frontmatter": {}}), patch.object(c, "_call_tool") as call:
                with self.assertRaisesRegex(GBrainError, "not owned"):
                    c.import_directory(folder)
                call.assert_not_called()

    def test_get_page_checks_body_hash_not_only_metadata(self):
        c = self.client()
        body = "# Original\nRecorded procedure.\n"
        digest = hashlib.sha256(body.encode()).hexdigest()
        slug = "pacman-memory/aabbccddeeff0011-v-" + digest[:12]
        metadata = {"pacman_memory": True, "pacman_payload_sha256": digest}
        page = {"frontmatter": metadata,
                "content": "---\ntitle: A\ntype: note\n---\n\npacman-memory\n\n" + body}
        with patch.object(c, "_call_tool", return_value=page):
            self.assertEqual(c.get_page(slug), page)
            page["content"] = page["content"].replace("Recorded procedure.", "Changed procedure.")
            with self.assertRaisesRegex(GBrainError, "does not match"):
                c.get_page(slug)

    def test_import_does_not_skip_wrong_content_with_matching_metadata(self):
        c = self.client()
        c._tools = {"get_page": {}, "put_page": {}}
        body = "# Correct opening\n"
        digest = hashlib.sha256(body.encode()).hexdigest()
        wrong = {"frontmatter": {"pacman_memory": True, "pacman_payload_sha256": digest},
                 "content": "---\ntype: note\n---\n\npacman-memory\n\n# Incorrect opening\n"}
        with tempfile.TemporaryDirectory() as folder:
            Path(folder, "aabbccddeeff0011.md").write_text(body)
            with patch.object(c, "discover_tools", return_value=c._tools), patch.object(c, "get_page", return_value=wrong):
                with self.assertRaisesRegex(GBrainError, "content differs"):
                    c.import_directory(folder)

    def test_import_verifies_actual_content_after_write(self):
        c = self.client()
        c._tools = {"get_page": {}, "put_page": {}}
        body = "# Correct opening\n"
        digest = hashlib.sha256(body.encode()).hexdigest()
        wrong = {"frontmatter": {"pacman_memory": True, "pacman_payload_sha256": digest},
                 "content": "pacman-memory\n\n# Stale opening\n"}
        with tempfile.TemporaryDirectory() as folder:
            Path(folder, "aabbccddeeff0011.md").write_text(body)
            with patch.object(c, "discover_tools", return_value=c._tools), patch.object(c, "_call_tool", return_value={}), patch.object(c, "get_page", side_effect=[_ToolFailure("page_not_found"), wrong]):
                with self.assertRaisesRegex(GBrainError, "has not published"):
                    c.import_directory(folder)


if __name__ == "__main__":
    unittest.main()
