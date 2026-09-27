import io
import json
import unittest
import urllib.error
import urllib.request
import urllib.response
from email.message import Message
from unittest.mock import patch

from memory.memorable import (MemorableClient, MemorableError, MemorableRefused,
                              game_trace_to_memorable, moves_from_draft, validate_extraction)


def response():
    return {"request_id": "request-123", "draft": {"title": "Pac-Man opening",
            "schema_version": "1.0.0", "steps": [
                {"seq": 1, "action": "pacman.step", "activity_class": "execute",
                 "command": "pacman.step --action E", "repeat_count": 2},
                {"seq": 2, "action": "pacman.step", "activity_class": "execute",
                 "command": "pacman.step --action N", "repeat_count": 1}],
            "preconditions": [], "postconditions": []}}


def trace():
    return {"episode_id": "real-episode", "layout": "mediumClassic", "frames": [
        {"action": "E", "action_ok": True, "grid": ["private full board"]},
        {"action": "E"}, {"action": "N", "action_ok": False}]}


class MemorableTests(unittest.TestCase):
    def test_redirect_never_sends_the_bearer_token_to_another_host(self):
        headers = Message()
        headers["Location"] = "https://different-host.example/extract"
        redirected = urllib.response.addinfourl(
            io.BytesIO(b""), headers, "https://example.com/v1/extract", 302)
        redirected.msg = "Found"
        with patch("urllib.request.HTTPSHandler.https_open", return_value=redirected) as transport:
            with self.assertRaisesRegex(MemorableError, "HTTP 302"):
                MemorableClient("mk_secret", base_url="https://example.com").extract_trace(
                    game_trace_to_memorable(trace()))
        self.assertEqual(transport.call_count, 1)
        self.assertEqual(transport.call_args[0][0].full_url, "https://example.com/v1/extract")

    def test_expansion_is_bounded_before_allocating_repeated_moves(self):
        for count in (2001, 10**30):
            with self.subTest(count=count):
                draft = response()["draft"]
                draft["steps"][0]["repeat_count"] = count
                with self.assertRaisesRegex(MemorableError, "move replay limit"):
                    moves_from_draft(draft)

    def test_uses_real_response_and_allowlisted_request(self):
        outgoing = game_trace_to_memorable(trace())
        outgoing["credentials"] = "do-not-send"
        with patch("urllib.request.OpenerDirector.open", return_value=io.BytesIO(json.dumps(response()).encode())) as open_url:
            result = MemorableClient("mk_test").extract_trace(outgoing)
        request = open_url.call_args[0][0]
        sent = json.loads(request.data)
        self.assertEqual(request.full_url, "https://memorable-extraction-api.memorable.workers.dev/v1/extract")
        self.assertEqual(request.get_header("Authorization"), "Bearer mk_test")
        self.assertTrue(request.get_header("User-agent").startswith("PacmanMemory/"))
        self.assertEqual(result["request_id"], "request-123")
        self.assertEqual(moves_from_draft(result["draft"]), ["E", "E", "N"])
        self.assertNotIn("credentials", sent)
        self.assertNotIn("grid", json.dumps(sent))
        self.assertTrue(sent["skip_embedding"])

    def test_does_not_guess_outcomes(self):
        calls = game_trace_to_memorable(trace())["tool_calls"]
        self.assertEqual(calls[0]["result"], {"ok": True})
        self.assertNotIn("result", calls[1])
        self.assertEqual(calls[2]["result"], {"ok": False})

    def test_200_refusal_is_not_success(self):
        refused = dict(response(), refused="allowance_exhausted")
        with patch("urllib.request.OpenerDirector.open", return_value=io.BytesIO(json.dumps(refused).encode())):
            with self.assertRaises(MemorableRefused):
                MemorableClient("mk_test").extract_trace(game_trace_to_memorable(trace()))

    def test_missing_key_never_calls_service(self):
        with patch.dict("os.environ", {}, clear=True), patch("urllib.request.OpenerDirector.open") as open_url:
            with self.assertRaises(MemorableError):
                MemorableClient().extract_trace(game_trace_to_memorable(trace()))
            open_url.assert_not_called()

    def test_http_error_does_not_print_server_detail_or_key(self):
        error = urllib.error.HTTPError("https://example.com", 401, "Unauthorized", {},
                                       io.BytesIO(b'{"error":"unauthorized","detail":"mk_secret"}'))
        with patch("urllib.request.OpenerDirector.open", side_effect=error):
            with self.assertRaises(MemorableError) as raised:
                MemorableClient("mk_secret").extract_trace(game_trace_to_memorable(trace()))
        self.assertEqual(str(raised.exception), "Memorable HTTP 401: unauthorized")

    def test_unknown_or_injected_command_is_rejected(self):
        draft = response()["draft"]
        draft["steps"][0]["command"] = "pacman.step --action E; rm something"
        with self.assertRaises(MemorableError):
            moves_from_draft(draft)

    def test_invalid_steps_are_rejected(self):
        invalid = response()
        invalid["draft"]["steps"][1]["seq"] = 1
        with self.assertRaises(MemorableError):
            validate_extraction(invalid)

    def test_no_executed_actions_is_rejected(self):
        with self.assertRaises(ValueError):
            game_trace_to_memorable({"episode_id": "empty", "frames": []})

    def test_plain_http_endpoint_rejected(self):
        with self.assertRaises(ValueError):
            MemorableClient("mk_secret", base_url="http://example.com")


if __name__ == "__main__":
    unittest.main()
