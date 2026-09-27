"""The documented Memorable extraction API, with no simulated sponsor output.

Memorable extracts a tool trace; the caller owns persistence and retrieval.
There is intentionally no invented cloud recall endpoint in this adapter.
"""

import json
import os
import re
import urllib.error
import urllib.parse
import urllib.request
from typing import Any, Dict, Optional


DEFAULT_API_URL = "https://memorable-extraction-api.memorable.workers.dev"
MAX_BODY_BYTES = 8 * 1024 * 1024
INPUT_FIELDS = frozenset(("command", "file_path", "path", "pattern", "url", "query"))
ACTIONS = {"N": "N", "S": "S", "E": "E", "W": "W",
           "North": "N", "South": "S", "East": "E", "West": "W"}


class MemorableError(RuntimeError):
    """A bounded error that never contains the request body or API key."""


class MemorableRefused(MemorableError):
    """A refusal must not be persisted or retried automatically."""


def _safe_label(value: Any) -> str:
    if not isinstance(value, str):
        return "unknown"
    return re.sub(r"[^A-Za-z0-9_.:-]", "", value)[:120] or "unknown"


def _wire_trace(trace: Dict[str, Any]) -> Dict[str, Any]:
    """Allowlist exactly the documented API fields; do not send whole games."""
    if not isinstance(trace, dict):
        raise ValueError("Memorable trace must be an object")
    for name, limit in (("session_id", 200), ("task_description", 2000), ("harness", 60)):
        value = trace.get(name)
        if not isinstance(value, str) or not value.strip() or len(value) > limit:
            raise ValueError("Memorable trace requires a valid " + name)
    calls = trace.get("tool_calls")
    if not isinstance(calls, list) or not 1 <= len(calls) <= 2000:
        raise ValueError("Memorable trace requires 1 to 2000 tool calls")
    clean_calls = []
    for call in calls:
        if not isinstance(call, dict) or not isinstance(call.get("name"), str) or not call["name"]:
            raise ValueError("Every tool call requires a name")
        args = call.get("input")
        if not isinstance(args, dict):
            raise ValueError("Every tool call requires an input object")
        clean_args = {key: value for key, value in args.items()
                      if key in INPUT_FIELDS and isinstance(value, str)}
        if not clean_args:
            raise ValueError("Tool input has no documented string argument fields")
        clean = {"name": call["name"], "input": clean_args}
        result = call.get("result")
        if result is not None:
            if not isinstance(result, dict):
                raise ValueError("Tool outcome must be an object")
            if isinstance(result.get("ok"), bool):
                clean["result"] = {"ok": result["ok"]}
            elif type(result.get("exit_code")) is int:
                clean["result"] = {"exit_code": result["exit_code"]}
            else:
                raise ValueError("Tool outcome requires a known boolean ok or integer exit_code")
        clean_calls.append(clean)
    return {"session_id": trace["session_id"], "task_description": trace["task_description"],
            "harness": trace["harness"], "skip_embedding": True, "tool_calls": clean_calls}


def validate_extraction(response: Any) -> Dict[str, Any]:
    """Keep provenance intact; never turn a refusal into a local success."""
    if not isinstance(response, dict):
        raise MemorableError("Memorable returned a non-object response")
    if response.get("refused"):
        raise MemorableRefused("Memorable refused extraction: " + _safe_label(response["refused"]))
    if response.get("error"):
        raise MemorableError("Memorable extraction failed: " + _safe_label(response["error"]))
    draft = response.get("draft")
    if not isinstance(draft, dict) or not isinstance(draft.get("title"), str) or not draft["title"].strip():
        raise MemorableError("Memorable response has no valid draft title")
    steps = draft.get("steps")
    if not isinstance(steps, list) or not steps:
        raise MemorableError("Memorable response has no extracted steps")
    last_seq = 0
    for step in steps:
        if not isinstance(step, dict) or type(step.get("seq")) is not int or step["seq"] <= last_seq:
            raise MemorableError("Memorable draft has invalid step ordering")
        last_seq = step["seq"]
        if not isinstance(step.get("action"), str) or not step["action"]:
            raise MemorableError("Memorable draft step has no action")
        if "command" in step and not isinstance(step["command"], str):
            raise MemorableError("Memorable draft step command must be a string")
        if "repeat_count" in step and (type(step["repeat_count"]) is not int or step["repeat_count"] < 1):
            raise MemorableError("Memorable draft step has invalid repeat_count")
    for field in ("preconditions", "postconditions"):
        if field in draft and (not isinstance(draft[field], list) or
                               not all(isinstance(x, str) for x in draft[field])):
            raise MemorableError("Memorable draft has invalid " + field)
    request_id = response.get("request_id")
    if request_id is not None and not isinstance(request_id, str):
        raise MemorableError("Memorable response has an invalid request_id")
    return {"draft": draft, "request_id": request_id}


class MemorableClient:
    def __init__(self, api_key: Optional[str] = None, timeout: float = 20,
                 base_url: Optional[str] = None):
        self.api_key = api_key or os.getenv("MEMORABLE_API_KEY")
        self.timeout = timeout
        self.base_url = (base_url or os.getenv("MEMORABLE_API_URL") or DEFAULT_API_URL).rstrip("/")
        url = urllib.parse.urlparse(self.base_url)
        if url.scheme != "https" or not url.netloc or url.username or url.password or url.query or url.fragment:
            raise ValueError("MEMORABLE_API_URL must be a credential-free HTTPS URL")
        if timeout <= 0:
            raise ValueError("Memorable timeout must be positive")

    @classmethod
    def from_env(cls, timeout: float = 20) -> "MemorableClient":
        return cls(timeout=timeout)

    def extract_trace(self, trace: Dict[str, Any]) -> Dict[str, Any]:
        if not self.api_key:
            raise MemorableError("Set MEMORABLE_API_KEY from https://memorable.sh/dash before live extraction")
        payload = json.dumps(_wire_trace(trace), separators=(",", ":"), allow_nan=False).encode("utf-8")
        if len(payload) > MAX_BODY_BYTES:
            raise ValueError("Memorable trace exceeds the 8 MB request limit")
        request = urllib.request.Request(self.base_url + "/v1/extract", data=payload,
                                         headers={"Authorization": "Bearer " + self.api_key,
                                                  "Content-Type": "application/json",
                                                  # Cloudflare's browser-integrity check rejects
                                                  # urllib's default Python User-Agent (1010).
                                                  "User-Agent": "PacmanMemory/1.0 (+https://www.memorable.sh/docs/integrate)"},
                                         method="POST")
        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as reply:
                raw = reply.read(MAX_BODY_BYTES + 1)
        except urllib.error.HTTPError as exc:
            # Server detail can echo user input. Only expose a safe status and code.
            code = "request_failed"
            try:
                body = json.loads(exc.read(8192))
                if isinstance(body, dict):
                    code = _safe_label(body.get("error", code))
            except (ValueError, OSError):
                pass
            raise MemorableError("Memorable HTTP %s: %s" % (exc.code, code)) from None
        except (urllib.error.URLError, TimeoutError, OSError):
            raise MemorableError("Memorable extraction connection failed or timed out") from None
        if len(raw) > MAX_BODY_BYTES:
            raise MemorableError("Memorable returned an oversized response")
        try:
            response = json.loads(raw)
        except (ValueError, UnicodeDecodeError):
            raise MemorableError("Memorable returned invalid JSON") from None
        return validate_extraction(response)


def game_trace_to_memorable(trace: Dict[str, Any], max_steps: int = 40) -> Dict[str, Any]:
    """Serialize actual executed game actions into the generic harness contract.

    Frames must be pre-action records, one per action actually passed to env.step.
    Optional frame.action_ok is the environment's recorded tool outcome. A game
    score, non-death, or presence of another frame does not invent that outcome.
    """
    if type(max_steps) is not int or not 1 <= max_steps <= 2000:
        raise ValueError("max_steps must be between 1 and 2000")
    calls = []
    for frame in trace.get("frames", [])[:max_steps]:
        if not isinstance(frame, dict) or frame.get("action") not in ACTIONS:
            raise ValueError("Every extracted game frame must contain an executed N/S/E/W action")
        action = ACTIONS[frame["action"]]
        call = {"name": "pacman.step", "input": {"command": "pacman.step --action " + action}}
        if isinstance(frame.get("action_ok"), bool):
            call["result"] = {"ok": frame["action_ok"]}
        calls.append(call)
    return _wire_trace({"session_id": trace.get("episode_id"),
                        "task_description": "Pac-Man %s opening: replay %d executed moves" %
                                            (trace.get("layout", "unknown"), len(calls)),
                        "harness": "pacman-agent", "tool_calls": calls})


def moves_from_draft(draft: Dict[str, Any]) -> list:
    """Strictly parse the returned procedure; never execute a recalled command.

    Repeated identical commands may be collapsed by the extractor. Any other
    action or syntax makes the draft unusable for our move replay contract.
    """
    validate_extraction({"draft": draft})
    moves = []
    for step in draft["steps"]:
        match = re.fullmatch(r"pacman\.step --action ([NSEW])", step.get("command", ""))
        if not match:
            raise MemorableError("Extracted draft contains an unsupported game command")
        moves.extend([match.group(1)] * step.get("repeat_count", 1))
        if len(moves) > 2000:
            raise MemorableError("Extracted draft exceeds the move replay limit")
    return moves
