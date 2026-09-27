"""Authenticated GBrain Streamable HTTP MCP, using only Python's stdlib.

Remote writes are confined to app-owned pages under pacman-memory/. No token or
raw remote error body is included in exceptions. Tool schemas are discovered
before use; configuration alone is never reported as live verification.
"""

import hashlib
import json
import re
import socket
import urllib.error
import urllib.parse
import urllib.request
import uuid
from pathlib import Path

from .gbrain import GBrainError


class _NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, *args, **kwargs):
        return None


class _ToolFailure(GBrainError):
    def __init__(self, code):
        self.code = code
        super().__init__("GBrain rejected the memory operation. Check the client's memory permissions and hosted activity record.")


class GBrainHTTPClient:
    def __init__(self, url, token, timeout=30, namespace="pacman-memory"):
        parsed = urllib.parse.urlsplit(url)
        if (parsed.scheme != "https" or not parsed.hostname or parsed.username
                or parsed.password or parsed.fragment or parsed.query):
            raise GBrainError("GBRAIN_MCP_URL must be an HTTPS endpoint without embedded credentials, query, or fragment.")
        if not isinstance(token, str) or not token or any(ord(c) < 33 or ord(c) > 126 for c in token):
            raise GBrainError("GBRAIN_ACCESS_TOKEN must be the actual unmasked ASCII bearer token, without whitespace.")
        if namespace != "pacman-memory":
            raise ValueError("remote writes are restricted to the pacman-memory namespace")
        if timeout <= 0:
            raise ValueError("timeout must be positive")
        self.url, self._token, self.timeout = url, token, timeout
        self.namespace = namespace
        self._session_id = None
        self._version = "2025-03-26"
        self._counter = 0
        self._initialized = False
        self._tools = None
        self._opener = urllib.request.build_opener(_NoRedirect())

    def available(self):
        """Configuration presence only; discover_tools performs a live check."""
        return True

    @staticmethod
    def _decode_response(response, request_id):
        limit = 8 * 1024 * 1024
        if response.headers.get("Content-Type", "").split(";", 1)[0] != "text/event-stream":
            raw = response.read(limit + 1)
            if len(raw) > limit:
                raise GBrainError("GBrain response exceeded the 8 MB limit.")
            return json.loads(raw.decode("utf-8")) if raw else {}
        total, data = 0, []
        while True:
            raw = response.readline(limit + 1)
            if not raw:
                break
            total += len(raw)
            if total > limit:
                raise GBrainError("GBrain event stream exceeded the 8 MB limit.")
            line = raw.decode("utf-8").rstrip("\r\n")
            if line.startswith("data:"):
                data.append(line[5:].lstrip(" "))
            elif not line and data:
                message = json.loads("\n".join(data))
                data = []
                if isinstance(message, dict) and message.get("id") == request_id:
                    return message
        if data:
            message = json.loads("\n".join(data))
            if isinstance(message, dict) and message.get("id") == request_id:
                return message
        raise GBrainError("GBrain closed its event stream before responding.")

    def _rpc(self, method, params=None, notification=False):
        self._counter += 1
        request_id = self._counter
        payload = {"jsonrpc": "2.0", "method": method}
        if not notification:
            payload["id"] = request_id
        if params is not None:
            payload["params"] = params
        headers = {"Authorization": "Bearer " + self._token,
                   "Content-Type": "application/json",
                   "Accept": "application/json, text/event-stream"}
        if self._initialized:
            headers["MCP-Protocol-Version"] = self._version
        if self._session_id:
            headers["Mcp-Session-Id"] = self._session_id
        request = urllib.request.Request(self.url, data=json.dumps(payload).encode("utf-8"), headers=headers)
        try:
            with self._opener.open(request, timeout=self.timeout) as response:
                session = response.headers.get("Mcp-Session-Id")
                if session:
                    if any(ord(c) < 33 or ord(c) > 126 for c in session):
                        raise GBrainError("GBrain returned an invalid MCP session identifier.")
                    self._session_id = session
                if notification:
                    return {}
                message = self._decode_response(response, request_id)
        except urllib.error.HTTPError as error:
            if error.code in (401, 403):
                hint = "Check the bearer token and grant this client Full access to the intended workspace memory."
            elif error.code == 404:
                hint = "The endpoint or MCP session was not found; create a new client and reconnect."
            elif error.code == 429:
                hint = "The hosted request allowance was reached; wait for the provider limit to reset."
            elif 300 <= error.code < 400:
                hint = "Redirects are refused so the bearer token cannot be forwarded; configure the final HTTPS MCP endpoint."
            else:
                hint = "Check the hosted workspace activity and availability."
            raise GBrainError("GBrain HTTP %s. %s" % (error.code, hint)) from None
        except (urllib.error.URLError, socket.timeout, TimeoutError, OSError):
            raise GBrainError("GBrain could not be reached within the timeout. Check network access and the hosted endpoint.") from None
        except (ValueError, UnicodeError):
            raise GBrainError("GBrain returned malformed JSON or an invalid event stream.") from None
        if not isinstance(message, dict) or message.get("id") != request_id:
            raise GBrainError("GBrain returned a mismatched JSON-RPC response.")
        if message.get("error"):
            raise GBrainError("GBrain rejected the MCP request. Verify the endpoint, protocol, and client's tool permissions.")
        if "result" not in message:
            raise GBrainError("GBrain's MCP response did not contain a result.")
        return message["result"]

    def _initialize(self):
        if self._initialized:
            return
        result = self._rpc("initialize", {"protocolVersion": self._version, "capabilities": {},
                                        "clientInfo": {"name": "pacman-memory", "version": "1.0"}})
        version = result.get("protocolVersion") if isinstance(result, dict) else None
        if version not in ("2025-03-26", "2025-06-18", "2025-11-25"):
            raise GBrainError("GBrain negotiated an unsupported MCP protocol version.")
        self._version = version
        self._initialized = True
        self._rpc("notifications/initialized", notification=True)

    def discover_tools(self):
        """Return names and input schemas only; no account memory is read."""
        self._initialize()
        if self._tools is not None:
            return self._tools
        discovered, cursor, seen = {}, None, set()
        for _ in range(20):
            result = self._rpc("tools/list", {"cursor": cursor} if cursor else {})
            if not isinstance(result, dict) or not isinstance(result.get("tools"), list):
                raise GBrainError("GBrain returned an invalid MCP tool catalog.")
            for tool in result["tools"]:
                if isinstance(tool, dict) and isinstance(tool.get("name"), str):
                    discovered[tool["name"]] = tool.get("inputSchema", {})
            cursor = result.get("nextCursor")
            if not cursor:
                self._tools = discovered
                return discovered
            if not isinstance(cursor, str) or cursor in seen:
                break
            seen.add(cursor)
        raise GBrainError("GBrain's tool catalog pagination did not complete.")

    def _call_tool(self, name, arguments):
        if name not in self.discover_tools():
            raise GBrainError("The hosted connection does not expose the required %s tool. Check its workspace-memory application and permission level." % name)
        result = self._rpc("tools/call", {"name": name, "arguments": arguments})
        if not isinstance(result, dict):
            raise GBrainError("GBrain returned an invalid MCP tool response.")
        value = result.get("structuredContent")
        if value is None:
            blocks = result.get("content", [])
            texts = [b.get("text") for b in blocks if isinstance(b, dict) and b.get("type") == "text"]
            try:
                value = json.loads("\n".join(texts))
            except (ValueError, TypeError):
                if result.get("isError"):
                    raise _ToolFailure("unknown") from None
                raise GBrainError("GBrain's tool response was not structured JSON.") from None
        if result.get("isError") or (isinstance(value, dict) and value.get("error")):
            error = value.get("error") if isinstance(value, dict) else None
            # gbrain.io wraps the engine's JSON error as a string in `why`.
            if isinstance(value, dict) and value.get("what") == "memory_refused":
                try:
                    engine_error = json.loads(value.get("why", ""))
                    error = engine_error.get("error") if isinstance(engine_error, dict) else error
                except (ValueError, TypeError):
                    pass
            code = error.get("code") if isinstance(error, dict) else error
            raise _ToolFailure(code if isinstance(code, str) else "unknown")
        return value

    def search(self, query, limit=5):
        if not isinstance(query, str) or not query.strip():
            return []
        if type(limit) is not int or not 1 <= limit <= 100:
            raise ValueError("limit must be an integer between 1 and 100")
        # Ask for a bounded candidate pool before excluding unrelated memory.
        # The hosted provider has also returned newly saved pages at a wider
        # limit while omitting them from the same query with limit 20.
        result = self._call_tool("search", {"query": self.namespace + " " + query, "limit": 100})
        rows = result.get("results") if isinstance(result, dict) else result
        if not isinstance(rows, list) or any(not isinstance(row, dict) for row in rows):
            raise GBrainError("GBrain search returned an unexpected result shape.")
        # Never hydrate unrelated workspace memory, even if search ranks it.
        return [row for row in rows if isinstance(row.get("slug"), str)
                and row["slug"].startswith(self.namespace + "/")][:limit]

    def get_page(self, slug):
        if (not isinstance(slug, str) or not slug.startswith(self.namespace + "/")
                or not re.fullmatch(r"[a-z0-9][a-z0-9/_-]*", slug) or "//" in slug):
            raise GBrainError("Hosted page reads are restricted to pacman-memory/.")
        result = self._call_tool("get_page", {"slug": slug, "include_content": True})
        if not isinstance(result, dict) or not isinstance(result.get("content"), str):
            raise GBrainError("GBrain get_page did not return canonical Markdown content.")
        payload = self._canonical_payload(result)
        metadata = result.get("frontmatter", {})
        digest = metadata.get("pacman_payload_sha256") if isinstance(metadata, dict) else None
        if (not isinstance(metadata, dict) or metadata.get("pacman_memory") is not True
                or not isinstance(digest, str) or not re.fullmatch(r"[a-f0-9]{64}", digest)):
            raise GBrainError("GBrain page is missing the Pac-Man app's integrity metadata.")
        if hashlib.sha256(payload.encode("utf-8")).hexdigest() != digest or not slug.endswith("-v-" + digest[:12]):
            raise GBrainError("GBrain page content does not match its recorded payload or immutable version.")
        return result

    def _canonical_payload(self, page):
        """Read body bytes independent of canonical YAML key ordering."""
        content = page.get("content")
        if not isinstance(content, str):
            raise GBrainError("GBrain page has no canonical Markdown content.")
        content = content.replace("\r\n", "\n")
        if content.startswith("---\n"):
            _, separator, content = content.partition("\n---\n")
            if not separator:
                raise GBrainError("GBrain page has malformed canonical frontmatter.")
            content = content.lstrip("\n")
        prefix = self.namespace + "\n\n"
        if not content.startswith(prefix):
            raise GBrainError("GBrain page is missing the Pac-Man payload marker.")
        return content[len(prefix):]

    def import_directory(self, directory):
        directory = Path(directory).expanduser().resolve()
        if not directory.is_dir():
            raise GBrainError("The memory Markdown import directory does not exist.")
        self.discover_tools()
        if "put_page" not in self._tools or "get_page" not in self._tools:
            raise GBrainError("Hosted import requires get_page and put_page; grant Full workspace-memory access to this connection.")
        imported, skipped = 0, 0
        for path in sorted(directory.rglob("*.md")):
            if path.is_symlink() or not path.is_file():
                continue
            relative = path.relative_to(directory).with_suffix("").as_posix()
            if not re.fullmatch(r"[a-z0-9][a-z0-9/_-]*", relative):
                raise GBrainError("Procedure Markdown filenames must use lowercase letters, numbers, hyphens, or underscores.")
            body = path.read_text(encoding="utf-8")
            if body.startswith("---\n"):
                raise GBrainError("Hosted import expects the app's body-only procedure Markdown, not an arbitrary frontmatter document.")
            digest = hashlib.sha256(body.encode("utf-8")).hexdigest()
            # The hosted catalog may predate compare-and-swap writes. Immutable
            # content-addressed versions never replace an existing procedure.
            slug = self.namespace + "/" + relative + "-v-" + digest[:12]
            content = ('---\ntype: note\ntitle: "Pac-Man memory ' + relative.replace("/", " ")
                       + '"\npacman_memory: true\npacman_payload_sha256: "' + digest
                       + '"\n---\n\n' + self.namespace + '\n\n' + body)
            params = {"slug": slug, "content": content}
            properties = self._tools["put_page"].get("properties", {})
            if "request_id" in properties:
                params["request_id"] = str(uuid.uuid5(uuid.NAMESPACE_URL, self.url + "/" + slug + "/" + digest))
            try:
                previous = self.get_page(slug)
            except _ToolFailure as error:
                if error.code not in ("page_not_found", "not_found"):
                    raise
                previous = None
            if previous:
                metadata = previous.get("frontmatter", {})
                if not isinstance(metadata, dict) or metadata.get("pacman_memory") is not True:
                    raise GBrainError("An existing hosted page at this slug is not owned by the Pac-Man app; refusing to overwrite it.")
                if metadata.get("pacman_payload_sha256") == digest:
                    if self._canonical_payload(previous) != body:
                        raise GBrainError("Existing GBrain page content differs from the requested payload; refusing to report a successful import.")
                    skipped += 1
                    continue
                raise GBrainError("An existing hosted page differs from this content-addressed version; refusing to overwrite it.")
            self._call_tool("put_page", params)
            # A returned mutation receipt can be pending; actual readable
            # content is required before reporting a successful import.
            verified = self.get_page(slug)
            metadata = verified.get("frontmatter", {})
            if (not isinstance(metadata, dict) or metadata.get("pacman_memory") is not True
                    or metadata.get("pacman_payload_sha256") != digest
                    or self._canonical_payload(verified) != body):
                raise GBrainError("GBrain has not published the requested page content yet. Retry the same import to recover its outcome.")
            imported += 1
        return {"status": "success", "imported": imported, "skipped": skipped,
                "errors": 0, "transport": "http_mcp", "namespace": self.namespace}
