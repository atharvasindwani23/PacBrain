"""Small, keyless adapter for the official garrytan/gbrain CLI.

The adapter invokes real GBrain commands. It never substitutes local search when
GBrain is unavailable; the caller decides how to expose a degraded mode.
"""

import json
import os
from pathlib import Path
import shutil
import subprocess
from typing import Any, Dict, List


class GBrainError(RuntimeError):
    """GBrain is missing, unavailable, or returned an unsuccessful response."""


class GBrainClient:
    def __init__(self, home: Path, binary: str = "gbrain", timeout: float = 30):
        self.home = Path(home).expanduser().resolve()
        self.binary = str(Path(binary).expanduser().resolve()) if "/" in str(binary) else str(binary)
        if timeout <= 0:
            raise ValueError("timeout must be positive")
        self.timeout = timeout

    def available(self) -> bool:
        """Check installation and isolated configuration without making a call."""
        return bool(shutil.which(self.binary)) and (self.home / ".gbrain" / "config.json").is_file()

    def _run(self, args: List[str]) -> Any:
        if not shutil.which(self.binary):
            raise GBrainError("GBrain CLI is not installed. Run bash scripts/setup_gbrain.sh and set GBRAIN_BIN to the printed launcher path.")
        if not (self.home / ".gbrain" / "config.json").is_file():
            raise GBrainError("The project GBrain is not initialized. Run bash scripts/setup_gbrain.sh with this same GBRAIN_HOME.")
        # GBRAIN_HOME is the parent of .gbrain, not the configuration directory.
        # Do not let an unrelated database or source selector redirect this
        # deliberately local brain. Running outside the checkout also prevents
        # Bun from automatically loading the application's .env file.
        env = {key: value for key, value in os.environ.items()
               if not key.startswith("GBRAIN_") and key != "DATABASE_URL"}
        env.update(GBRAIN_HOME=str(self.home), GBRAIN_SKIP_STARTUP_HOOKS="1", GBRAIN_MEMORABLE="0")
        try:
            result = subprocess.run(
                [self.binary] + args,
                cwd=str(self.home), env=env, capture_output=True,
                text=True, timeout=self.timeout, check=False,
            )
        except subprocess.TimeoutExpired:
            raise GBrainError("GBrain timed out. Stop any GBrain server or Memorable viewer holding this PGLite database, then retry; increase timeout for a large import.") from None
        except OSError:
            raise GBrainError("GBrain could not start. Check the launcher path and its Bun runtime with bash scripts/setup_gbrain.sh.") from None
        if result.returncode:
            # Raw subprocess output may contain private URLs or credentials.
            # Classify known failures without echoing that output into traces.
            diagnostic = (result.stderr + result.stdout).lower()
            if any(token in diagnostic for token in ("pglite_busy", "live_serve", "lock", "database is busy")):
                hint = "Stop the other GBrain server or Memorable viewer using this PGLite database; do not remove a live database lock."
            elif "embedding" in diagnostic:
                hint = "Use a keyless brain initialized with --pglite --no-embedding; imports must use --no-embed."
            elif "configured" in diagnostic or "init" in diagnostic:
                hint = "Run bash scripts/setup_gbrain.sh with this project's GBRAIN_HOME."
            else:
                hint = "Run the isolated GBrain launcher with doctor --json to inspect its configuration."
            raise GBrainError("GBrain %s failed (exit %s). %s" % (args[0], result.returncode, hint))
        try:
            return json.loads(result.stdout)
        except (ValueError, TypeError):
            raise GBrainError("GBrain returned invalid JSON. Verify the official CLI version with the project launcher; its --json stdout must contain one JSON value.") from None

    def import_directory(self, directory: Path) -> Dict[str, Any]:
        """Index Markdown, including ignored runtime artifacts, without APIs."""
        directory = Path(directory).expanduser().resolve()
        if not directory.is_dir():
            raise GBrainError("The memory Markdown import directory does not exist.")
        result = self._run([
            "import", str(directory), "--no-embed", "--include-gitignored",
            "--fresh", "--json",
        ])
        if not isinstance(result, dict):
            raise GBrainError("GBrain import returned an unexpected JSON shape; expected an import summary object.")
        if result.get("errors", 0):
            raise GBrainError("GBrain could not import every memory page. Run the project GBrain doctor and inspect the generated Markdown.")
        return result

    def search(self, query: str, limit: int = 5) -> List[Dict[str, Any]]:
        """Return GBrain's actual ranked rows, preserving slugs and metadata."""
        if not isinstance(query, str) or not query.strip():
            return []
        if isinstance(limit, bool) or not isinstance(limit, int) or not 1 <= limit <= 100:
            raise ValueError("limit must be an integer between 1 and 100")
        result = self._run(["search", query, "--limit", str(limit), "--json"])
        if not isinstance(result, list) or any(not isinstance(row, dict) for row in result):
            raise GBrainError("GBrain search returned an unexpected JSON shape; expected an array of result objects.")
        return result

    def get_page(self, slug: str) -> Dict[str, Any]:
        """Hydrate a result from GBrain even when this runner lacks local JSON."""
        if not isinstance(slug, str) or not slug.strip():
            raise ValueError("slug must be a nonempty string")
        result = self._run(["get", slug, "--json", "--include-content"])
        if not isinstance(result, dict) or not isinstance(result.get("content"), str):
            raise GBrainError("GBrain get returned an unexpected JSON shape; expected a page with canonical Markdown content.")
        return result
