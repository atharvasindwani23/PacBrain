"""Offline integration demo, explicitly using a synthetic trace and local memory."""

import json
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from memory.core import MemoryStore, ProcedureCursor
from memory.service import ingest, recall_procedure


def main():
    fixture = Path(__file__).resolve().parents[1] / "examples" / "synthetic_trace.json"
    trace = json.loads(fixture.read_text())
    with tempfile.TemporaryDirectory() as directory:
        store = MemoryStore(directory)
        saved = ingest(trace, store, extractor="local")
        recalled = recall_procedure(trace["layout"], trace["frames"][0], store, "local")
        cursor = ProcedureCursor(recalled["procedure"])
        first = cursor.next_action(trace["frames"][0], trace["layout"])
        changed = dict(trace["frames"][1], grid="%%%%%%%%%\n% PG... %\n%%%%%%%%%")
        interrupted = cursor.next_action(changed, trace["layout"])
        assert first == "E" and interrupted is None and cursor.reason == "ghost_nearby"
        print(json.dumps({"synthetic_fixture": True, "live_sponsor_calls": False,
                          "stored": saved, "first_recalled_action": first,
                          "action_when_ghost_changes": interrupted, "stop_reason": cursor.reason}, indent=2))


if __name__ == "__main__":
    main()
