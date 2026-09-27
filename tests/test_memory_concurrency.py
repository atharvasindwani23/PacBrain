"""Regression for simultaneous four-worker hydration of the same procedure."""

import json
import os
import re
import tempfile
import threading
import unittest
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from unittest.mock import patch

from memory.core import MemoryStore, extract_procedure
from tests.test_memory_edges import corridor_trace


class ConcurrentMemoryTests(unittest.TestCase):
    def test_four_writers_publish_complete_json_and_markdown(self):
        procedure = extract_procedure(corridor_trace())
        json_barrier = threading.Barrier(4)
        markdown_barrier = threading.Barrier(4)
        real_replace = os.replace
        staging_paths = []
        lock = threading.Lock()

        def synchronized_replace(source, destination):
            # All workers finish staging each file before any publication.
            # A fixed shared temporary filename fails this exact schedule.
            with lock:
                staging_paths.append(source)
            suffix = Path(destination).suffix
            (json_barrier if suffix == ".json" else markdown_barrier).wait(timeout=5)
            real_replace(source, destination)

        with tempfile.TemporaryDirectory() as folder:
            store = MemoryStore(folder)
            with patch("memory.core.os.replace", side_effect=synchronized_replace), ThreadPoolExecutor(max_workers=4) as pool:
                futures = [pool.submit(store.save, procedure) for _ in range(4)]
                for future in futures:
                    self.assertEqual(future.result(timeout=8).name, procedure["id"] + ".json")
            self.assertEqual(len(staging_paths), 8)
            self.assertEqual(len(set(staging_paths)), 8)
            self.assertEqual(store.procedures(), [procedure])
            markdown = Path(folder, procedure["id"] + ".md").read_text()
            payload = re.search(r"```json\n(.*?)\n```", markdown, re.DOTALL).group(1)
            self.assertEqual(json.loads(payload), procedure)
            self.assertEqual(sorted(p.suffix for p in Path(folder).iterdir()), [".json", ".md"])


if __name__ == "__main__":
    unittest.main()
