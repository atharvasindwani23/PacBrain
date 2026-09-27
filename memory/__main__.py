"""python3 -m memory: setup diagnostics, explicit extraction, and retrieval."""

import argparse
import json
import os
import sys
from pathlib import Path

from .core import MemoryStore
from .gbrain import GBrainError
from .gbrain_http import GBrainHTTPClient
from .memorable import MemorableError
from .service import gbrain_client, ingest, load_env, recall_procedure


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--env-file", default=".env")
    parser.add_argument("--memory-dir", default="data/memory")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("doctor", help="Local configuration checks; does not spend API credits")
    extract = sub.add_parser("extract", help="Extract one real trace, optionally index in GBrain")
    extract.add_argument("trace")
    extract.add_argument("--extractor", choices=["memorable", "local"], default="memorable")
    extract.add_argument("--gbrain", action="store_true")
    extract.add_argument("--max-steps", type=int, default=40)
    recall = sub.add_parser("recall")
    recall.add_argument("--layout", required=True)
    recall.add_argument("--provider", choices=["gbrain", "local"], default="gbrain")
    recall.add_argument("--observation", help="JSON observation to validate against")
    args = parser.parse_args(argv)
    try:
        load_env(args.env_file)
        client = gbrain_client()
        store = MemoryStore(args.memory_dir)
        if args.command == "doctor":
            hosted = isinstance(client, GBrainHTTPClient)
            result = {"memorable_key_configured": bool(os.getenv("MEMORABLE_API_KEY")),
                      "memorable_live_verified": False, "gbrain_configured": client.available(),
                      "gbrain_transport": "http_mcp" if hosted else "local_cli",
                      "local_procedures": len(store.procedures()),
                      "note": "Configuration only; live readiness requires an extraction and recall."}
            if not hosted:
                result.update(gbrain_home=str(client.home), gbrain_binary=client.binary)
        elif args.command == "extract":
            trace = json.loads(Path(args.trace).read_text())
            result = ingest(trace, store=store, extractor=args.extractor,
                            gbrain=client if args.gbrain else None, max_steps=args.max_steps)
        else:
            observation = json.loads(Path(args.observation).read_text()) if args.observation else None
            result = recall_procedure(args.layout, observation, store, args.provider, client)
        print(json.dumps(result, indent=2))
        return 0
    except (ValueError, OSError, MemorableError, GBrainError) as exc:
        print(json.dumps({"error": str(exc)}), file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
