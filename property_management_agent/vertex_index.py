"""
CLI for pushing all locally-stored content (email_archive +
attachment_extractions) into the Vertex AI Search data store, so it can
be queried side-by-side with the local sqlite-vec RAG.

Run after the one-time GCP setup (see _vertex_search.py docstring):
    python3.11 -m property_management_agent.vertex_index           # default
    python3.11 -m property_management_agent.vertex_index --status  # just probe
    python3.11 -m property_management_agent.vertex_index --limit 10
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path


def _bootstrap() -> None:
    here = Path(__file__).resolve().parent
    repo_root = here.parent
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))
    try:
        from dotenv import load_dotenv  # type: ignore
        env_path = here / ".env"
        if env_path.exists():
            load_dotenv(env_path)
    except ImportError:
        pass


def main() -> int:
    _bootstrap()
    parser = argparse.ArgumentParser(description="Push local content into Vertex AI Search.")
    parser.add_argument("--status", action="store_true",
                        help="Probe Vertex AI Search availability and exit.")
    parser.add_argument("--limit", type=int, default=0,
                        help="Cap docs per source (0 = no cap).")
    parser.add_argument("--json", action="store_true",
                        help="Emit raw JSON instead of human summary.")
    args = parser.parse_args()

    try:
        from property_management_agent import _vertex_search as v
    except Exception as e:
        print(f"FATAL: could not import vertex backend: {e}", file=sys.stderr)
        return 2

    if args.status:
        s = v.get_status()
        if args.json:
            print(json.dumps(s, indent=2))
        else:
            if s.get("available"):
                print(f"✅ Vertex AI Search available.")
                print(f"   Project   : {s.get('project_id')}")
                print(f"   Location  : {s.get('location')}")
                print(f"   Data store: {s.get('data_store')}")
            else:
                print("❌ Vertex AI Search not configured:")
                print(s.get("reason", ""))
        return 0 if s.get("available") else 1

    rep = v.index_everything(limit=args.limit)
    if args.json:
        print(json.dumps(rep, indent=2))
        return 1 if rep.get("isError") or rep.get("needsSetup") else 0

    if rep.get("needsSetup"):
        print("❌ " + rep.get("message", "Setup required."))
        return 2
    ea = rep.get("email_archive", {})
    at = rep.get("attachment_extractions", {})
    print("Indexed into Vertex AI Search:")
    print(f"  email_archive          : {ea.get('indexed', 0)} indexed, "
          f"{ea.get('errors', 0)} errors")
    print(f"  attachment_extractions : {at.get('indexed', 0)} indexed, "
          f"{at.get('errors', 0)} errors")
    for e in (ea.get("errors_detail") or [])[:5]:
        print(f"    ✗ email/{e.get('doc_id')}: {e.get('error', '')[:120]}")
    for e in (at.get("errors_detail") or [])[:5]:
        print(f"    ✗ attachment/{e.get('doc_id')}: {e.get('error', '')[:120]}")
    print()
    print(rep.get("note", ""))
    return 0


if __name__ == "__main__":
    sys.exit(main())
