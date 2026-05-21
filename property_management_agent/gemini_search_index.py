"""
CLI for the Gemini File Search backend.

Run:
    python3.11 -m property_management_agent.gemini_search_index --status
    python3.11 -m property_management_agent.gemini_search_index             # push all
    python3.11 -m property_management_agent.gemini_search_index --limit 10
    python3.11 -m property_management_agent.gemini_search_index --json
"""
from __future__ import annotations

import argparse
import json
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
    p = argparse.ArgumentParser(description="Push local content into Gemini File Search.")
    p.add_argument("--status", action="store_true", help="Probe availability and exit.")
    p.add_argument("--limit", type=int, default=0, help="Cap docs per source.")
    p.add_argument("--json", action="store_true", help="Emit raw JSON.")
    args = p.parse_args()

    try:
        from property_management_agent import _gemini_file_search as g
    except Exception as e:
        print(f"FATAL: could not import Gemini File Search backend: {e}", file=sys.stderr)
        return 2

    if args.status:
        s = g.get_status()
        if args.json:
            print(json.dumps(s, indent=2))
        else:
            if s.get("available"):
                print("✅ Gemini File Search reachable.")
                if s.get("store_exists"):
                    print(f"   Store           : {s.get('display_name')}  ({s.get('store_name','')})")
                    print(f"   Active docs     : {s.get('active_documents')}")
                    print(f"   Pending docs    : {s.get('pending_documents')}")
                    print(f"   Failed docs     : {s.get('failed_documents')}")
                    print(f"   Size            : {s.get('size_bytes')} bytes")
                    print(f"   Embedding model : {s.get('embedding_model')}")
                else:
                    print("   Store not yet created — will be on first index/search.")
            else:
                print("❌ Gemini File Search not available:")
                print(s.get("reason", ""))
        return 0 if s.get("available") else 1

    rep = g.index_everything(limit=args.limit)
    if args.json:
        print(json.dumps(rep, indent=2))
        return 1 if rep.get("isError") or rep.get("needsSetup") else 0

    if rep.get("needsSetup"):
        print("❌ " + rep.get("message", "Setup required."))
        return 2
    ea = rep.get("email_archive", {})
    at = rep.get("attachment_extractions", {})
    print("Indexed into Gemini File Search:")
    print(f"  email_archive          : {ea.get('indexed', 0)} indexed, "
          f"{ea.get('errors', 0)} errors")
    print(f"  attachment_extractions : {at.get('indexed', 0)} indexed, "
          f"{at.get('errors', 0)} errors")
    for e in (ea.get("errors_detail") or [])[:5]:
        print(f"    ✗ {e.get('source_id')}: {e.get('error', '')[:140]}")
    for e in (at.get("errors_detail") or [])[:5]:
        print(f"    ✗ {e.get('source_id')}: {e.get('error', '')[:140]}")
    print()
    print(rep.get("note", ""))
    return 0


if __name__ == "__main__":
    sys.exit(main())
