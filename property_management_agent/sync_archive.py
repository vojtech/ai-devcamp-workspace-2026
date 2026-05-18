"""
Standalone CLI entry-point for syncing the email archive from Drive.

Designed to be run from cron / launchd / GitHub Actions / etc. without
involving the live ADK agent process. Reads the same .env file the agent
uses, then calls ingest_email_archive_from_drive() and prints a summary.

Usage:
    python3.11 -m property_management_agent.sync_archive
    python3.11 -m property_management_agent.sync_archive --limit 50
    python3.11 -m property_management_agent.sync_archive --force   # full re-embed

Example crontab line (sync every hour):
    0 * * * * cd /Users/you/Developer/DevCamp && \\
        /usr/local/bin/python3.11 -m property_management_agent.sync_archive \\
        >> /tmp/property_sync.log 2>&1

Exit code:
    0  — sync completed (may include skipped/unchanged)
    1  — at least one file failed to ingest
    2  — fatal error (e.g. .env missing, Drive auth misconfigured)
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path


def _bootstrap() -> None:
    """Make the package importable + load .env, regardless of cwd."""
    here = Path(__file__).resolve().parent
    repo_root = here.parent
    # Insert the parent so `from property_management_agent...` works even
    # when invoked from cron with cwd=/$HOME.
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))

    try:
        from dotenv import load_dotenv  # type: ignore
    except ImportError:
        print("python-dotenv is not installed — env vars must already be set.",
              file=sys.stderr)
        return
    env_path = here / ".env"
    if env_path.exists():
        load_dotenv(env_path)


def main() -> int:
    _bootstrap()

    parser = argparse.ArgumentParser(
        description="Sync the email archive from the Drive 'emails' folder.",
    )
    parser.add_argument(
        "--limit", type=int, default=0,
        help="Process at most N files this run (0 = no limit; default).",
    )
    parser.add_argument(
        "--force", action="store_true",
        help="Re-embed ALL files regardless of modifiedTime. Use after model upgrades.",
    )
    parser.add_argument(
        "--json", action="store_true",
        help="Emit the raw JSON report instead of the human summary.",
    )
    args = parser.parse_args()

    try:
        from property_management_agent.agent import ingest_email_archive_from_drive
    except Exception as e:
        print(f"FATAL: could not import the agent — {type(e).__name__}: {e}",
              file=sys.stderr)
        return 2

    try:
        raw = ingest_email_archive_from_drive(limit=args.limit, force_reembed=args.force)
        report = json.loads(raw)
    except Exception as e:
        print(f"FATAL: sync failed — {type(e).__name__}: {e}", file=sys.stderr)
        return 2

    if args.json:
        print(json.dumps(report, indent=2))
        return 1 if report.get("errors", 0) > 0 else 0

    # Human-readable summary
    print(f"Email archive sync — {report.get('files_processed', 0)} files seen in Drive")
    print(f"  new:               {report.get('new', 0)}")
    print(f"  updated:           {report.get('updated', 0)}")
    print(f"  skipped_unchanged: {report.get('skipped_unchanged', 0)}")
    print(f"  errors:            {report.get('errors', 0)}")
    for e in report.get("errors_detail", [])[:10]:
        print(f"    - {e.get('file', '?')}: {e.get('error', '')[:200]}")
    arch = report.get("now_in_archive") or {}
    print(f"  archive total:     {arch.get('total_threads', '?')} "
          f"({arch.get('embedded_threads', '?')} embedded)")

    return 1 if report.get("errors", 0) > 0 else 0


if __name__ == "__main__":
    sys.exit(main())
