"""
Drive Agent — finds files in folders explicitly shared with a Google Cloud
service account, and returns shareable links (webViewLink).

Auth model: SERVICE ACCOUNT (NOT user OAuth).
The service account can only see folders that have been explicitly shared
with its email address (`<sa-name>@<project>.iam.gserviceaccount.com`).
This means the agent has folder-scoped access — not the user's entire
Drive — even though the OAuth-level scope is `drive.readonly`.

Configuration (.env):
  DRIVE_SERVICE_ACCOUNT_JSON  — Path to the SA JSON key.
                                Defaults to property_management_agent/service-account.json

  DRIVE_FOLDERS               — Comma-separated list of `label:folder_id`
                                pairs (or bare ids if you don't need labels).
                                Example:
                                    DRIVE_FOLDERS=emails:12iOR49...,attachments:1qtWt5c...

  Labels are how the LLM refers to a specific folder ("search the emails
  folder", "list attachments"). If the user doesn't set labels, every
  folder is searchable but only by id.
"""
import json
import logging
import os
from typing import Optional

from google.adk.agents import Agent
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError

from .._auth import get_drive_credentials

logger = logging.getLogger(__name__)


# ── Folder configuration ──────────────────────────────────────────────────────
# Parse DRIVE_FOLDERS once at import. Falls back to legacy DRIVE_FOLDER_ID /
# DRIVE_FOLDER_NAME if DRIVE_FOLDERS is empty.

def _parse_folder_config() -> dict[str, str]:
    """Returns {label: folder_id}. Bare ids get auto-labelled folder_N."""
    raw = os.getenv("DRIVE_FOLDERS", "").strip()
    out: dict[str, str] = {}
    if not raw:
        # Back-compat: honour the old single-folder env vars
        legacy = os.getenv("DRIVE_FOLDER_ID", "").strip()
        if legacy:
            out["default"] = legacy
        return out
    for i, part in enumerate(raw.split(",")):
        part = part.strip()
        if not part:
            continue
        if ":" in part:
            label, _, folder_id = part.partition(":")
            label = label.strip().lower()
            folder_id = folder_id.strip()
            if label and folder_id:
                out[label] = folder_id
        else:
            out[f"folder_{i+1}"] = part
    return out


FOLDERS: dict[str, str] = _parse_folder_config()


def _allowed_folder_ids() -> list[str]:
    return list(FOLDERS.values())


def _resolve_label(folder_label: str = "") -> tuple[list[str], Optional[str]]:
    """Return ([folder_ids], error). If folder_label is set, restrict to
    that label only. Otherwise return ALL configured folders."""
    if folder_label:
        key = folder_label.strip().lower()
        if key not in FOLDERS:
            return [], (
                f"Unknown folder label '{folder_label}'. "
                f"Configured labels: {', '.join(FOLDERS) or '(none)'}."
            )
        return [FOLDERS[key]], None
    if not FOLDERS:
        return [], (
            "No Drive folders configured. Set DRIVE_FOLDERS in .env to a "
            "comma-separated list of `label:folder_id` pairs, then restart."
        )
    return _allowed_folder_ids(), None


def _drive_service():
    creds, msg = get_drive_credentials()
    if creds is None:
        return None, {"isError": True, "needsAuth": True, "message": msg}
    try:
        return build("drive", "v3", credentials=creds, cache_discovery=False), None
    except Exception as e:
        return None, {"isError": True, "message": f"Could not build Drive service: {e}"}


def _in_parents_clause(folder_ids: list[str]) -> str:
    """Drive query: `'a' in parents or 'b' in parents`."""
    return "(" + " or ".join(f"'{fid}' in parents" for fid in folder_ids) + ")"


# ── Tools ──────────────────────────────────────────────────────────────────────

def list_configured_folders() -> str:
    """List the folder labels the agent has access to, with their IDs.
    Use this when the user asks 'what folders can you see?'"""
    return json.dumps({
        "count": len(FOLDERS),
        "folders": [{"label": k, "folder_id": v} for k, v in FOLDERS.items()],
    })


def find_file_in_folder(
    query: str,
    folder_label: str = "",
    limit: int = 10,
) -> str:
    """
    Find files in the configured Drive folders by name (partial match).

    Args:
        query: Text to match in the file name (Drive 'contains' semantics).
        folder_label: Restrict to ONE labeled folder (e.g. "emails",
                      "attachments"). If empty, searches ALL configured folders.
        limit: Max results to return (default 10, max 50).

    Returns:
        JSON: {"count": N, "folders_searched": [...], "files": [
            {id, name, mimeType, webViewLink, modifiedTime, parents}, ...
        ]}

    webViewLink IS the shareable URL — anyone the file is shared with can
    open it. The service account does not modify share permissions.
    """
    service, err = _drive_service()
    if err:
        return json.dumps(err)

    folder_ids, folder_err = _resolve_label(folder_label)
    if folder_err:
        return json.dumps({"isError": True, "message": folder_err})

    parts = ["trashed=false", _in_parents_clause(folder_ids)]
    if query:
        safe_q = query.replace("'", "\\'")
        parts.append(f"name contains '{safe_q}'")
    drive_q = " and ".join(parts)

    try:
        results = (
            service.files()
            .list(
                q=drive_q,
                spaces="drive",
                fields="files(id, name, mimeType, webViewLink, modifiedTime, parents)",
                pageSize=max(1, min(int(limit) if str(limit).isdigit() else 10, 50)),
                orderBy="modifiedTime desc",
                # Service account doesn't own the files; we only need 'me'-style
                # listing when the SA *is* the owner. The next two params ensure
                # we list shared content correctly:
                supportsAllDrives=True,
                includeItemsFromAllDrives=True,
            )
            .execute()
        )
        return json.dumps({
            "count": len(results.get("files", [])),
            "folders_searched": folder_ids,
            "files": results.get("files", []),
        })
    except HttpError as e:
        return json.dumps({"isError": True, "message": f"Drive API error: {e}"})


def list_folder(folder_label: str = "", limit: int = 50) -> str:
    """
    List everything in a labeled folder (or all configured folders), newest
    first. Useful when the user has no specific filename in mind.

    Args:
        folder_label: One configured label, or "" for all folders.
        limit: Max files to return (default 50, max 200).
    """
    service, err = _drive_service()
    if err:
        return json.dumps(err)
    folder_ids, folder_err = _resolve_label(folder_label)
    if folder_err:
        return json.dumps({"isError": True, "message": folder_err})

    drive_q = f"trashed=false and {_in_parents_clause(folder_ids)}"
    try:
        results = (
            service.files()
            .list(
                q=drive_q,
                spaces="drive",
                fields="files(id, name, mimeType, webViewLink, modifiedTime, parents)",
                pageSize=max(1, min(int(limit) if str(limit).isdigit() else 50, 200)),
                orderBy="modifiedTime desc",
                supportsAllDrives=True,
                includeItemsFromAllDrives=True,
            )
            .execute()
        )
        return json.dumps({
            "count": len(results.get("files", [])),
            "folders_listed": folder_ids,
            "files": results.get("files", []),
        })
    except HttpError as e:
        return json.dumps({"isError": True, "message": f"Drive API error: {e}"})


def get_file_link(file_id: str) -> str:
    """
    Return metadata + webViewLink for a specific file id.

    Args:
        file_id: Drive file id (from find_file_in_folder or list_folder).
    """
    service, err = _drive_service()
    if err:
        return json.dumps(err)
    try:
        meta = (
            service.files()
            .get(
                fileId=file_id,
                fields="id, name, mimeType, webViewLink, modifiedTime, parents",
                supportsAllDrives=True,
            )
            .execute()
        )
        return json.dumps(meta)
    except HttpError as e:
        return json.dumps({"isError": True, "message": f"Drive API error: {e}"})


# ── Agent definition ───────────────────────────────────────────────────────────

drive_agent = Agent(
    name="drive_agent",
    model="gemini-2.5-flash",
    description=(
        "Finds files in the user's shared Google Drive folders and returns "
        "shareable links. Access is restricted to specific labeled folders "
        "(e.g. 'emails', 'attachments') — the agent cannot see anything "
        "else in the user's Drive."
    ),
    instruction="""
You are the Google Drive sub-agent for a property management system.

You have folder-scoped access: only specific folders that the user has
EXPLICITLY shared with the service account are visible to you. The folders
are referred to by short labels (e.g. "emails", "attachments").

TOOLS

  list_configured_folders()
    Returns the labels and IDs of the folders you can see. Use this if the
    user asks "what folders do you have?" or you're not sure which label
    to search.

  find_file_in_folder(query, folder_label="", limit=10)
    Search by file name (partial match). Pass `folder_label` to restrict
    to one folder (e.g. "emails") — strongly preferred when the user's
    question implies a specific folder. Otherwise searches ALL folders.

  list_folder(folder_label="", limit=50)
    List the contents of a labeled folder, newest first. Use this when
    the user doesn't have a specific filename ("what's in attachments?").

  get_file_link(file_id)
    Fetch metadata + webViewLink for a known file id.

WORKFLOW

- The user has TWO labeled folders configured by default:
    * "emails"      — structured JSON files (one per email thread)
    * "attachments" — files extracted from emails (PDFs, images, etc.)
  Always pick the right `folder_label` based on the request:
    "find the lease attachment"    → folder_label="attachments"
    "find email thread about X"    → folder_label="emails"
    "show me the boiler invoice"   → folder_label="attachments"
    "what files are there?"        → list_folder() (all folders)

- For each match, present:
    📄 <name>            (modified <modifiedTime>)
       <webViewLink>

  Don't summarise file contents — you only have metadata.

- If a tool returns {"needsAuth": true, "message": "..."}, STOP and reply
  with the exact message from the tool so the user can resolve setup
  (e.g. drop the service-account.json file into place).

- If the user asks for something that clearly isn't in any configured
  folder, say so — don't pretend a result exists.
""",
    tools=[
        list_configured_folders,
        find_file_in_folder,
        list_folder,
        get_file_link,
    ],
)
