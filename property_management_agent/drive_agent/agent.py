"""
Drive Agent — finds files in a dedicated Google Drive folder and returns
shareable links (webViewLink). Used as a sub-agent via AgentTool from the
root property management agent.

Configuration (.env):
  DRIVE_FOLDER_ID    — Google Drive folder ID (preferred, faster)
                        Get it from the folder's URL:
                          https://drive.google.com/drive/folders/<THIS_IS_THE_ID>
  DRIVE_FOLDER_NAME  — Or specify by name; the agent resolves it to an ID
                        on first use. Slower because of the extra lookup.

If neither is set the tools search across the user's entire Drive — usable
for one-off queries but not what the user asked for.
"""
import json
import logging
import os
from typing import Optional

from google.adk.agents import Agent
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError

from .._auth import get_credentials_or_message

logger = logging.getLogger(__name__)

# Read once at import time. If the user changes .env they'll restart anyway.
DRIVE_FOLDER_ID = os.getenv("DRIVE_FOLDER_ID", "").strip()
DRIVE_FOLDER_NAME = os.getenv("DRIVE_FOLDER_NAME", "").strip()

# Cache for name→id resolution so we don't hit Drive on every tool call
_resolved_folder_id: Optional[str] = None


def _get_drive_service():
    creds, msg = get_credentials_or_message()
    if creds is None:
        return None, {"isError": True, "needsAuth": True, "message": msg}
    try:
        return build("drive", "v3", credentials=creds, cache_discovery=False), None
    except Exception as e:
        return None, {"isError": True, "message": f"Could not build Drive service: {e}"}


def _resolve_dedicated_folder_id(service) -> tuple[Optional[str], Optional[str]]:
    """Resolve the configured DRIVE_FOLDER_ID / DRIVE_FOLDER_NAME to an ID.
    Returns (folder_id, error_message)."""
    global _resolved_folder_id

    if DRIVE_FOLDER_ID:
        return DRIVE_FOLDER_ID, None

    if not DRIVE_FOLDER_NAME:
        return None, None  # neither configured — caller may still search all of Drive

    if _resolved_folder_id:
        return _resolved_folder_id, None

    try:
        # Escape single quotes inside the folder name for the Drive query
        safe = DRIVE_FOLDER_NAME.replace("'", "\\'")
        results = (
            service.files()
            .list(
                q=f"mimeType='application/vnd.google-apps.folder' "
                  f"and name='{safe}' and trashed=false",
                spaces="drive",
                fields="files(id, name)",
                pageSize=10,
            )
            .execute()
        )
        folders = results.get("files", [])
        if not folders:
            return None, (
                f"No folder named '{DRIVE_FOLDER_NAME}' found in your Drive. "
                f"Set DRIVE_FOLDER_ID in .env to the folder ID instead."
            )
        _resolved_folder_id = folders[0]["id"]
        logger.info(
            f"Resolved DRIVE_FOLDER_NAME='{DRIVE_FOLDER_NAME}' to id={_resolved_folder_id}"
        )
        return _resolved_folder_id, None
    except HttpError as e:
        return None, f"Drive API error while resolving folder name: {e}"


# ── Tools ──────────────────────────────────────────────────────────────────────

def find_file_in_folder(
    query: str,
    folder_id: str = "",
    folder_name: str = "",
    limit: int = 10,
) -> str:
    """
    Find files in a Google Drive folder by name (full-text match).

    Args:
        query: Text to match in the file name (Drive 'contains' semantics).
        folder_id: Override the configured DRIVE_FOLDER_ID for this call.
        folder_name: Override the configured DRIVE_FOLDER_NAME for this call.
        limit: Max results to return (default 10, max 50).

    Returns:
        JSON string: {"count": N, "folder_id": "...", "files": [
            {"id": "...", "name": "...", "mimeType": "...",
             "webViewLink": "...", "modifiedTime": "..."},
            ...
        ]}

    The webViewLink in each result IS the shareable link — opening it grants
    whatever access the file already has in Drive (the agent does not change
    permissions; that requires a broader OAuth scope).
    """
    service, err = _get_drive_service()
    if err:
        return json.dumps(err)

    # Determine which folder to search
    if folder_id:
        target_id = folder_id
    elif folder_name:
        # Caller wants a specific named folder for this query only
        safe = folder_name.replace("'", "\\'")
        try:
            folder_results = (
                service.files()
                .list(
                    q=f"mimeType='application/vnd.google-apps.folder' "
                      f"and name='{safe}' and trashed=false",
                    spaces="drive",
                    fields="files(id, name)",
                    pageSize=5,
                )
                .execute()
            )
            folders = folder_results.get("files", [])
            if not folders:
                return json.dumps({
                    "isError": True,
                    "message": f"No folder named '{folder_name}' found.",
                })
            target_id = folders[0]["id"]
        except HttpError as e:
            return json.dumps({"isError": True, "message": f"Drive API error: {e}"})
    else:
        target_id, folder_err = _resolve_dedicated_folder_id(service)
        if folder_err:
            return json.dumps({"isError": True, "message": folder_err})

    # Build the file-search query
    safe_q = (query or "").replace("'", "\\'")
    parts = ["trashed=false"]
    if target_id:
        parts.append(f"'{target_id}' in parents")
    if safe_q:
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
            )
            .execute()
        )
        files = results.get("files", [])
        return json.dumps({
            "count": len(files),
            "folder_id": target_id,
            "files": files,
        })
    except HttpError as e:
        return json.dumps({"isError": True, "message": f"Drive API error: {e}"})


def get_file_link(file_id: str) -> str:
    """
    Return the shareable webViewLink for a specific Drive file ID.

    Args:
        file_id: Google Drive file ID (from find_file_in_folder).

    Returns:
        JSON string: {"id": "...", "name": "...", "webViewLink": "...",
                      "mimeType": "...", "modifiedTime": "..."}.
    """
    service, err = _get_drive_service()
    if err:
        return json.dumps(err)
    try:
        meta = (
            service.files()
            .get(
                fileId=file_id,
                fields="id, name, mimeType, webViewLink, modifiedTime",
            )
            .execute()
        )
        return json.dumps(meta)
    except HttpError as e:
        return json.dumps({"isError": True, "message": f"Drive API error: {e}"})


def list_dedicated_folder(limit: int = 50) -> str:
    """
    List all files in the dedicated folder (configured via DRIVE_FOLDER_ID
    or DRIVE_FOLDER_NAME in .env). Useful when the user just wants to see
    what's there without a specific name to search for.

    Args:
        limit: Max files to return (default 50, max 200).

    Returns:
        JSON string with the same shape as find_file_in_folder.
    """
    service, err = _get_drive_service()
    if err:
        return json.dumps(err)
    target_id, folder_err = _resolve_dedicated_folder_id(service)
    if folder_err:
        return json.dumps({"isError": True, "message": folder_err})
    if not target_id:
        return json.dumps({
            "isError": True,
            "message": (
                "No dedicated folder configured. Set DRIVE_FOLDER_ID or "
                "DRIVE_FOLDER_NAME in .env so I know which folder to list."
            ),
        })
    try:
        results = (
            service.files()
            .list(
                q=f"'{target_id}' in parents and trashed=false",
                spaces="drive",
                fields="files(id, name, mimeType, webViewLink, modifiedTime)",
                pageSize=max(1, min(int(limit) if str(limit).isdigit() else 50, 200)),
                orderBy="modifiedTime desc",
            )
            .execute()
        )
        files = results.get("files", [])
        return json.dumps({
            "count": len(files),
            "folder_id": target_id,
            "files": files,
        })
    except HttpError as e:
        return json.dumps({"isError": True, "message": f"Drive API error: {e}"})


# ── Agent definition ───────────────────────────────────────────────────────────

drive_agent = Agent(
    name="drive_agent",
    model="gemini-2.5-flash",
    description=(
        "Finds files in the configured Google Drive folder and returns "
        "shareable links. Use it whenever the user asks for a document, "
        "spreadsheet, PDF, or any file that lives in their property "
        "management Drive folder."
    ),
    instruction="""
You are the Google Drive sub-agent for a property management system.

You have three tools:

  find_file_in_folder(query, folder_id="", folder_name="", limit=10)
    Search the dedicated Drive folder by file name (partial match).
    Returns matching files including a webViewLink (the shareable URL).

  get_file_link(file_id)
    Return the webViewLink and metadata for a known file id.

  list_dedicated_folder(limit=50)
    List everything in the dedicated folder, newest first. Use this when
    the user asks "what's in the folder" or has no specific filename.

WORKFLOW
- When the user asks for a file by name, call find_file_in_folder with the
  most distinctive word(s) from their request as `query`. Drive searches
  with substring/contains semantics.
- If multiple matches come back, present a short list (name + modified
  time + link) and let the user disambiguate. If only one match, return
  it immediately with its webViewLink.
- Always include the webViewLink in your reply — that IS the shareable link.
- If a tool returns {"needsAuth": true, "message": "..."}, STOP and reply
  with the exact text from "message" so the user can sign in.

RESPONSE FORMAT
For each file returned, format as:
  📄 <name>         (last modified <modifiedTime>)
     <webViewLink>

Don't summarise file contents — you don't have them. Just return the link.
""",
    tools=[find_file_in_folder, get_file_link, list_dedicated_folder],
)
