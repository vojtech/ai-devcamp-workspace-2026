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
import io
import json
import logging
import os
from typing import Optional

from google import genai
from google.adk.agents import Agent
from google.genai import types as genai_types
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError
from googleapiclient.http import MediaIoBaseDownload

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


def download_drive_file_content(file_id: str, max_bytes: int = 5_000_000) -> str:
    """
    Download the raw text content of a Drive file (e.g. a JSON email thread).

    Args:
        file_id: Drive file id (from find_file_in_folder).
        max_bytes: Refuse files larger than this (default 5 MB) — guards
                   against accidentally pulling huge binaries.

    Returns:
        JSON: {"file_id": "...", "name": "...", "mimeType": "...",
               "size_bytes": N, "content": "<decoded text>"}.
        On error: {"isError": true, "message": "..."}.

    Only works for text-like files (JSON, plain text). Google Docs / Sheets /
    binary attachments return an error — use get_file_link for those.
    """
    service, err = _drive_service()
    if err:
        return json.dumps(err)
    try:
        meta = (
            service.files()
            .get(
                fileId=file_id,
                fields="id, name, mimeType, size, webViewLink",
                supportsAllDrives=True,
            )
            .execute()
        )
        size = int(meta.get("size", 0) or 0)
        if size > max_bytes:
            return json.dumps({
                "isError": True,
                "message": f"File '{meta.get('name')}' is {size} bytes (limit {max_bytes}).",
            })
        mime = meta.get("mimeType", "")
        if mime.startswith("application/vnd.google-apps."):
            return json.dumps({
                "isError": True,
                "message": (
                    f"'{meta.get('name')}' is a Google-native doc ({mime}). "
                    "Use get_file_link instead, or export it to text/JSON first."
                ),
            })

        req = service.files().get_media(fileId=file_id, supportsAllDrives=True)
        buf = io.BytesIO()
        downloader = MediaIoBaseDownload(buf, req, chunksize=1024 * 1024)
        done = False
        while not done:
            _, done = downloader.next_chunk()
        raw = buf.getvalue()
        try:
            content = raw.decode("utf-8")
        except UnicodeDecodeError:
            content = raw.decode("utf-8", errors="replace")
        return json.dumps({
            "file_id": meta["id"],
            "name": meta.get("name", ""),
            "mimeType": mime,
            "size_bytes": len(raw),
            "content": content,
        })
    except HttpError as e:
        return json.dumps({"isError": True, "message": f"Drive API error: {e}"})


# ── Multimodal file reader (Gemini 2.5 Flash) ─────────────────────────────────
# Reads PDFs and images natively through Gemini multimodal — no separate OCR
# pipeline, no Vision API, no Document AI. One Google API key handles every
# file type. Tier 1 Postpay → inputs are not used for training.

# Cap inline file size sent to Gemini. The model accepts ~20 MB of inline
# content per request; we stay conservative for predictable latency.
_GEMINI_MAX_INLINE_BYTES = 10 * 1024 * 1024  # 10 MB

# Lazy client — only built when first multimodal call is made.
_gemini_client: Optional["genai.Client"] = None


def _get_gemini_client() -> Optional["genai.Client"]:
    global _gemini_client
    if _gemini_client is not None:
        return _gemini_client
    api_key = os.getenv("GOOGLE_API_KEY", "").strip()
    if not api_key:
        return None
    _gemini_client = genai.Client(api_key=api_key)
    return _gemini_client


def _download_drive_bytes(
    service, file_id: str, max_bytes: int
) -> tuple[Optional[bytes], Optional[str]]:
    """Download a Drive file as raw bytes (no UTF-8 decode). Returns
    (bytes, None) on success or (None, error_message) on failure."""
    try:
        req = service.files().get_media(fileId=file_id, supportsAllDrives=True)
        buf = io.BytesIO()
        downloader = MediaIoBaseDownload(buf, req, chunksize=1024 * 1024)
        done = False
        while not done:
            _, done = downloader.next_chunk()
            if buf.tell() > max_bytes:
                return None, f"File exceeds the {max_bytes}-byte cap."
        return buf.getvalue(), None
    except HttpError as e:
        return None, f"Drive API error: {e}"


def _gemini_extract(file_bytes: bytes, mime_type: str, instruction: str) -> str:
    """Send file bytes + an instruction to Gemini 2.5 Flash. Returns the
    model's plain text response (empty string on hard failure)."""
    client = _get_gemini_client()
    if client is None:
        raise RuntimeError(
            "GOOGLE_API_KEY is not set; cannot call Gemini multimodal."
        )
    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=[
            genai_types.Part.from_bytes(data=file_bytes, mime_type=mime_type),
            instruction,
        ],
    )
    return (response.text or "").strip()


def read_drive_file(file_id: str, max_bytes: int = _GEMINI_MAX_INLINE_BYTES) -> str:
    """
    Read a Drive file's content and return it as text — dispatched by mimeType:

        text/*, application/json, application/xml
            → returned as-is (UTF-8 decoded)
        application/pdf
            → Gemini multimodal extracts text (handles scanned PDFs too)
        image/* (PNG, JPEG, WEBP, HEIC, etc.)
            → Gemini multimodal transcribes any text AND describes the scene
        application/vnd.google-apps.document (Google Docs)
            → Drive export to text/plain
        application/vnd.google-apps.spreadsheet (Google Sheets)
            → Drive export to text/csv
        anything else
            → JSON error with content_type="unsupported"

    Args:
        file_id:   Drive file id (from find_file_in_folder / list_folder).
        max_bytes: Refuse files larger than this (default 10 MB).

    Returns:
        JSON: {
            "file_id":       "...",
            "name":          "invoice-Nov-2024.pdf",
            "mime_type":     "application/pdf",
            "web_view_link": "https://drive.google.com/file/d/.../view",
            "size_bytes":    N,
            "content_type":  "text" | "pdf_extracted" | "image_described" |
                             "google_doc_exported" | "google_sheet_exported" |
                             "unsupported",
            "content":       "<the extracted/described text>"
        }
    """
    service, err = _drive_service()
    if err:
        return json.dumps(err)

    # Fetch metadata first so we can dispatch by mimeType + report size.
    try:
        meta = (
            service.files()
            .get(
                fileId=file_id,
                fields="id, name, mimeType, size, webViewLink",
                supportsAllDrives=True,
            )
            .execute()
        )
    except HttpError as e:
        return json.dumps({"isError": True, "message": f"Drive API error: {e}"})

    mime = meta.get("mimeType", "")
    name = meta.get("name", "")
    size = int(meta.get("size", 0) or 0)
    base = {
        "file_id": meta["id"],
        "name": name,
        "mime_type": mime,
        "web_view_link": meta.get("webViewLink", ""),
        "size_bytes": size,
    }

    # Google-native types report size=0 (they're stored as Google objects),
    # so we only size-check after we know we're hitting binary content below.

    # ── 1) Plain-text-like files ───────────────────────────────────────
    if mime.startswith("text/") or mime in ("application/json", "application/xml"):
        if size > max_bytes:
            return json.dumps({
                **base, "isError": True,
                "message": f"File too large for inline read ({size} > {max_bytes}).",
            })
        raw, dl_err = _download_drive_bytes(service, file_id, max_bytes)
        if dl_err:
            return json.dumps({**base, "isError": True, "message": dl_err})
        return json.dumps({
            **base,
            "content_type": "text",
            "content": raw.decode("utf-8", errors="replace"),
        })

    # ── 2) Google Docs → export to text/plain ───────────────────────────
    if mime == "application/vnd.google-apps.document":
        return _export_google_native(service, base, "text/plain", "google_doc_exported")

    # ── 3) Google Sheets → export to CSV ────────────────────────────────
    if mime == "application/vnd.google-apps.spreadsheet":
        return _export_google_native(service, base, "text/csv", "google_sheet_exported")

    # ── 4) PDF → Gemini multimodal extraction ───────────────────────────
    if mime == "application/pdf":
        if size > max_bytes:
            return json.dumps({
                **base, "isError": True,
                "message": f"PDF too large ({size} > {max_bytes}) for inline Gemini read.",
            })
        raw, dl_err = _download_drive_bytes(service, file_id, max_bytes)
        if dl_err:
            return json.dumps({**base, "isError": True, "message": dl_err})
        try:
            text = _gemini_extract(
                raw,
                "application/pdf",
                "Extract all text content from this PDF. Preserve paragraph "
                "breaks and any tabular structure. If the PDF is a scanned "
                "document, OCR it. Return ONLY the extracted text — no "
                "commentary, no preamble, no markdown fences.",
            )
            return json.dumps({**base, "content_type": "pdf_extracted", "content": text})
        except Exception as e:
            return json.dumps({
                **base, "isError": True,
                "message": f"Gemini PDF extraction failed: {type(e).__name__}: {e}",
            })

    # ── 5) Image → Gemini multimodal description + OCR ──────────────────
    if mime.startswith("image/"):
        if size > max_bytes:
            return json.dumps({
                **base, "isError": True,
                "message": f"Image too large ({size} > {max_bytes}).",
            })
        raw, dl_err = _download_drive_bytes(service, file_id, max_bytes)
        if dl_err:
            return json.dumps({**base, "isError": True, "message": dl_err})
        try:
            text = _gemini_extract(
                raw,
                mime,
                "You are inspecting a photo/screenshot relevant to property "
                "management. Do BOTH of the following in your response:\n"
                "1. Transcribe ALL text visible in the image verbatim.\n"
                "2. Describe what the image shows factually — e.g. 'damp "
                "patch on bedroom ceiling, approx 30cm across', 'invoice "
                "scan from contractor X dated …', 'meter reading 12345'.\n"
                "Be specific. No preamble.",
            )
            return json.dumps({**base, "content_type": "image_described", "content": text})
        except Exception as e:
            return json.dumps({
                **base, "isError": True,
                "message": f"Gemini image read failed: {type(e).__name__}: {e}",
            })

    # ── 6) Unsupported ──────────────────────────────────────────────────
    return json.dumps({
        **base,
        "content_type": "unsupported",
        "message": (
            f"MIME type '{mime}' is not currently supported by read_drive_file. "
            "Use get_file_link to return the Drive URL instead."
        ),
    })


def _export_google_native(service, base: dict, export_mime: str, content_type: str) -> str:
    """Helper: export a Google-native file (Doc / Sheet) to plain text."""
    try:
        req = service.files().export_media(fileId=base["file_id"], mimeType=export_mime)
        buf = io.BytesIO()
        downloader = MediaIoBaseDownload(buf, req, chunksize=1024 * 1024)
        done = False
        while not done:
            _, done = downloader.next_chunk()
        return json.dumps({
            **base,
            "content_type": content_type,
            "content": buf.getvalue().decode("utf-8", errors="replace"),
            "size_bytes": buf.tell(),  # exported size, not source size
        })
    except HttpError as e:
        return json.dumps({**base, "isError": True, "message": f"Drive export failed: {e}"})


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

  download_drive_file_content(file_id, max_bytes=5000000)
    Download the raw text content of a file (JSON, plain text). Used by
    the archive-ingestion workflow to read email-thread JSONs into the DB.
    NOT for binary attachments or Google-native docs — use read_drive_file.

  read_drive_file(file_id, max_bytes=10485760)
    READ THE ACTUAL CONTENT of a file regardless of type. Dispatches by
    mimeType:
      - text / JSON / XML   → returns raw text
      - PDF                 → Gemini multimodal extracts the text (handles
                              scanned PDFs via OCR)
      - image (PNG / JPEG / WEBP / HEIC / …) → Gemini transcribes any
                              visible text AND describes what the image shows
      - Google Doc          → exported to plain text
      - Google Sheet        → exported to CSV
    Use this whenever the user asks "what does this file say", "summarise
    this PDF", "what does the photo show", "extract the invoice details",
    "transcribe the meter reading", etc.

WORKFLOW

- The user has TWO labeled folders configured by default:
    * "emails"      — structured JSON files (one per email thread)
    * "attachments" — files extracted from emails (PDFs, images, etc.)
  Always pick the right `folder_label` based on the request:
    "find the lease attachment"    → folder_label="attachments"
    "find email thread about X"    → folder_label="emails"
    "show me the boiler invoice"   → folder_label="attachments"
    "what files are there?"        → list_folder() (all folders)

- For LISTING / FINDING a file → return name + webViewLink as before.
- For READING / SUMMARISING / OCRing → call read_drive_file(file_id)
  AFTER you've identified the right file. Then synthesise an answer
  from its returned `content` field. Always include the Drive link so
  the user can verify the original.

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
        download_drive_file_content,
        read_drive_file,
    ],
)
