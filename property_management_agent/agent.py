"""
Property Management Email Analyzer — root agent.

Architecture:
  root_agent (property_email_analyzer)
  │  Gmail tools  → Gmail REST API via googleapiclient
  │    • search_threads(query, page_size)
  │    • get_thread(thread_id)
  │  (Same OAuth flow as the Gmail MCP path; bypasses the gmailmcp.googleapis.com
  │   developer-preview gate by talking to the standard Gmail API directly.)
  │  Python extraction tools (deterministic regex helpers)
  │    • extract_employees_from_text
  │    • extract_managers_from_text
  │    • extract_contacts_from_text
  │    • extract_tasks_from_text
  │    • extract_meetings_from_text
  │
  ├─ AgentTool → classifier_agent  (LLM that tags each email into JSON)
  ├─ AgentTool → drive_agent       (finds files in a dedicated Drive folder)
  └─ AgentTool → database_agent    (manages SQLite via direct Python functions)

OAuth (shared across all sub-agents) lives in _auth.py.
"""
import base64
import json
import logging
import os
import re
import sys

from google.adk.agents import Agent
from google.adk.tools.agent_tool import AgentTool
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError

from ._auth import get_credentials_or_message
from .classifier_agent.agent import classifier_agent
from .database_agent.agent import database_agent
from .drive_agent.agent import drive_agent

logging.basicConfig(stream=sys.stderr, level=logging.INFO)
logger = logging.getLogger(__name__)


# ── Gmail REST API client (via googleapiclient) ───────────────────────────────
# Uses the standard Gmail API (gmail.googleapis.com), not the Gmail MCP server.
# Same OAuth token, no developer-preview gating.

def _get_gmail_service():
    """Build a Gmail API service object using the user's OAuth credentials.
    Returns (service, error_dict_or_None). If error_dict is set, surface it
    to the user instead of using the service."""
    creds, user_message = get_credentials_or_message()
    if creds is None:
        return None, {"isError": True, "needsAuth": True, "message": user_message}
    try:
        return build("gmail", "v1", credentials=creds, cache_discovery=False), None
    except Exception as e:
        return None, {"isError": True, "message": f"Could not build Gmail service: {e}"}


def _decode_body(part: dict) -> str:
    """Recursively walk a Gmail message payload to extract plain-text body."""
    mime = part.get("mimeType", "")
    body_data = part.get("body", {}).get("data")
    if mime == "text/plain" and body_data:
        try:
            return base64.urlsafe_b64decode(body_data).decode("utf-8", errors="replace")
        except Exception:
            return ""
    # Prefer text/plain in subparts; fall back to first text/html stripped.
    text_plain = ""
    text_html = ""
    for sub in part.get("parts", []) or []:
        sub_mime = sub.get("mimeType", "")
        if sub_mime == "text/plain":
            text_plain = text_plain or _decode_body(sub)
        elif sub_mime == "text/html":
            text_html = text_html or _decode_body(sub)
        elif sub_mime.startswith("multipart/"):
            inner = _decode_body(sub)
            if inner:
                text_plain = text_plain or inner
    if text_plain:
        return text_plain
    if text_html and not body_data:
        return re.sub(r"<[^>]+>", " ", text_html)
    if mime == "text/html" and body_data:
        raw = base64.urlsafe_b64decode(body_data).decode("utf-8", errors="replace")
        return re.sub(r"<[^>]+>", " ", raw)
    return ""


def search_threads(query: str = "", page_size: int = 20) -> str:
    """
    Search Gmail threads using the standard Gmail REST API.
    Use Gmail search syntax in the query, e.g. "from:@acme-property.com".

    Args:
        query: Gmail search query string (Gmail syntax).
        page_size: Max threads to return (default 20, max 100).

    Returns:
        JSON string with {"threads": [{"id": "...", "snippet": "..."}, ...]}.
    """
    service, err = _get_gmail_service()
    if err:
        return json.dumps(err)
    try:
        result = (
            service.users()
            .threads()
            .list(userId="me", q=query, maxResults=max(1, min(page_size, 100)))
            .execute()
        )
        threads = result.get("threads", [])
        return json.dumps({
            "count": len(threads),
            "threads": [{"id": t["id"], "snippet": t.get("snippet", "")} for t in threads],
        })
    except HttpError as e:
        return json.dumps({"isError": True, "message": f"Gmail API error: {e}"})


def get_thread(thread_id: str) -> str:
    """
    Fetch the full content of a Gmail thread by its ID, using the Gmail REST API.

    Args:
        thread_id: The Gmail thread ID (returned by search_threads).

    Returns:
        JSON string with {"id": "...", "messages": [{from, to, subject, date, body}, ...]}.
    """
    service, err = _get_gmail_service()
    if err:
        return json.dumps(err)
    try:
        thread = (
            service.users()
            .threads()
            .get(userId="me", id=thread_id, format="full")
            .execute()
        )
        messages_out = []
        for msg in thread.get("messages", []):
            payload = msg.get("payload", {})
            headers = {h["name"]: h["value"] for h in payload.get("headers", [])}
            messages_out.append({
                "id": msg.get("id", ""),
                "from": headers.get("From", ""),
                "to": headers.get("To", ""),
                "cc": headers.get("Cc", ""),
                "subject": headers.get("Subject", ""),
                "date": headers.get("Date", ""),
                "body": _decode_body(payload),
            })
        return json.dumps({"id": thread.get("id", thread_id), "messages": messages_out})
    except HttpError as e:
        return json.dumps({"isError": True, "message": f"Gmail API error: {e}"})


# ── Python extraction helpers (called by the LLM as tools) ────────────────────

EMAIL_RE = re.compile(r"[\w.+-]+@[\w-]+\.[a-z]{2,}", re.IGNORECASE)
PHONE_RE = re.compile(r"(?:07\d{9}|\d{5}\s?\d{6}|\+\d[\d\s\-]{8,14})")
NAME_RE = re.compile(r"\b([A-Z][a-z]+ [A-Z][a-z]+)\b")


def extract_employees_from_text(text: str) -> str:
    """
    Extract employee names, email addresses, and roles from raw email text.
    Looks for email signatures with the pattern: Name / Role / email.
    Returns a JSON array of {name, email, role} objects.

    Args:
        text: Raw email body or thread content as a plain string.
    """
    seen: dict[str, dict] = {}

    # Signature block: Name\nRole\nOrg\nemail | phone
    sig_block = re.compile(
        r"([A-Z][a-z]+ [A-Z][a-z]+)\s*\n"
        r"([^\n]{3,60})\s*\n"
        r"(?:[^\n]{3,60}\s*\n)?"          # optional org line
        r"([^\n]*@[^\n]+)",
        re.MULTILINE,
    )
    for m in sig_block.finditer(text):
        name, role, email_line = m.groups()
        emails = EMAIL_RE.findall(email_line)
        if emails:
            addr = emails[0].lower()
            if addr not in seen:
                seen[addr] = {"name": name.strip(), "email": addr, "role": role.strip()}

    # Fallback: any bare email address
    for addr in EMAIL_RE.findall(text):
        addr = addr.lower()
        if addr not in seen:
            seen[addr] = {"name": "", "email": addr, "role": ""}

    return json.dumps(list(seen.values()))


def extract_managers_from_text(text: str) -> str:
    """
    Identify current and previous property managers mentioned in the text.
    Returns a JSON array of {name, email, status, notes} objects.

    Args:
        text: Raw email body or thread content.
    """
    managers = []
    # NOTE: name group must NOT use re.IGNORECASE — capitalisation is the signal.
    #       Use inline (?i:...) only for keyword portions.
    PROPER_NAME = r"([A-Z][a-z]+ [A-Z][a-z]+)"
    previous_pats = [
        re.compile(r"(?i:previous (?:property )?manager(?:\s+was)?\s*[:\-]?\s*)" + PROPER_NAME),
        re.compile(r"(?i:(?:formerly|previously) managed by\s*)" + PROPER_NAME),
        re.compile(r"(?i:former (?:property )?manager[:\s]+)" + PROPER_NAME),
    ]
    current_pats = [
        re.compile(r"(?i:new (?:property )?manager(?:\s+is)?\s*[:\-]?\s*)" + PROPER_NAME),
        re.compile(r"(?i:handing over\b[^.]*?\bto\s+)" + PROPER_NAME),
        re.compile(r"(?i:your (?:new )?property manager(?:\s+is)?\s*[:\-]?\s*)" + PROPER_NAME),
    ]

    def _nearby_email(text: str, pos: int) -> str:
        snippet = text[max(0, pos - 300): pos + 300]
        emails = EMAIL_RE.findall(snippet)
        return emails[0].lower() if emails else ""

    for pat in previous_pats:
        for m in pat.finditer(text):
            managers.append({
                "name": m.group(1).strip(),
                "email": _nearby_email(text, m.start()),
                "status": "previous",
                "notes": f"Detected: '{m.group(0)[:80]}'",
            })
    for pat in current_pats:
        for m in pat.finditer(text):
            managers.append({
                "name": m.group(1).strip(),
                "email": _nearby_email(text, m.start()),
                "status": "current",
                "notes": f"Detected: '{m.group(0)[:80]}'",
            })

    return json.dumps(managers)


def extract_contacts_from_text(text: str) -> str:
    """
    Extract external key contacts (tenants, landlords, contractors, maintenance)
    from email body text.
    Returns a JSON array of {name, role, email, phone, company, notes} objects.

    Args:
        text: Raw email body or thread content.
    """
    contacts = []
    contact_line = re.compile(
        r"[-–]\s*(Tenant|Landlord|Maintenance|Contractor|Contact|Agent|Supplier)"
        r"[:\s]+([A-Z][a-z]+ [A-Z][a-z]+)",
        re.IGNORECASE,
    )
    for m in contact_line.finditer(text):
        role, name = m.group(1), m.group(2)
        snippet = text[m.start(): m.start() + 250]
        emails = EMAIL_RE.findall(snippet)
        phones = PHONE_RE.findall(snippet)
        # Try to find company name (word before Ltd/Limited/Inc/LLC)
        company_m = re.search(r"([A-Z][A-Za-z\s]+(?:Ltd|Limited|Inc|LLC|LLP))", snippet)
        contacts.append({
            "name": name.strip(),
            "role": role.strip(),
            "email": emails[0] if emails else "",
            "phone": phones[0] if phones else "",
            "company": company_m.group(1).strip() if company_m else "",
            "notes": "",
        })
    return json.dumps(contacts)


def extract_tasks_from_text(text: str, source_email: str = "") -> str:
    """
    Extract action items and tasks from numbered or bulleted lists in email text.
    Returns a JSON array of {title, description, due_date, priority, source_email} objects.

    Args:
        text: Raw email body or thread content.
        source_email: Subject line or identifier of the source email.
    """
    tasks = []
    # Numbered list items
    task_pat = re.compile(
        r"^\s*\d+[.)]\s+(.+?)(?:\s*[-–]\s*(?:due|deadline)[:\s]+([^\n,;(]{3,40}))?(?:\s*\(([^)]+)\))?$",
        re.IGNORECASE | re.MULTILINE,
    )
    due_pat = re.compile(r"\bdue\s+([^\n,;(]{3,40})", re.IGNORECASE)
    priority_pat = re.compile(r"\b(HIGH|URGENT|CRITICAL)\b", re.IGNORECASE)

    for m in task_pat.finditer(text):
        title = m.group(1).strip()
        if len(title) < 5 or len(title) > 200:
            continue
        due = (m.group(2) or "").strip()
        if not due:
            dm = due_pat.search(m.group(0))
            due = dm.group(1).strip() if dm else ""
        priority = "high" if priority_pat.search(m.group(0)) else "normal"
        tasks.append({
            "title": title,
            "description": "",
            "assigned_to": "",
            "due_date": due,
            "priority": priority,
            "source_email": source_email,
        })
    return json.dumps(tasks)


def extract_meetings_from_text(text: str, source_email: str = "") -> str:
    """
    Extract scheduled meetings, calls, or appointments from email text.
    Returns a JSON array of {title, date, time, attendees, location, agenda, source_email}.

    Args:
        text: Raw email body or thread content.
        source_email: Subject line or identifier of the source email.
    """
    MEETING_KEYWORDS = re.compile(
        r"\b(meeting|review|call|sync|standup|workshop|inspection|appointment|walkthrough)\b",
        re.IGNORECASE,
    )
    if not MEETING_KEYWORDS.search(text):
        return json.dumps([])

    date_pat = re.compile(r"\b(\d{1,2}(?:st|nd|rd|th)?\s+\w+\s+\d{4}|\d{4}-\d{2}-\d{2})\b", re.IGNORECASE)
    time_pat = re.compile(r"\b(\d{1,2}:\d{2}\s*(?:AM|PM|am|pm)?)\b")
    location_pat = re.compile(r"(?:Location|Venue|Place|Room)[:\s]+([^\n]+)", re.IGNORECASE)
    attendees_pat = re.compile(r"Attendees?[:\s]+([^\n]+)", re.IGNORECASE)
    agenda_pat = re.compile(r"Agenda[:\s]*((?:.|\n)+?)(?:\n\n|\Z)", re.IGNORECASE)

    # Extract subject from first line or use source_email
    first_line = text.strip().split("\n")[0][:120]
    title = first_line if first_line else source_email

    dates = date_pat.findall(text)
    times = time_pat.findall(text)
    loc_m = location_pat.search(text)
    att_m = attendees_pat.search(text)
    agenda_m = agenda_pat.search(text)

    meeting = {
        "title": source_email or title,
        "date": dates[0] if dates else "",
        "time": times[0] if times else "",
        "attendees": att_m.group(1).strip() if att_m else "",
        "location": loc_m.group(1).strip() if loc_m else "",
        "agenda": agenda_m.group(0).strip()[:500] if agenda_m else "",
        "source_email": source_email,
    }
    return json.dumps([meeting])


# ── Email archive (vector RAG over Drive JSON threads) ───────────────────────
# Single Python function that does the whole ingestion loop. The LLM just
# calls this once to backfill the archive — we don't want it issuing one
# tool call per email for hundreds of files.

from ._embeddings import embed_for_query, embed_for_storage  # noqa: E402
from .database_agent.db import (  # noqa: E402
    save_email_archive_entry,
    search_email_archive as _db_search_email_archive,
    count_email_archive,
)
from .drive_agent.agent import (  # noqa: E402
    download_drive_file_content as _download_drive_file_content,
    list_folder as _drive_list_folder,
)


def _parse_thread_json(raw: str) -> dict:
    """Parse an exported Gmail thread JSON (the format you're storing in
    the Drive 'emails' folder) into the fields the archive needs.

    Schema (input):
      {threadId, subject, messageCount, messages: [
        {messageId, from, to, cc, date, body, attachments?: [
          {filename, mimeType, sizeBytes}
        ]}
      ]}

    Returns:
      {thread_id, subject, snippet, body_text, message_count, participants}
      — empty strings / 0 where fields are missing. body_text is the
      concatenation of every message body, prefixed with a one-line header
      so the embedding can pick up dates and senders as semantic signal.
    """
    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        return {
            "thread_id": "", "subject": "", "snippet": raw[:200].strip(),
            "body_text": raw, "message_count": 1, "participants": "",
        }
    if not isinstance(data, dict):
        return {
            "thread_id": "", "subject": "", "snippet": "",
            "body_text": raw, "message_count": 1, "participants": "",
        }

    thread_id = str(data.get("threadId") or "").strip()
    subject = str(data.get("subject") or "").strip()
    msg_count_field = data.get("messageCount")
    messages = data.get("messages") if isinstance(data.get("messages"), list) else []

    participants: set[str] = set()
    blocks: list[str] = []
    for m in messages:
        if not isinstance(m, dict):
            continue
        sender = str(m.get("from") or "").strip()
        date = str(m.get("date") or "").strip()
        body = str(m.get("body") or "").strip()
        if sender:
            participants.add(sender)
        # Prefix each message with a small header so date/sender become part
        # of the embedded text (otherwise "what did X say last March" can't
        # match without metadata in the body).
        header_bits = [b for b in (date, f"From: {sender}" if sender else "") if b]
        header = " | ".join(header_bits)
        if header and body:
            blocks.append(f"[{header}]\n{body}")
        elif body:
            blocks.append(body)

    body_text = "\n\n---\n\n".join(blocks)
    snippet = (body_text.replace("\n", " ").strip())[:200]

    return {
        "thread_id": thread_id,
        "subject": subject,
        "snippet": snippet,
        "body_text": body_text,
        "message_count": int(msg_count_field) if isinstance(msg_count_field, int) else len(messages),
        "participants": ", ".join(sorted(participants)),
    }


def ingest_email_archive_from_drive(limit: int = 0, force_reembed: bool = False) -> str:
    """
    Incrementally sync the email archive from the Drive 'emails' folder.

    On every call:
      - lists the Drive folder (one API call),
      - looks up each file by its Drive file_id in the local DB,
      - decides whether to (a) skip unchanged files, (b) ingest brand-new
        files, or (c) re-embed files whose Drive modifiedTime is newer than
        what's stored locally (i.e. a new reply was added to an existing
        thread and the export pipeline re-wrote the JSON).

    Args:
        limit: Max files to process this run (0 = no limit). Useful for
               smoke tests; on a steady-state schedule, leave at 0.
        force_reembed: If True, re-embed and overwrite ALL files regardless
                       of modifiedTime. Use only when the embedding model
                       changes or you want a clean rebuild.

    Returns:
        JSON summary: {
            "files_processed": N,
            "new":              N,   # files not previously in the archive
            "updated":          N,   # files whose Drive mtime was newer
            "skipped_unchanged":N,   # files already up-to-date in the DB
            "errors":           N,
            "errors_detail":    [...],
            "now_in_archive":   {total_threads, embedded_threads, ...}
        }
    """
    listing_raw = _drive_list_folder("emails", limit=500)
    listing = json.loads(listing_raw)
    if listing.get("isError"):
        return json.dumps(listing)
    files = [f for f in listing.get("files", [])
             if f.get("name", "").endswith(".json")]
    if limit and limit > 0:
        files = files[:limit]

    # Build a (drive_file_id -> drive_modified_time) map of what's stored,
    # so we can decide per-file: skip / ingest-new / update.
    # Indexing by drive_file_id (Drive's immutable id) is more reliable than
    # thread_id because Drive may re-export under a new filename/path while
    # keeping the same file_id, or change the file_id when the export tool
    # re-creates the file.
    seen: dict[str, str] = {}
    from .database_agent.db import get_conn as _db_conn
    with _db_conn() as c:
        for r in c.execute(
            "SELECT drive_file_id, drive_modified_time FROM email_archive"
        ).fetchall():
            seen[r["drive_file_id"]] = r["drive_modified_time"] or ""

    new_count = 0
    updated_count = 0
    skipped_unchanged = 0
    errors: list[dict] = []

    for f in files:
        file_id = f["id"]
        name = f.get("name", "")
        drive_mtime = f.get("modifiedTime", "") or ""

        stored_mtime = seen.get(file_id)
        is_new = stored_mtime is None
        # ISO 8601 timestamps with Z suffix sort lexicographically — safe
        # to compare with `>=`. Empty stored mtime is treated as stale.
        if not force_reembed and not is_new:
            if stored_mtime and drive_mtime and stored_mtime >= drive_mtime:
                skipped_unchanged += 1
                continue

        try:
            dl_raw = _download_drive_file_content(file_id)
            dl = json.loads(dl_raw)
            if dl.get("isError"):
                errors.append({"file": name, "error": dl.get("message", "download failed")})
                continue
            parsed = _parse_thread_json(dl.get("content", ""))
            thread_id = parsed["thread_id"] or file_id
            if not parsed["body_text"].strip():
                errors.append({"file": name, "error": "empty body after parse"})
                continue
            embedding = embed_for_storage(parsed["body_text"])
            save_result_raw = save_email_archive_entry(
                thread_id=thread_id,
                drive_file_id=file_id,
                file_name=name,
                embedding=embedding,
                subject=parsed["subject"],
                snippet=parsed["snippet"],
                body_text=parsed["body_text"],
                participants=parsed["participants"],
                web_view_link=f.get("webViewLink", ""),
                message_count=parsed["message_count"],
                drive_modified_time=drive_mtime,
            )
            save_result = json.loads(save_result_raw)
            if save_result.get("isError"):
                errors.append({"file": name, "error": save_result.get("message", "save failed")})
                continue
            if is_new:
                new_count += 1
            else:
                updated_count += 1
        except Exception as e:
            errors.append({"file": name, "error": f"{type(e).__name__}: {e}"})

    return json.dumps({
        "files_processed": len(files),
        "new": new_count,
        "updated": updated_count,
        "skipped_unchanged": skipped_unchanged,
        "errors": len(errors),
        "errors_detail": errors[:10],
        "now_in_archive": json.loads(count_email_archive()),
    })


def search_email_archive(
    query: str,
    category: str = "",
    urgency: str = "",
    requires_action: str = "",
    limit: int = 10,
) -> str:
    """
    Semantic search over the ingested email archive. Embeds the query and
    runs KNN against the email_archive_vec table, optionally filtered by
    classification (category / urgency / requires_action).

    Args:
        query: Natural-language question or keyword phrase.
        category: One of maintenance/billing/leasing/legal/handover/complaint/
                  emergency/administrative/communication/other. "" = no filter.
        urgency: low/normal/high/urgent. "" = no filter.
        requires_action: "true" / "false" / "" (no filter).
        limit: Max results (default 10, max 50).

    Returns:
        JSON: {"count": N, "results": [{thread_id, subject, snippet,
            participants, web_view_link, distance, category, urgency,
            requires_action}, ...]} ordered by similarity (lowest distance first).
    """
    if not query or not query.strip():
        return json.dumps({"isError": True, "message": "query is empty"})
    try:
        qv = embed_for_query(query)
    except Exception as e:
        return json.dumps({
            "isError": True,
            "message": f"Failed to embed query (check GOOGLE_API_KEY): {e}",
        })
    return _db_search_email_archive(
        query_embedding=qv,
        limit=limit,
        category=category,
        urgency=urgency,
        requires_action=requires_action,
    )


def search_archive_vertex(query: str, limit: int = 10) -> str:
    """
    Alternative archive search backed by Vertex AI Search (managed RAG).
    Same shape of result as search_email_archive but the chunking, embedding,
    and reranking are all done by Google. Used for accuracy comparisons.

    Requires the one-time GCP setup (Discovery Engine API + IAM role) — if
    that's not done, this tool returns a clear "needsSetup" message with the
    exact commands to run.
    """
    from . import _vertex_search as _v
    return json.dumps(_v.search(query, limit=limit))


def vertex_index_status() -> str:
    """Probe Vertex AI Search: is it set up, reachable, and pointing at the
    right project? Returns availability + setup instructions on failure."""
    from . import _vertex_search as _v
    return json.dumps(_v.get_status())


def vertex_index_all(limit: int = 0) -> str:
    """Push every email_archive thread and every attachment_extractions row
    into the Vertex AI Search data store. Idempotent. Newly-pushed docs
    take 5–30 min to become searchable (Google's async indexing)."""
    from . import _vertex_search as _v
    return json.dumps(_v.index_everything(limit=limit))


# ── Optional: run a sync on `adk web` startup ─────────────────────────────────
# Controlled by AUTO_SYNC_ARCHIVE_ON_STARTUP. Disabled by default — set it to
# "true" / "1" / "yes" in .env to enable. The sync runs in a daemon thread so
# it doesn't block agent startup; errors are logged but do not crash the agent.

def _maybe_run_startup_sync() -> None:
    raw = os.getenv("AUTO_SYNC_ARCHIVE_ON_STARTUP", "").strip().lower()
    if raw not in ("1", "true", "yes", "on"):
        return

    import threading

    def _run() -> None:
        try:
            logger.info("Auto-sync: starting email archive sync from Drive...")
            result = json.loads(ingest_email_archive_from_drive())
            logger.info(
                "Auto-sync complete: new=%d updated=%d skipped=%d errors=%d",
                result.get("new", 0),
                result.get("updated", 0),
                result.get("skipped_unchanged", 0),
                result.get("errors", 0),
            )
            if result.get("errors", 0) > 0:
                for e in result.get("errors_detail", [])[:5]:
                    logger.warning("Auto-sync error: %s — %s",
                                    e.get("file", "?"), e.get("error", "")[:200])
        except Exception as e:  # noqa: BLE001
            logger.error("Auto-sync failed: %s: %s", type(e).__name__, e)

    threading.Thread(target=_run, daemon=True, name="archive_auto_sync").start()


_maybe_run_startup_sync()


# ── Root agent ─────────────────────────────────────────────────────────────────

db_tool = AgentTool(agent=database_agent)
classifier_tool = AgentTool(agent=classifier_agent)
drive_tool = AgentTool(agent=drive_agent)

root_agent = Agent(
    name="property_email_analyzer",
    model="gemini-2.5-flash",
    description=(
        "Email intelligence agent. Answers freeform questions about your "
        "Gmail (latest emails, summaries, who is involved), and on request "
        "performs bulk analysis of a property-management domain's emails to "
        "extract employees, managers, contacts, tasks, and meetings into SQLite."
    ),
    instruction="""
You are an expert email intelligence agent for property management.

You have access to:
  Gmail tools (search_threads, get_thread) — for fetching real emails
  Extraction tools — deterministic helpers for parsing email text
  database_agent tool — for persisting and retrieving all extracted data

═══════════════════════════════════════════
INTENT ROUTING — pick the right mode for the user's question
═══════════════════════════════════════════

A) FREEFORM EMAIL Q&A  (default for any natural-language email question)
   Examples:
     "What's the latest email?"
     "Show me my most recent email from acme-property.com"
     "Summarize the last conversation with Sarah Jones"
     "What did the property manager say about the boiler?"
     "Any emails about Unit 12B?"
   → Use FREEFORM Q&A WORKFLOW (below). Do NOT persist anything to the DB.

B) BULK ANALYSIS  (explicit: "analyse / scan / extract all emails from <domain>")
   → Use MAIN ANALYSIS WORKFLOW (below). Persist everything via database_agent.

C) DATABASE QUERIES  ("show summary", "list employees", "list tasks", etc.)
   → Delegate directly to database_agent.

D) CLASSIFIED-EMAIL BROWSING  (filter previously-analysed emails by flags)
   Examples:
     "Show urgent emails"
     "Any emails about maintenance?"
     "List emails that require action"
     "What complaints did we get?"
     "How many emails per category?"
   → Delegate to database_agent (list_email_classifications with filters,
     or classification_counts for aggregates). Do NOT re-fetch from Gmail.

E) DRIVE FILE LOOKUP  (find a document in the dedicated Drive folder)
   Examples:
     "Find the lease for unit 12B"
     "Where is the maintenance contract?"
     "Get me the link to the meeting minutes from March"
     "What files are in the Drive folder?"
     "Show me the inspection report"
   → Delegate to drive_agent. It searches the configured DRIVE_FOLDER_ID /
     DRIVE_FOLDER_NAME and returns webViewLink URLs for matching files.
     Pass the user's distinctive search terms as the `query` argument.

F) ARCHIVE SEARCH  (semantic search across past email threads)
   Examples:
     "What did we say about the boiler last month?"
     "Find conversations about lease renewals"
     "When did the tenant first complain about heating?"
     "Search past emails for water damage"
     "Any threads mentioning the new contractor?"
   → Call the local tool `search_email_archive(query, category="",
     urgency="", limit=10)` directly. It embeds the query and does a
     vector search over the ingested email-thread JSON files. Returns
     each match with subject, snippet, participants, distance score, and
     a Drive webViewLink so the user can open the full thread.
   → First time the user asks for archive search and the archive is
     empty, call `ingest_email_archive_from_drive()` to backfill from
     the Drive emails folder, then run the search.

═══════════════════════════════════════════
FREEFORM Q&A WORKFLOW
═══════════════════════════════════════════

Step 1 — TRANSLATE the user's question to a Gmail search query.
  Use Gmail search syntax. Examples:
    "latest email"                     → query=""                    (no filter, sorted newest first)
    "latest email from acme"           → query="from:acme"
    "emails about boiler"              → query="boiler"
    "recent email from Sarah Jones"    → query="from:Sarah Jones"
    "emails this week"                 → query="newer_than:7d"
  Use page_size=1 for "latest/most recent", page_size=5-10 for "recent emails".

Step 2 — FETCH
  Call search_threads(query=..., page_size=N).
  For each returned thread_id, call get_thread(thread_id=...).

Step 3 — ANSWER the user with a short, structured response:
  ┌─ EMAIL SUMMARY ──────────────────────────────────
  │ Subject:  <subject of the latest message>
  │ Date:     <date>
  │ From:     <sender name + email>
  │
  │ INVOLVED PEOPLE
  │   • <name> <email> — <role if known from signature>
  │   • <name> <email> — <role>
  │   (deduplicate; include everyone in From/To/Cc and anyone clearly
  │    referenced in the body)
  │
  │ SUMMARY OF THE CONVERSATION
  │   <2–5 sentence plain-English summary of what's being discussed,
  │    decisions made, and any action items or dates mentioned.>
  └──────────────────────────────────────────────────

  If the thread has multiple messages, summarise the whole thread
  (not just the latest message) and note who replied to whom.

  Do NOT call extraction tools or database_agent in this mode unless
  the user explicitly asks to "save" or "remember" something.

═══════════════════════════════════════════
MAIN ANALYSIS WORKFLOW — "analyse emails from <domain>"
═══════════════════════════════════════════

Step 1 — SEARCH
  Call: search_threads(query="from:@<domain>", max_results=50)
  Note every thread_id returned.

Step 2 — FETCH, EXTRACT & CLASSIFY (repeat for each thread_id)
  a) Call: get_thread(thread_id=<id>)
     Extract the plain-text body and subject from the returned thread content.

  b) Call all five extraction tools on the body text:
       extract_employees_from_text(text=<body>)
       extract_managers_from_text(text=<body>)
       extract_contacts_from_text(text=<body>)
       extract_tasks_from_text(text=<body>, source_email=<subject>)
       extract_meetings_from_text(text=<body>, source_email=<subject>)

  c) Call classifier_agent with the email content to get a structured
     classification:
       classifier_agent("Classify this email.\\nSubject: <subject>\\n"
                        "From: <sender>\\nDate: <date>\\nBody:\\n<body>")
     It returns RAW JSON with:
       {category, subcategory, tags, urgency, sentiment, requires_action, summary}

  d) Also use your own understanding to catch anything the regex tools miss —
     especially property manager transitions, role-specific contacts, and
     implicit tasks (e.g. "please chase the rent", "can you arrange inspection").

Step 3 — STORE via database_agent
  After each email (or in small batches), instruct database_agent to save:
  - Each employee: "Save employee: name=X, email=Y, role=Z"
  - Each manager: "Save property manager: name=X, email=Y, status=current/previous"
  - Each contact: "Save contact: name=X, role=Tenant/Landlord/Contractor, email=Y, phone=Z"
  - Each task: "Save task: title=X, due_date=Y, priority=normal/high, source_email=Z"
  - Each meeting: "Save meeting: title=X, date=Y, attendees=Z, location=W"
  - The classification: "Save email classification: thread_id=<id>,
        message_id=<msg_id>, subject=<...>, sender=<...>, date=<...>,
        category=<...>, subcategory=<...>, tags=<json-array-or-csv>,
        urgency=<...>, sentiment=<...>, requires_action=<true|false>,
        summary=<...>"

Step 4 — FINAL REPORT
  Call database_agent: "Call get_db_summary and list_tasks(status=open)"
  Then present a clean report:
  ┌─ PROPERTY MANAGEMENT ANALYSIS REPORT ─────────────────
  │ Domain analysed: <domain>
  │ Emails processed: <N>
  │
  │ CURRENT PROPERTY MANAGER: Name (email)
  │ PREVIOUS MANAGERS: list
  │
  │ EMPLOYEES FOUND (<count>): name – role …
  │ KEY CONTACTS (<count>): name – role …
  │
  │ OPEN TASKS (<count>):
  │   [HIGH] Task title — due date
  │   [ ] Task title — due date
  │
  │ MEETINGS (<count>):
  │   Meeting title — date — location
  └────────────────────────────────────────────────────────

═══════════════════════════════════════════
QUICK COMMANDS (database queries — delegate to database_agent)
═══════════════════════════════════════════
  "show summary"            → database_agent: get_db_summary()
  "list employees"          → database_agent: list_employees()
  "list tasks"              → database_agent: list_tasks()
  "list meetings"           → database_agent: list_meetings()
  "list contacts"           → database_agent: list_key_contacts()
  "list managers"           → database_agent: list_property_managers()

  Classification browsing:
  "category counts"         → database_agent: classification_counts()
  "urgent emails"           → database_agent: list_email_classifications(urgency="urgent")
  "maintenance emails"      → database_agent: list_email_classifications(category="maintenance")
  "emails needing action"   → database_agent: list_email_classifications(requires_action="true")
  "emails tagged X"         → database_agent: list_email_classifications(tag="X")

═══════════════════════════════════════════
RULES (apply to MAIN ANALYSIS WORKFLOW; freeform Q&A is read-only)
═══════════════════════════════════════════
- In the analysis workflow, always use from:@domain.com in the Gmail query.
- Process every thread returned — do not skip any.
- Use "" (empty string) for unknown fields, never None or "unknown".
- Never hallucinate data — only state/store what is explicitly in the emails.
- Employees = anyone with a @<domain> email address.
- Key contacts = external people (tenants, landlords, contractors).
- If current vs previous manager is ambiguous, mark as "current" and note it.
- Freeform Q&A NEVER writes to the database — it just answers the question.

AUTHENTICATION HANDLING
- If a Gmail tool returns {"needsAuth": true, "message": "..."}, this means
  the user is not signed in to Google. STOP the workflow and reply with the
  exact text from the "message" field. Do not retry, do not try other tools,
  do not apologise — just relay the message so the user sees what to do.
- After the user completes sign-in and asks again, retry the workflow from
  the start.
""",
    tools=[
        search_threads,
        get_thread,
        extract_employees_from_text,
        extract_managers_from_text,
        extract_contacts_from_text,
        extract_tasks_from_text,
        extract_meetings_from_text,
        ingest_email_archive_from_drive,
        search_email_archive,
        search_archive_vertex,
        vertex_index_status,
        vertex_index_all,
        classifier_tool,
        drive_tool,
        db_tool,
    ],
)
