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
  └─ AgentTool → database_agent (manages SQLite via direct Python functions)
"""
import base64
import json
import logging
import os
import re
import sys
import threading
from typing import Optional

from google.adk.agents import Agent
from google.adk.tools.agent_tool import AgentTool
from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError

from .database_agent.agent import database_agent

logging.basicConfig(stream=sys.stderr, level=logging.INFO)
logger = logging.getLogger(__name__)

AGENT_DIR = os.path.dirname(os.path.abspath(__file__))
TOKEN_PATH = os.path.join(AGENT_DIR, "token.json")
CREDENTIALS_PATH = os.path.join(AGENT_DIR, "credentials-web.json")
AUTH_PORT = 8080  # OAuth callback port — must match a redirect URI registered
                  # in your Google Cloud OAuth client (credentials-web.json).
SCOPES = ["https://www.googleapis.com/auth/gmail.readonly"]

# ── Auth state (shared between background thread and callers) ─────────────────
_auth_thread: Optional[threading.Thread] = None
_auth_done = threading.Event()
_auth_error: Optional[str] = None


def _run_auth_flow() -> None:
    """Runs the OAuth flow in a background thread, opens the browser, saves token.json."""
    global _auth_error
    try:
        flow = InstalledAppFlow.from_client_secrets_file(CREDENTIALS_PATH, SCOPES)
        creds = flow.run_local_server(port=AUTH_PORT, open_browser=True)
        with open(TOKEN_PATH, "w") as f:
            f.write(creds.to_json())
        logger.info("Gmail auth completed — token.json saved.")
    except Exception as e:
        _auth_error = str(e)
        logger.error(f"Gmail auth failed: {e}")
    finally:
        _auth_done.set()


def _get_valid_credentials() -> Optional[Credentials]:
    """Load valid credentials from token.json, refreshing if expired. Returns None if missing."""
    if not os.path.exists(TOKEN_PATH):
        return None
    try:
        creds = Credentials.from_authorized_user_file(TOKEN_PATH, SCOPES)
        if creds.valid:
            return creds
        if creds.expired and creds.refresh_token:
            logger.info("Refreshing Gmail OAuth token...")
            creds.refresh(Request())
            with open(TOKEN_PATH, "w") as f:
                f.write(creds.to_json())
            return creds
    except Exception as e:
        logger.warning(f"Error loading credentials: {e}")
    return None


def _ensure_authenticated() -> tuple[Optional[str], Optional[str]]:
    """Return (token, user_message). If token is None, user_message tells the
    user what to do (browser opened, sign-in in progress, etc.)."""
    global _auth_thread, _auth_error

    creds = _get_valid_credentials()
    if creds is not None:
        return creds.token, None

    # No valid token — need to authenticate
    if not os.path.exists(CREDENTIALS_PATH):
        return None, (
            f"Gmail authentication required but credentials-web.json is missing.\n"
            f"Place your OAuth client JSON at: {CREDENTIALS_PATH}"
        )

    # Previous auth attempt failed — reset and allow retry
    if _auth_done.is_set() and _auth_error:
        err = _auth_error
        _auth_thread = None
        _auth_done.clear()
        _auth_error = None
        return None, (
            f"Gmail authentication failed: {err}\n"
            "Please ask me again to retry — a new browser window will open."
        )

    # Auth still in progress
    if _auth_thread is not None and _auth_thread.is_alive():
        return None, (
            "Gmail sign-in is in progress. Please complete the sign-in in the "
            "browser window that opened, then ask me again to continue."
        )

    # Start a fresh auth flow — opens the browser automatically
    _auth_done.clear()
    _auth_error = None
    _auth_thread = threading.Thread(target=_run_auth_flow, daemon=True)
    _auth_thread.start()
    return None, (
        "Gmail authentication required. A browser window has just been opened "
        "for you to sign in to your Google account. Please grant the requested "
        "permissions, then ask me again to fetch your emails."
    )


# ── Gmail REST API client (via googleapiclient) ───────────────────────────────
# Uses the standard Gmail API (gmail.googleapis.com), not the Gmail MCP server.
# Same OAuth token, no developer-preview gating.

def _get_gmail_service():
    """Build a Gmail API service object using the user's OAuth credentials.
    Returns (service, error_dict_or_None). If error_dict is set, surface it
    to the user instead of using the service."""
    token, user_message = _ensure_authenticated()
    if token is None:
        return None, {"isError": True, "needsAuth": True, "message": user_message}
    try:
        creds = _get_valid_credentials()
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


# ── Root agent ─────────────────────────────────────────────────────────────────

db_tool = AgentTool(agent=database_agent)

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

Step 2 — FETCH & EXTRACT (repeat for each thread_id)
  a) Call: get_thread(thread_id=<id>)
     Extract the plain-text body and subject from the returned thread content.

  b) Call all five extraction tools on the body text:
       extract_employees_from_text(text=<body>)
       extract_managers_from_text(text=<body>)
       extract_contacts_from_text(text=<body>)
       extract_tasks_from_text(text=<body>, source_email=<subject>)
       extract_meetings_from_text(text=<body>, source_email=<subject>)

  c) Also use your own understanding to catch anything the regex tools miss —
     especially property manager transitions, role-specific contacts, and
     implicit tasks (e.g. "please chase the rent", "can you arrange inspection").

Step 3 — STORE via database_agent
  After each email (or in small batches), instruct the database_agent tool to save:
  - Each employee: "Save employee: name=X, email=Y, role=Z"
  - Each manager: "Save property manager: name=X, email=Y, status=current/previous"
  - Each contact: "Save contact: name=X, role=Tenant/Landlord/Contractor, email=Y, phone=Z"
  - Each task: "Save task: title=X, due_date=Y, priority=normal/high, source_email=Z"
  - Each meeting: "Save meeting: title=X, date=Y, attendees=Z, location=W"

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
  "show summary"     → database_agent: get_db_summary()
  "list employees"   → database_agent: list_employees()
  "list tasks"       → database_agent: list_tasks()
  "list meetings"    → database_agent: list_meetings()
  "list contacts"    → database_agent: list_key_contacts()
  "list managers"    → database_agent: list_property_managers()

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
        db_tool,
    ],
)
