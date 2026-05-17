"""
Property Management Email Analyzer — root agent.

Architecture:
  root_agent (property_email_analyzer)
  │  Gmail MCP tools  → https://gmailmcp.googleapis.com/mcp/v1  (JSON-RPC over httpx)
  │    • search_threads(query, page_size, page_token)
  │    • get_thread(thread_id, message_format)
  │  (Direct httpx calls — MCP Python SDK transports have async bugs with ADK)
  │  Python extraction tools (deterministic regex helpers)
  │    • extract_employees_from_text
  │    • extract_managers_from_text
  │    • extract_contacts_from_text
  │    • extract_tasks_from_text
  │    • extract_meetings_from_text
  │
  └─ AgentTool → database_agent (manages SQLite via direct Python functions)
"""
import json
import logging
import os
import re
import sys
from typing import Optional

import httpx
from google.adk.agents import Agent
from google.adk.tools.agent_tool import AgentTool
from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials

from .database_agent.agent import database_agent

logging.basicConfig(stream=sys.stderr, level=logging.INFO)
logger = logging.getLogger(__name__)

AGENT_DIR = os.path.dirname(os.path.abspath(__file__))
TOKEN_PATH = os.path.join(AGENT_DIR, "token.json")
GMAIL_MCP_URL = "https://gmailmcp.googleapis.com/mcp/v1"
SCOPES = [
    "https://www.googleapis.com/auth/gmail.readonly",
    "https://www.googleapis.com/auth/gmail.compose",
]


def _load_access_token() -> str:
    """Return a valid OAuth access token, refreshing if needed.
    Returns empty string if token.json is missing or invalid."""
    if not os.path.exists(TOKEN_PATH):
        logger.warning(
            "token.json not found. Run:  python3.11 property_management_agent/login.py"
        )
        return ""
    try:
        creds = Credentials.from_authorized_user_file(TOKEN_PATH, SCOPES)
        if not creds.valid:
            if creds.expired and creds.refresh_token:
                logger.info("Refreshing Gmail OAuth token...")
                creds.refresh(Request())
                with open(TOKEN_PATH, "w") as f:
                    f.write(creds.to_json())
            else:
                logger.warning(
                    "Gmail token invalid. Run:  python3.11 property_management_agent/login.py"
                )
                return ""
        return creds.token
    except Exception as e:
        logger.warning(f"Could not load Gmail token: {e}")
        return ""


# ── Gmail MCP JSON-RPC client ──────────────────────────────────────────────────
# Calls the official Google Gmail MCP server (gmailmcp.googleapis.com) directly
# via httpx. The MCP Python SDK transports (Streamable HTTP, SSE) have async
# task-management bugs with ADK, so we bypass them and speak JSON-RPC ourselves.

def _mcp_call(tool_name: str, arguments: dict) -> dict:
    """Invoke a Gmail MCP tool via JSON-RPC over HTTP. Returns the parsed result dict."""
    token = _load_access_token()
    if not token:
        return {"isError": True, "error": "Not authenticated. Run login.py first."}
    payload = {
        "jsonrpc": "2.0",
        "id": 1,
        "method": "tools/call",
        "params": {"name": tool_name, "arguments": arguments},
    }
    try:
        resp = httpx.post(
            GMAIL_MCP_URL,
            json=payload,
            headers={
                "Authorization": f"Bearer {token}",
                "Content-Type": "application/json",
                "Accept": "application/json",
            },
            timeout=60.0,
        )
        resp.raise_for_status()
        data = resp.json()
        if "error" in data:
            return {"isError": True, "error": data["error"]}
        return data.get("result", {})
    except httpx.HTTPError as e:
        return {"isError": True, "error": f"HTTP error: {e}"}


def search_threads(query: str = "", page_size: int = 20, page_token: str = "") -> str:
    """
    Search Gmail threads using the official Gmail MCP server.
    Use Gmail search syntax in the query, e.g. "from:@acme-property.com".

    Args:
        query: Gmail search query string (Gmail syntax).
        page_size: Max threads to return (default 20, max 50).
        page_token: Optional token to fetch next page of results.

    Returns:
        JSON string with thread list. Each thread has an id and snippet.
    """
    args: dict = {"query": query, "pageSize": max(1, min(page_size, 50))}
    if page_token:
        args["pageToken"] = page_token
    result = _mcp_call("search_threads", args)
    if result.get("isError"):
        return json.dumps(result)
    # Unwrap MCP content[].text → parsed JSON when possible
    content = result.get("content", [])
    if content and content[0].get("type") == "text":
        text = content[0].get("text", "")
        try:
            return json.dumps(json.loads(text))
        except json.JSONDecodeError:
            return json.dumps({"raw": text})
    return json.dumps(result)


def get_thread(thread_id: str, message_format: str = "FULL_CONTENT") -> str:
    """
    Fetch the full content of a Gmail thread by its ID, using the official Gmail MCP server.

    Args:
        thread_id: The Gmail thread ID (returned by search_threads).
        message_format: "FULL_CONTENT" (default) or "MINIMAL".

    Returns:
        JSON string with the full thread including all messages (from, to,
        subject, date, body text).
    """
    result = _mcp_call(
        "get_thread",
        {"threadId": thread_id, "messageFormat": message_format},
    )
    if result.get("isError"):
        return json.dumps(result)
    content = result.get("content", [])
    if content and content[0].get("type") == "text":
        text = content[0].get("text", "")
        try:
            return json.dumps(json.loads(text))
        except json.JSONDecodeError:
            return json.dumps({"raw": text})
    return json.dumps(result)


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
    description="Analyses emails from a property management company domain and extracts structured intelligence.",
    instruction="""
You are an expert email intelligence agent for property management.

You have access to:
  Gmail MCP tools (search_threads, get_thread) — for fetching real emails
  Extraction tools — deterministic helpers for parsing email text
  database_agent tool — for persisting and retrieving all extracted data

═══════════════════════════════════════════
MAIN WORKFLOW — "analyse emails from <domain>"
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
QUICK COMMANDS
═══════════════════════════════════════════
  "show summary"     → database_agent: get_db_summary()
  "list employees"   → database_agent: list_employees()
  "list tasks"       → database_agent: list_tasks()
  "list meetings"    → database_agent: list_meetings()
  "list contacts"    → database_agent: list_key_contacts()
  "list managers"    → database_agent: list_property_managers()

═══════════════════════════════════════════
RULES
═══════════════════════════════════════════
- Always use from:@domain.com in the Gmail search query.
- Process every thread returned — do not skip any.
- Use "" (empty string) for unknown fields, never None or "unknown".
- Never hallucinate data — only store what is explicitly in the emails.
- Employees = anyone with a @<domain> email address.
- Key contacts = external people (tenants, landlords, contractors).
- If current vs previous manager is ambiguous, mark as "current" and note it.
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
