"""
SQLite schema and CRUD helpers used directly by the database_agent tools.
DB_PATH defaults to property_data.db next to this file, or from DB_PATH env var.
"""
import json
import os
import sqlite3
from datetime import datetime
from typing import Any

# Resolve DB_PATH so a relative value from .env (e.g. DB_PATH=property_data.db)
# is interpreted relative to the property_management_agent package dir, NOT the
# current working directory. Otherwise the DB ends up wherever `adk web` was
# launched, which silently splits writes across multiple files.
_AGENT_DIR = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
_RAW_DB_PATH = os.getenv("DB_PATH") or "property_data.db"
DB_PATH = (
    _RAW_DB_PATH
    if os.path.isabs(_RAW_DB_PATH)
    else os.path.normpath(os.path.join(_AGENT_DIR, _RAW_DB_PATH))
)


def get_conn() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_db() -> None:
    with get_conn() as conn:
        conn.executescript("""
            CREATE TABLE IF NOT EXISTS employees (
                id          INTEGER PRIMARY KEY AUTOINCREMENT,
                name        TEXT NOT NULL,
                email       TEXT UNIQUE NOT NULL,
                role        TEXT,
                department  TEXT,
                phone       TEXT,
                created_at  DATETIME DEFAULT CURRENT_TIMESTAMP
            );

            CREATE TABLE IF NOT EXISTS property_managers (
                id          INTEGER PRIMARY KEY AUTOINCREMENT,
                name        TEXT NOT NULL,
                email       TEXT NOT NULL,
                status      TEXT NOT NULL DEFAULT 'current',
                properties  TEXT,
                notes       TEXT,
                created_at  DATETIME DEFAULT CURRENT_TIMESTAMP
            );

            CREATE TABLE IF NOT EXISTS key_contacts (
                id          INTEGER PRIMARY KEY AUTOINCREMENT,
                name        TEXT NOT NULL,
                email       TEXT,
                phone       TEXT,
                role        TEXT,
                company     TEXT,
                notes       TEXT,
                created_at  DATETIME DEFAULT CURRENT_TIMESTAMP
            );

            CREATE TABLE IF NOT EXISTS tasks (
                id           INTEGER PRIMARY KEY AUTOINCREMENT,
                title        TEXT NOT NULL,
                description  TEXT,
                assigned_to  TEXT,
                due_date     TEXT,
                status       TEXT DEFAULT 'open',
                priority     TEXT DEFAULT 'normal',
                source_email TEXT,
                created_at   DATETIME DEFAULT CURRENT_TIMESTAMP
            );

            CREATE TABLE IF NOT EXISTS meetings (
                id           INTEGER PRIMARY KEY AUTOINCREMENT,
                title        TEXT NOT NULL,
                date         TEXT,
                time         TEXT,
                attendees    TEXT,
                location     TEXT,
                agenda       TEXT,
                outcome      TEXT,
                source_email TEXT,
                created_at   DATETIME DEFAULT CURRENT_TIMESTAMP
            );

            -- Per-email classification produced by the classifier_agent.
            -- One row per (thread_id, message_id); tags is a JSON array string.
            CREATE TABLE IF NOT EXISTS email_classifications (
                id               INTEGER PRIMARY KEY AUTOINCREMENT,
                thread_id        TEXT NOT NULL,
                message_id       TEXT DEFAULT '',
                subject          TEXT,
                sender           TEXT,
                date             TEXT,
                category         TEXT,
                subcategory      TEXT,
                tags             TEXT DEFAULT '[]',
                urgency          TEXT DEFAULT 'normal',
                sentiment        TEXT DEFAULT 'neutral',
                requires_action  INTEGER DEFAULT 0,
                summary          TEXT,
                classified_at    DATETIME DEFAULT CURRENT_TIMESTAMP,
                UNIQUE(thread_id, message_id)
            );

            CREATE INDEX IF NOT EXISTS idx_class_category ON email_classifications(category);
            CREATE INDEX IF NOT EXISTS idx_class_urgency  ON email_classifications(urgency);
            CREATE INDEX IF NOT EXISTS idx_class_action   ON email_classifications(requires_action);
        """)


# ── Role merging ───────────────────────────────────────────────────────────────
# Same person can be seen in different emails with different roles. We store
# all their roles as a single comma-separated string, deduplicated case-
# insensitively while preserving original casing for display.

def _merge_roles(existing: str, new: str) -> str:
    """Merge two role strings into a deduplicated comma-separated list.

    "Owner" + "Owner/Committee Member" → "Owner, Owner/Committee Member"
    "Tenant, Client" + "owner" → "Tenant, Client, owner"  (case-insensitive dedupe)
    "" + "Owner" → "Owner"
    """
    def _split(s: str) -> list[str]:
        return [p.strip() for p in (s or "").split(",") if p.strip()]

    seen: dict[str, str] = {}  # lowercase → original casing (first seen wins)
    for role in _split(existing) + _split(new):
        key = role.lower()
        if key not in seen:
            seen[key] = role
    return ", ".join(seen.values())


def _consolidate_duplicates() -> None:
    """One-shot consolidation pass: merge duplicate user rows that already
    exist in the DB. Idempotent — safe to run on every init.

    Dedupe keys:
      employees       — LOWER(email)
      key_contacts    — LOWER(email) when email present; else LOWER(name)
    Roles are merged via _merge_roles.
    property_managers and tasks/meetings are left alone (different semantics).
    """
    with get_conn() as conn:
        # ── employees: dedupe by LOWER(email) ──────────────────────────────────
        rows = conn.execute(
            "SELECT id, name, email, role FROM employees WHERE email != '' ORDER BY id"
        ).fetchall()
        groups: dict[str, list] = {}
        for r in rows:
            groups.setdefault(r["email"].lower(), []).append(r)
        for _, group in groups.items():
            if len(group) <= 1:
                continue
            keeper = group[0]
            merged = keeper["role"] or ""
            for dup in group[1:]:
                merged = _merge_roles(merged, dup["role"] or "")
                conn.execute("DELETE FROM employees WHERE id=?", (dup["id"],))
            conn.execute("UPDATE employees SET role=? WHERE id=?", (merged, keeper["id"]))

        # ── key_contacts: dedupe by LOWER(email) (or LOWER(name) if no email) ──
        rows = conn.execute(
            "SELECT id, name, email, phone, role, company, notes FROM key_contacts ORDER BY id"
        ).fetchall()
        groups = {}
        for r in rows:
            key = (r["email"] or "").lower().strip()
            if not key:
                key = "name::" + (r["name"] or "").lower().strip()
            if not key or key == "name::":
                continue
            groups.setdefault(key, []).append(r)
        for _, group in groups.items():
            if len(group) <= 1:
                continue
            keeper = group[0]
            merged_role = keeper["role"] or ""
            merged_phone = keeper["phone"] or ""
            merged_company = keeper["company"] or ""
            merged_notes = keeper["notes"] or ""
            for dup in group[1:]:
                merged_role = _merge_roles(merged_role, dup["role"] or "")
                merged_phone = merged_phone or (dup["phone"] or "")
                merged_company = merged_company or (dup["company"] or "")
                if dup["notes"] and dup["notes"] not in merged_notes:
                    merged_notes = (merged_notes + "; " + dup["notes"]).strip("; ")
                conn.execute("DELETE FROM key_contacts WHERE id=?", (dup["id"],))
            conn.execute("""
                UPDATE key_contacts
                SET role=?, phone=?, company=?, notes=?
                WHERE id=?
            """, (merged_role, merged_phone, merged_company, merged_notes, keeper["id"]))


init_db()
_consolidate_duplicates()


# ── Employees ──────────────────────────────────────────────────────────────────

def upsert_employee(name: str, email: str, role: str = "", department: str = "", phone: str = "") -> str:
    """Insert or update an employee record. Roles accumulate: if the same
    email is seen with a different role, the new role is appended to the
    existing comma-separated role list (deduplicated case-insensitively)."""
    with get_conn() as conn:
        existing = conn.execute(
            "SELECT id, role FROM employees WHERE LOWER(email)=LOWER(?)", (email,)
        ).fetchone()
        if existing:
            merged_role = _merge_roles(existing["role"] or "", role)
            conn.execute("""
                UPDATE employees
                SET name=?,
                    role=?,
                    department=COALESCE(NULLIF(?,  ''), department),
                    phone=COALESCE(NULLIF(?,  ''), phone)
                WHERE id=?
            """, (name, merged_role, department, phone, existing["id"]))
            return f"Employee '{name}' ({email}) updated. Roles: {merged_role}"
        conn.execute("""
            INSERT INTO employees (name, email, role, department, phone)
            VALUES (?, ?, ?, ?, ?)
        """, (name, email, role, department, phone))
    return f"Employee '{name}' ({email}) saved."


def list_employees() -> str:
    """Return all employees as a JSON string."""
    with get_conn() as conn:
        rows = conn.execute("SELECT * FROM employees ORDER BY name").fetchall()
    return json.dumps([dict(r) for r in rows], default=str)


# ── Property Managers ──────────────────────────────────────────────────────────

def upsert_property_manager(
    name: str, email: str, status: str = "current", properties: str = "", notes: str = ""
) -> str:
    """Insert or update a property manager. status: 'current' or 'previous'."""
    if status not in ("current", "previous"):
        return f"Error: status must be 'current' or 'previous', got '{status}'"
    with get_conn() as conn:
        existing = conn.execute(
            "SELECT id FROM property_managers WHERE email=?", (email,)
        ).fetchone()
        if existing:
            conn.execute("""
                UPDATE property_managers
                SET name=?, status=?,
                    properties=COALESCE(NULLIF(?,  ''), properties),
                    notes=COALESCE(NULLIF(?,  ''), notes)
                WHERE email=?
            """, (name, status, properties, notes, email))
        else:
            conn.execute("""
                INSERT INTO property_managers (name, email, status, properties, notes)
                VALUES (?, ?, ?, ?, ?)
            """, (name, email, status, properties, notes))
    return f"Property manager '{name}' ({status}) saved."


def list_property_managers() -> str:
    """Return all property managers as a JSON string."""
    with get_conn() as conn:
        rows = conn.execute(
            "SELECT * FROM property_managers ORDER BY status DESC, name"
        ).fetchall()
    return json.dumps([dict(r) for r in rows], default=str)


# ── Key Contacts ───────────────────────────────────────────────────────────────

def upsert_key_contact(
    name: str, role: str, email: str = "", phone: str = "", company: str = "", notes: str = ""
) -> str:
    """Insert or update a key contact (tenant, contractor, landlord, etc.).

    Dedupe key: LOWER(email) when email is present, otherwise LOWER(name).
    When the same person is seen with a different role, the new role is
    APPENDED to the existing comma-separated role list (deduplicated
    case-insensitively) — they are no longer split into multiple rows.
    """
    with get_conn() as conn:
        if email:
            existing = conn.execute(
                "SELECT id, role, phone, company, notes FROM key_contacts "
                "WHERE LOWER(email)=LOWER(?)",
                (email,),
            ).fetchone()
        else:
            existing = conn.execute(
                "SELECT id, role, phone, company, notes FROM key_contacts "
                "WHERE LOWER(name)=LOWER(?) AND (email='' OR email IS NULL)",
                (name,),
            ).fetchone()

        if existing:
            merged_role = _merge_roles(existing["role"] or "", role)
            conn.execute("""
                UPDATE key_contacts
                SET name=?,
                    email=COALESCE(NULLIF(?,  ''), email),
                    phone=COALESCE(NULLIF(?,  ''), phone),
                    role=?,
                    company=COALESCE(NULLIF(?,  ''), company),
                    notes=COALESCE(NULLIF(?,  ''), notes)
                WHERE id=?
            """, (name, email, phone, merged_role, company, notes, existing["id"]))
            return f"Contact '{name}' updated. Roles: {merged_role}"

        conn.execute("""
            INSERT INTO key_contacts (name, email, phone, role, company, notes)
            VALUES (?, ?, ?, ?, ?, ?)
        """, (name, email, phone, role, company, notes))
    return f"Contact '{name}' ({role}) saved."


def list_key_contacts() -> str:
    """Return all key contacts as a JSON string."""
    with get_conn() as conn:
        rows = conn.execute("SELECT * FROM key_contacts ORDER BY role, name").fetchall()
    return json.dumps([dict(r) for r in rows], default=str)


# ── Tasks ──────────────────────────────────────────────────────────────────────

def add_task(
    title: str,
    description: str = "",
    assigned_to: str = "",
    due_date: str = "",
    priority: str = "normal",
    source_email: str = "",
) -> str:
    """Add a new task. priority: 'normal' or 'high'."""
    with get_conn() as conn:
        conn.execute("""
            INSERT INTO tasks (title, description, assigned_to, due_date, priority, source_email)
            VALUES (?, ?, ?, ?, ?, ?)
        """, (title, description, assigned_to, due_date, priority, source_email))
    return f"Task '{title}' saved."


def update_task_status(task_id: int, status: str) -> str:
    """Update task status: 'open', 'in_progress', or 'completed'."""
    with get_conn() as conn:
        conn.execute("UPDATE tasks SET status=? WHERE id=?", (status, task_id))
    return f"Task {task_id} updated to '{status}'."


def list_tasks(status: str = "") -> str:
    """Return tasks as JSON, optionally filtered by status."""
    with get_conn() as conn:
        if status:
            rows = conn.execute(
                "SELECT * FROM tasks WHERE status=? ORDER BY priority DESC, due_date",
                (status,),
            ).fetchall()
        else:
            rows = conn.execute(
                "SELECT * FROM tasks ORDER BY priority DESC, due_date"
            ).fetchall()
    return json.dumps([dict(r) for r in rows], default=str)


# ── Meetings ───────────────────────────────────────────────────────────────────

def add_meeting(
    title: str,
    date: str = "",
    time: str = "",
    attendees: str = "",
    location: str = "",
    agenda: str = "",
    source_email: str = "",
) -> str:
    """Add a meeting extracted from email."""
    with get_conn() as conn:
        conn.execute("""
            INSERT INTO meetings (title, date, time, attendees, location, agenda, source_email)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        """, (title, date, time, attendees, location, agenda, source_email))
    return f"Meeting '{title}' saved."


def list_meetings() -> str:
    """Return all meetings as JSON."""
    with get_conn() as conn:
        rows = conn.execute(
            "SELECT * FROM meetings ORDER BY date DESC, created_at DESC"
        ).fetchall()
    return json.dumps([dict(r) for r in rows], default=str)


# ── Email Classifications ─────────────────────────────────────────────────────
# Produced by the classifier_agent. Used to filter emails in the DB and let
# the AI browse data by category / urgency / action-required / tags.

VALID_CATEGORIES = {
    "maintenance", "billing", "leasing", "legal", "handover",
    "complaint", "emergency", "administrative", "communication", "other",
}
VALID_URGENCY = {"low", "normal", "high", "urgent"}
VALID_SENTIMENT = {"positive", "neutral", "negative"}


def save_email_classification(
    thread_id: str,
    category: str,
    summary: str,
    message_id: str = "",
    subject: str = "",
    sender: str = "",
    date: str = "",
    subcategory: str = "",
    tags: str = "",
    urgency: str = "normal",
    sentiment: str = "neutral",
    requires_action: bool = False,
) -> str:
    """
    Save (or replace) the classification for a specific email.

    Args:
        thread_id: Gmail thread ID (required).
        category: One of maintenance, billing, leasing, legal, handover,
                  complaint, emergency, administrative, communication, other.
        summary: 1-2 sentence summary of the email content.
        message_id: Gmail message ID inside the thread (optional).
        subject, sender, date: Email metadata.
        subcategory: Free-form refinement (e.g. "boiler repair", "rent arrears").
        tags: Either a JSON array string like '["urgent","follow-up"]' OR a
              comma-separated string like "urgent,follow-up" (auto-normalised).
        urgency: low | normal | high | urgent.
        sentiment: positive | neutral | negative.
        requires_action: True if this email needs follow-up action.
    """
    if category not in VALID_CATEGORIES:
        category = "other"
    if urgency not in VALID_URGENCY:
        urgency = "normal"
    if sentiment not in VALID_SENTIMENT:
        sentiment = "neutral"

    # Normalise tags to a JSON array string
    tags_json = "[]"
    if tags:
        tags = tags.strip()
        try:
            parsed = json.loads(tags)
            if isinstance(parsed, list):
                tags_json = json.dumps([str(t).strip() for t in parsed if str(t).strip()])
        except (json.JSONDecodeError, ValueError):
            parts = [t.strip() for t in tags.split(",") if t.strip()]
            tags_json = json.dumps(parts)

    with get_conn() as conn:
        conn.execute("""
            INSERT INTO email_classifications
                (thread_id, message_id, subject, sender, date, category,
                 subcategory, tags, urgency, sentiment, requires_action, summary)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(thread_id, message_id) DO UPDATE SET
                subject         = COALESCE(NULLIF(excluded.subject,''), subject),
                sender          = COALESCE(NULLIF(excluded.sender,''), sender),
                date            = COALESCE(NULLIF(excluded.date,''), date),
                category        = excluded.category,
                subcategory     = COALESCE(NULLIF(excluded.subcategory,''), subcategory),
                tags            = excluded.tags,
                urgency         = excluded.urgency,
                sentiment       = excluded.sentiment,
                requires_action = excluded.requires_action,
                summary         = excluded.summary,
                classified_at   = CURRENT_TIMESTAMP
        """, (
            thread_id, message_id, subject, sender, date, category,
            subcategory, tags_json, urgency, sentiment, int(bool(requires_action)),
            summary,
        ))
    return f"Classification for thread '{thread_id}' ({category}) saved."


def list_email_classifications(
    category: str = "",
    urgency: str = "",
    sentiment: str = "",
    requires_action: str = "",
    tag: str = "",
    limit: int = 50,
) -> str:
    """
    Browse/filter email classifications. Any argument left as "" is ignored.

    Args:
        category: Filter to a single category (maintenance, billing, etc.).
        urgency:  Filter to a single urgency level (low/normal/high/urgent).
        sentiment: Filter to positive/neutral/negative.
        requires_action: "true" or "false" to filter; "" means no filter.
        tag: Match a single tag (substring match against the JSON tags column).
        limit: Max rows to return (default 50).

    Returns:
        JSON array of classification records, newest first.
    """
    where: list[str] = []
    params: list[Any] = []
    if category:
        where.append("category = ?"); params.append(category)
    if urgency:
        where.append("urgency = ?"); params.append(urgency)
    if sentiment:
        where.append("sentiment = ?"); params.append(sentiment)
    if requires_action.lower() in ("true", "false"):
        where.append("requires_action = ?")
        params.append(1 if requires_action.lower() == "true" else 0)
    if tag:
        where.append("tags LIKE ?"); params.append(f'%"{tag}"%')

    sql = "SELECT * FROM email_classifications"
    if where:
        sql += " WHERE " + " AND ".join(where)
    sql += " ORDER BY classified_at DESC LIMIT ?"
    params.append(max(1, min(int(limit) if str(limit).isdigit() else 50, 500)))

    with get_conn() as conn:
        rows = conn.execute(sql, params).fetchall()
    return json.dumps([dict(r) for r in rows], default=str)


def get_email_classification(thread_id: str, message_id: str = "") -> str:
    """Return the classification for a specific (thread_id, message_id) as JSON, or empty object."""
    with get_conn() as conn:
        if message_id:
            row = conn.execute(
                "SELECT * FROM email_classifications WHERE thread_id=? AND message_id=?",
                (thread_id, message_id),
            ).fetchone()
        else:
            row = conn.execute(
                "SELECT * FROM email_classifications WHERE thread_id=? ORDER BY classified_at DESC LIMIT 1",
                (thread_id,),
            ).fetchone()
    return json.dumps(dict(row) if row else {}, default=str)


def classification_counts() -> str:
    """Return aggregate counts by category, urgency, and requires_action for browsing UIs."""
    with get_conn() as conn:
        by_category = {
            r["category"] or "uncategorised": r["c"]
            for r in conn.execute(
                "SELECT category, COUNT(*) AS c FROM email_classifications GROUP BY category"
            ).fetchall()
        }
        by_urgency = {
            r["urgency"] or "normal": r["c"]
            for r in conn.execute(
                "SELECT urgency, COUNT(*) AS c FROM email_classifications GROUP BY urgency"
            ).fetchall()
        }
        action_count = conn.execute(
            "SELECT COUNT(*) FROM email_classifications WHERE requires_action=1"
        ).fetchone()[0]
        total = conn.execute("SELECT COUNT(*) FROM email_classifications").fetchone()[0]
    return json.dumps({
        "total": total,
        "by_category": by_category,
        "by_urgency": by_urgency,
        "requires_action": action_count,
    })


# ── Summary ────────────────────────────────────────────────────────────────────

def get_db_summary() -> str:
    """Return counts and current property manager as JSON."""
    with get_conn() as conn:
        data = {
            "employees": conn.execute("SELECT COUNT(*) FROM employees").fetchone()[0],
            "property_managers": conn.execute("SELECT COUNT(*) FROM property_managers").fetchone()[0],
            "key_contacts": conn.execute("SELECT COUNT(*) FROM key_contacts").fetchone()[0],
            "open_tasks": conn.execute("SELECT COUNT(*) FROM tasks WHERE status='open'").fetchone()[0],
            "high_priority_tasks": conn.execute(
                "SELECT COUNT(*) FROM tasks WHERE priority='high' AND status='open'"
            ).fetchone()[0],
            "meetings": conn.execute("SELECT COUNT(*) FROM meetings").fetchone()[0],
            "classified_emails": conn.execute(
                "SELECT COUNT(*) FROM email_classifications"
            ).fetchone()[0],
            "urgent_classified_emails": conn.execute(
                "SELECT COUNT(*) FROM email_classifications WHERE urgency='urgent'"
            ).fetchone()[0],
        }
        pm = conn.execute(
            "SELECT name, email FROM property_managers WHERE status='current' LIMIT 1"
        ).fetchone()
        data["current_property_manager"] = dict(pm) if pm else None
    return json.dumps(data)
