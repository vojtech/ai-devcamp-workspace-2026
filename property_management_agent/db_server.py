"""
SQLite MCP server — acts as the database manager agent for property management data.
Launched as a stdio subprocess by the root agent via McpToolset.
"""
import json
import os
import sqlite3
import sys
from datetime import datetime

from mcp.server.fastmcp import FastMCP

logging_stream = sys.stderr
DB_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "property_management.db")

mcp = FastMCP("PropertyManagementDB")


def _get_conn() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def _init_db() -> None:
    with _get_conn() as conn:
        conn.executescript("""
            CREATE TABLE IF NOT EXISTS employees (
                id          INTEGER PRIMARY KEY AUTOINCREMENT,
                name        TEXT NOT NULL,
                email       TEXT UNIQUE NOT NULL,
                role        TEXT,
                department  TEXT,
                first_seen  TEXT DEFAULT (datetime('now')),
                last_seen   TEXT DEFAULT (datetime('now'))
            );

            CREATE TABLE IF NOT EXISTS property_managers (
                id          INTEGER PRIMARY KEY AUTOINCREMENT,
                name        TEXT NOT NULL,
                email       TEXT UNIQUE NOT NULL,
                status      TEXT NOT NULL CHECK(status IN ('current', 'previous')),
                notes       TEXT,
                updated_at  TEXT DEFAULT (datetime('now'))
            );

            CREATE TABLE IF NOT EXISTS key_contacts (
                id           INTEGER PRIMARY KEY AUTOINCREMENT,
                name         TEXT NOT NULL,
                email        TEXT,
                phone        TEXT,
                role         TEXT,
                contact_type TEXT,
                notes        TEXT,
                updated_at   TEXT DEFAULT (datetime('now'))
            );

            CREATE TABLE IF NOT EXISTS tasks (
                id           INTEGER PRIMARY KEY AUTOINCREMENT,
                title        TEXT NOT NULL,
                description  TEXT,
                status       TEXT DEFAULT 'open' CHECK(status IN ('open','in_progress','completed')),
                due_date     TEXT,
                assigned_to  TEXT,
                source_email TEXT,
                created_at   TEXT DEFAULT (datetime('now')),
                updated_at   TEXT DEFAULT (datetime('now'))
            );

            CREATE TABLE IF NOT EXISTS meetings (
                id           INTEGER PRIMARY KEY AUTOINCREMENT,
                title        TEXT NOT NULL,
                meeting_date TEXT,
                location     TEXT,
                attendees    TEXT,
                notes        TEXT,
                source_email TEXT,
                created_at   TEXT DEFAULT (datetime('now'))
            );
        """)


_init_db()


# ── Employees ──────────────────────────────────────────────────────────────────

@mcp.tool()
def upsert_employee(name: str, email: str, role: str = "", department: str = "") -> str:
    """Insert or update an employee. Returns the stored record as JSON."""
    now = datetime.utcnow().isoformat()
    with _get_conn() as conn:
        conn.execute("""
            INSERT INTO employees (name, email, role, department, first_seen, last_seen)
            VALUES (?, ?, ?, ?, ?, ?)
            ON CONFLICT(email) DO UPDATE SET
                name       = excluded.name,
                role       = COALESCE(NULLIF(excluded.role, ''), role),
                department = COALESCE(NULLIF(excluded.department, ''), department),
                last_seen  = excluded.last_seen
        """, (name, email, role, department, now, now))
    return json.dumps({"status": "ok", "email": email})


@mcp.tool()
def list_employees() -> str:
    """Return all known employees as a JSON array."""
    with _get_conn() as conn:
        rows = conn.execute("SELECT * FROM employees ORDER BY name").fetchall()
    return json.dumps([dict(r) for r in rows])


# ── Property Managers ──────────────────────────────────────────────────────────

@mcp.tool()
def upsert_property_manager(name: str, email: str, status: str, notes: str = "") -> str:
    """Insert or update a property manager. status must be 'current' or 'previous'."""
    if status not in ("current", "previous"):
        return json.dumps({"error": "status must be 'current' or 'previous'"})
    now = datetime.utcnow().isoformat()
    with _get_conn() as conn:
        conn.execute("""
            INSERT INTO property_managers (name, email, status, notes, updated_at)
            VALUES (?, ?, ?, ?, ?)
            ON CONFLICT(email) DO UPDATE SET
                name       = excluded.name,
                status     = excluded.status,
                notes      = COALESCE(NULLIF(excluded.notes, ''), notes),
                updated_at = excluded.updated_at
        """, (name, email, status, notes, now))
    return json.dumps({"status": "ok", "email": email, "manager_status": status})


@mcp.tool()
def list_property_managers() -> str:
    """Return all property managers (current and previous) as a JSON array."""
    with _get_conn() as conn:
        rows = conn.execute(
            "SELECT * FROM property_managers ORDER BY status DESC, name"
        ).fetchall()
    return json.dumps([dict(r) for r in rows])


# ── Key Contacts ───────────────────────────────────────────────────────────────

@mcp.tool()
def upsert_key_contact(
    name: str,
    role: str,
    contact_type: str,
    email: str = "",
    phone: str = "",
    notes: str = "",
) -> str:
    """Insert or update a key contact (e.g. maintenance coordinator, billing).
    contact_type examples: maintenance, billing, leasing, legal, emergency."""
    now = datetime.utcnow().isoformat()
    with _get_conn() as conn:
        existing = conn.execute(
            "SELECT id FROM key_contacts WHERE name = ? AND contact_type = ?",
            (name, contact_type),
        ).fetchone()
        if existing:
            conn.execute("""
                UPDATE key_contacts
                SET email=COALESCE(NULLIF(?,  ''), email),
                    phone=COALESCE(NULLIF(?,  ''), phone),
                    role =COALESCE(NULLIF(?,  ''), role),
                    notes=COALESCE(NULLIF(?,  ''), notes),
                    updated_at=?
                WHERE id=?
            """, (email, phone, role, notes, now, existing["id"]))
        else:
            conn.execute("""
                INSERT INTO key_contacts (name, email, phone, role, contact_type, notes, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (name, email, phone, role, contact_type, notes, now))
    return json.dumps({"status": "ok", "name": name, "contact_type": contact_type})


@mcp.tool()
def list_key_contacts() -> str:
    """Return all key contacts as a JSON array."""
    with _get_conn() as conn:
        rows = conn.execute(
            "SELECT * FROM key_contacts ORDER BY contact_type, name"
        ).fetchall()
    return json.dumps([dict(r) for r in rows])


# ── Tasks ──────────────────────────────────────────────────────────────────────

@mcp.tool()
def add_task(
    title: str,
    description: str = "",
    due_date: str = "",
    assigned_to: str = "",
    source_email: str = "",
) -> str:
    """Add a new task extracted from an email. Returns the new task id."""
    now = datetime.utcnow().isoformat()
    with _get_conn() as conn:
        cur = conn.execute("""
            INSERT INTO tasks (title, description, due_date, assigned_to, source_email, created_at, updated_at)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        """, (title, description, due_date, assigned_to, source_email, now, now))
    return json.dumps({"status": "ok", "task_id": cur.lastrowid})


@mcp.tool()
def update_task_status(task_id: int, status: str) -> str:
    """Update a task's status. status must be 'open', 'in_progress', or 'completed'."""
    if status not in ("open", "in_progress", "completed"):
        return json.dumps({"error": "invalid status"})
    now = datetime.utcnow().isoformat()
    with _get_conn() as conn:
        conn.execute(
            "UPDATE tasks SET status=?, updated_at=? WHERE id=?",
            (status, now, task_id),
        )
    return json.dumps({"status": "ok", "task_id": task_id})


@mcp.tool()
def list_tasks(status: str = "") -> str:
    """Return tasks as JSON. Optionally filter by status: open, in_progress, completed."""
    with _get_conn() as conn:
        if status:
            rows = conn.execute(
                "SELECT * FROM tasks WHERE status=? ORDER BY due_date, created_at",
                (status,),
            ).fetchall()
        else:
            rows = conn.execute(
                "SELECT * FROM tasks ORDER BY due_date, created_at"
            ).fetchall()
    return json.dumps([dict(r) for r in rows])


# ── Meetings ───────────────────────────────────────────────────────────────────

@mcp.tool()
def add_meeting(
    title: str,
    meeting_date: str = "",
    location: str = "",
    attendees: str = "",
    notes: str = "",
    source_email: str = "",
) -> str:
    """Add a meeting extracted from an email. attendees is a comma-separated string."""
    now = datetime.utcnow().isoformat()
    with _get_conn() as conn:
        cur = conn.execute("""
            INSERT INTO meetings (title, meeting_date, location, attendees, notes, source_email, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        """, (title, meeting_date, location, attendees, notes, source_email, now))
    return json.dumps({"status": "ok", "meeting_id": cur.lastrowid})


@mcp.tool()
def list_meetings() -> str:
    """Return all meetings as a JSON array, ordered by date."""
    with _get_conn() as conn:
        rows = conn.execute(
            "SELECT * FROM meetings ORDER BY meeting_date DESC, created_at DESC"
        ).fetchall()
    return json.dumps([dict(r) for r in rows])


# ── Summary ────────────────────────────────────────────────────────────────────

@mcp.tool()
def get_summary() -> str:
    """Return a high-level summary: counts for each table and current property manager."""
    with _get_conn() as conn:
        counts = {
            "employees": conn.execute("SELECT COUNT(*) FROM employees").fetchone()[0],
            "property_managers": conn.execute("SELECT COUNT(*) FROM property_managers").fetchone()[0],
            "key_contacts": conn.execute("SELECT COUNT(*) FROM key_contacts").fetchone()[0],
            "open_tasks": conn.execute("SELECT COUNT(*) FROM tasks WHERE status='open'").fetchone()[0],
            "meetings": conn.execute("SELECT COUNT(*) FROM meetings").fetchone()[0],
        }
        current_pm = conn.execute(
            "SELECT name, email FROM property_managers WHERE status='current' LIMIT 1"
        ).fetchone()
        counts["current_property_manager"] = dict(current_pm) if current_pm else None
    return json.dumps(counts)


if __name__ == "__main__":
    mcp.run(transport="stdio")
